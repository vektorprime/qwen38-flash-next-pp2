"""In-window cost/invariance bench on a CMP 170HX (GA100): (1) Marlin MoE stock vs invariant (hook3 rule), timed with
CUDA graphs at decode sizes and eagerly at prefill; (2) invariant Triton GEMM vs cuBLAS for the model's dense shapes,
CUDA-graph timed at M=4/32 and eager at M=2048, plus a cross-config invariance check on sm80."""
import json, math, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, "/w/hook3")
import inv_gemm
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
from vllm import _custom_ops as ops
torch.manual_seed(0); dev = "cuda"
SMS = torch.cuda.get_device_properties(0).multi_processor_count
res = {"device": torch.cuda.get_device_name(), "sms": SMS}
def gtime(fn, n=50):  # CUDA-graph timing (no launch overhead), us per call
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); return round((time.perf_counter() - t0) / (5 * n) * 1e6, 1)
def etime(fn, n=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return round((time.perf_counter() - t0) / n * 1e6, 1)
# ---------------- Marlin MoE
E, H, I, qt = 512, 2560, 640, scalar_types.uint8b128
def pack(K, N):
    a, b = [], []
    for e in range(E):
        _, qw, s, _, _, _ = marlin_quantize((torch.randn(K, N, device=dev) / K ** 0.5).to(torch.bfloat16), qt, -1, False); a.append(qw); b.append(s)
    return torch.stack(a), torch.stack(b)
w1, s1 = pack(H, 2 * I); w2, s2 = pack(I, H)
X = torch.randn(4200, H, device=dev).to(torch.bfloat16)
tw, ti = torch.topk(torch.softmax(torch.randn(4200, E, device=dev), -1), 10, -1); tw = (tw / tw.sum(-1, keepdim=True)).float(); ti = ti.to(torch.int32)
orig_align, orig_gemm = mm.moe_align_block_size, ops.moe_wna16_marlin_gemm
CS, CL = {"thread_k": 64, "thread_n": 128, "blocks_per_sm": 1}, {"thread_k": 64, "thread_n": 256, "blocks_per_sm": 1}
def L_for(tn): return math.lcm(SMS // math.gcd(SMS, 1280 // tn), SMS // math.gcd(SMS, 2560 // tn))
def inv_align(topk_ids, bs, ne, em=None, *a, **k):
    sid, eid, ntpp = orig_align(topk_ids, bs, ne, em, *a, **k)
    L = L_for((CS if bs <= 16 else CL)["thread_n"]); pmax = L - 1
    if pmax == 0: return sid, eid, ntpp
    sen = topk_ids.numel(); n64 = ntpp.to(torch.int64)
    sid = torch.cat([sid, sid.new_full((pmax * bs,), sen)]); eid = torch.cat([eid, eid.new_zeros(pmax)])
    sid.scatter_(0, n64 + torch.arange(pmax * bs, device=dev), sid.new_full((pmax * bs,), sen))
    B = n64 // bs; eid.scatter_(0, B + torch.arange(pmax, device=dev), eid.new_zeros(pmax))
    return sid, eid, (n64 + ((L - B % L) % L) * bs).to(ntpp.dtype)
def inv_gemm_marlin(*a, **k): return orig_gemm(*a, **{**k, **(CS if k["moe_block_size"] <= 16 else CL)})
def moe(idx, inv):
    if inv: mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = inv_align, inv_gemm_marlin
    try:
        return mm.fused_marlin_moe(X[idx], w1, w2, None, None, s1, s2, tw[idx], ti[idx], quant_type_id=qt.id, global_num_experts=E)
    finally:
        mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = orig_align, orig_gemm
ar = lambda a, b: torch.arange(a, b, device=dev)
r = {"L_small": L_for(128), "L_large": L_for(256)}
for inv in (False, True):
    ref4, ref1k = moe(ar(0, 4), inv), moe(ar(0, 1000), inv)
    bad = sum(int((moe(torch.cat([ar(0, 4), ar(2048, 2048 + x)]), inv)[:4] != ref4).any(1).sum()) for x in (0, 4, 12, 28, 60, 124, 252, 1020, 2040))
    bad += sum(int((moe(torch.cat([ar(0, 1000), ar(1100, 1100 + x)]), inv)[:1000] != ref1k).any(1).sum()) for x in (1, 8, 40, 1000))
    t = {}
    for M in (1, 4, 8, 16, 32):
        idx = ar(0, M); t[M] = gtime(lambda: moe(idx, inv), 20)
    idx = ar(0, 2040); t[2040] = etime(lambda: moe(idx, inv))
    r["invariant" if inv else "stock"] = {"rows_differing_across_batches": bad, "us_graph_M1_32__eager_M2040": t}
    print("marlin", "invariant" if inv else "stock", r["invariant" if inv else "stock"], flush=True)
res["marlin_moe"] = r
del w1, w2; torch.cuda.empty_cache()
# ---------------- dense GEMM
SHAPES = {"gdn in_proj_qkvz": (2560, 16384), "gdn in_proj_ba": (2560, 96), "gdn out_proj": (6144, 2560), "moe router gate": (2560, 512),
          "shared expert gate_up": (2560, 1280), "shared expert down": (640, 2560), "shared_expert_gate": (2560, 1), "hc down+inject": (10240, 336),
          "hc up": (320, 10240), "ple key_proj": (2560, 10240), "attn q_proj": (2560, 12288), "attn k/v_proj": (2560, 512),
          "qsa index_qk_proj": (2560, 640), "attn o_proj": (6144, 2560), "lm_head": (2560, 248320), "mtp fc": (2560, 2560)}
COUNT = {"gdn in_proj_qkvz": 36, "gdn in_proj_ba": 36, "gdn out_proj": 36, "moe router gate": 49, "shared expert gate_up": 49, "shared expert down": 49,
         "shared_expert_gate": 49, "hc down+inject": 98, "hc up": 98, "ple key_proj": 1, "attn q_proj": 13, "attn k/v_proj": 26, "qsa index_qk_proj": 13,
         "attn o_proj": 13, "lm_head": 2, "mtp fc": 2}
g = {}; tot = {"cublas_M4": 0, "inv_M4": 0, "cublas_M32": 0, "inv_M32": 0, "cublas_M2048": 0, "inv_M2048": 0}; bad = 0
for name, (K, N) in SHAPES.items():
    Wt = (torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16); Xg = torch.randn(2048, K, device=dev).to(torch.bfloat16)
    ref = inv_gemm.linear(Xg[:4], Wt)
    bad += sum(int(not torch.equal(inv_gemm.linear(Xg[:M], Wt)[:4], ref)) for M in (4, 5, 17, 64, 65, 300, 2048))
    x4, x32, x2k = Xg[:4].clone(), Xg[:32].clone(), Xg
    row = {"cublas_M4": gtime(lambda: F.linear(x4, Wt)), "inv_M4": gtime(lambda: inv_gemm.linear(x4, Wt)),
           "cublas_M32": gtime(lambda: F.linear(x32, Wt)), "inv_M32": gtime(lambda: inv_gemm.linear(x32, Wt)),
           "cublas_M2048": etime(lambda: F.linear(x2k, Wt)), "inv_M2048": etime(lambda: inv_gemm.linear(x2k, Wt))}
    g[name] = row
    for k in tot: tot[k] += row[k] * COUNT[name]
    print(f"gemm {name:22s} {row}", flush=True)
    del Wt, Xg; torch.cuda.empty_cache()
res["dense_gemm"] = {"per_shape_us": g, "invariance_failures": bad, "per_forward_total_us (x layer counts)": {k: round(v) for k, v in tot.items()}}
print("gemm totals per forward (us):", res["dense_gemm"]["per_forward_total_us (x layer counts)"], "invariance failures:", bad, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
