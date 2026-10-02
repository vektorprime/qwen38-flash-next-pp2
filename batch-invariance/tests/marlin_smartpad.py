"""Exact batch-invariant Marlin MoE with minimal padding ("smart pad").
Marlin's scheduler (marlin_template.h): global = blocks * n_tiles output tiles; if global > grid: part2 = global % grid
(+grid if <= grid/3) tiles are stream-K split, else all global tiles are split with iters = ceil(k_tiles*global/grid).
If global is a multiple of grid, part2 = grid and iters = k_tiles: every threadblock gets exactly one whole tile -> no K
split anywhere -> per-row result independent of the batch. So: pin blocks_per_sm=1 (grid = SMs) and thread_n, and pad the
number of expert blocks on the device to a multiple of L = lcm(grid/gcd(grid,n13), grid/gcd(grid,n2)). Graph-safe.
Tests: invariance across batch sizes/compositions; whether the thread config itself changes per-row results; timing.
"""
import json, math, sys, time, os, torch
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
from vllm import _custom_ops as ops
torch.manual_seed(0)
dev = "cuda"; E, H, I, TOPK = 512, 2560, 640, 10; qt = scalar_types.uint8b128
SMS = torch.cuda.get_device_properties(0).multi_processor_count
def pack(K, N):
    qws, ss = [], []
    for e in range(E):
        w = (torch.randn(K, N, device=dev) / K ** 0.5).to(torch.bfloat16)
        _, qw, s, _, _, _ = marlin_quantize(w, qt, -1, False); qws.append(qw); ss.append(s)
    return torch.stack(qws), torch.stack(ss)
w1, s1 = pack(H, 2 * I); w2, s2 = pack(I, H)
NMAX = 4096
X = torch.randn(NMAX, H, device=dev).to(torch.bfloat16)
tw, ti = torch.topk(torch.softmax(torch.randn(NMAX, E, device=dev), -1), TOPK, -1)
tw = (tw / tw.sum(-1, keepdim=True)).float(); ti = ti.to(torch.int32)
orig_gemm, orig_align = ops.moe_wna16_marlin_gemm, mm.moe_align_block_size

def lcm(a, b): return a * b // math.gcd(a, b)

class SmartPad:
    def __init__(self, cfg_small, cfg_large):
        self.cfg_small, self.cfg_large = cfg_small, cfg_large   # exec configs for thread_m_blocks == 1 / > 1
    def L(self, tn):
        g = SMS  # blocks_per_sm = 1
        return lcm(g // math.gcd(g, 1280 // tn), g // math.gcd(g, 2560 // tn))
    def align(self, topk_ids, block_size, num_experts, expert_map=None, *a, **k):
        sid, eid, ntpp = orig_align(topk_ids, block_size, num_experts, expert_map, *a, **k)
        cfg = self.cfg_small if block_size <= 16 else self.cfg_large
        L = self.L(cfg["thread_n"]); Pmax = L - 1
        if Pmax == 0:
            return sid, eid, ntpp
        sentinel = topk_ids.numel()
        sid = torch.cat([sid, sid.new_full((Pmax * block_size,), sentinel)]); eid = torch.cat([eid, eid.new_zeros(Pmax)])
        pos = ntpp.to(torch.int64) + torch.arange(Pmax * block_size, device=dev); sid.scatter_(0, pos, sid.new_full((Pmax * block_size,), sentinel))
        B = ntpp.to(torch.int64) // block_size
        bpos = B + torch.arange(Pmax, device=dev); eid.scatter_(0, bpos, eid.new_zeros(Pmax))
        P = (L - B % L) % L
        return sid, eid, (ntpp.to(torch.int64) + P * block_size).to(ntpp.dtype)
    def gemm(self, *a, **k):
        cfg = self.cfg_small if k["moe_block_size"] <= 16 else self.cfg_large
        return orig_gemm(*a, **{**k, **cfg, "blocks_per_sm": 1})

def run(idx, sp):
    if sp: mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = sp.align, sp.gemm
    try:
        return mm.fused_marlin_moe(X[idx].contiguous(), w1, w2, None, None, s1, s2, tw[idx].contiguous(), ti[idx].contiguous(), quant_type_id=qt.id, global_num_experts=E)
    finally:
        mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = orig_align, orig_gemm
ar = lambda a, b: torch.arange(a, b, device=dev)

def invariance(sp):
    probe4 = ar(0, 4); ref4 = run(probe4, sp); bad4 = 0
    for extra in (0, 4, 12, 28, 60, 124, 252, 508, 1020, 2040):
        bad4 += int((run(torch.cat([probe4, ar(2048, 2048 + extra)]), sp)[:4] != ref4).any(1).sum())
    probe = ar(0, 1000); ref = run(probe, sp); bad = 0
    for extra in (1, 8, 40, 500, 1048):
        bad += int((run(torch.cat([probe, ar(2048, 2048 + extra)]), sp)[:1000] != ref).any(1).sum())
    base = run(ar(0, 2032), sp); badp = sum(int((run(ar(0, M), sp)[:2032] != base).any(1).sum()) for M in (2040, 2048))
    return {"4 tok x10 batches (of 40)": bad4, "1000 tok x5 batches (of 5000)": bad, "prefill 2032 vs 2040/2048 (of 4064)": badp}

def timing(sp):
    t = {}
    for M in (1, 2, 4, 8, 16, 32, 64, 256, 2040):
        idx = ar(0, M)
        for _ in range(3): run(idx, sp)
        n = 20 if M > 256 else 100
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): run(idx, sp)
        torch.cuda.synchronize(); t[M] = round((time.perf_counter() - t0) / n * 1e3, 3)
    return t

res = {"device": torch.cuda.get_device_name(), "sms": SMS}
res["stock"] = {"invariance": invariance(None), "time_ms": timing(None)}
print("stock", json.dumps(res["stock"]), flush=True)
CANDS = {"small(128,128)+large(64,256)": ({"thread_k": 128, "thread_n": 128}, {"thread_k": 64, "thread_n": 256}),
         "small(64,128)+large(64,128)": ({"thread_k": 64, "thread_n": 128}, {"thread_k": 64, "thread_n": 128}),
         "small(128,128)+large(64,128)": ({"thread_k": 128, "thread_n": 128}, {"thread_k": 64, "thread_n": 128})}
for name, (cs, cl) in CANDS.items():
    sp = SmartPad(cs, cl)
    try:
        r = {"L_small": sp.L(cs["thread_n"]), "L_large": sp.L(cl["thread_n"]), "invariance": invariance(sp), "time_ms": timing(sp)}
    except Exception as e:  # noqa
        r = {"error": repr(e)[:200]}
    res[name] = r; print(name, json.dumps(r), flush=True)
# does the thread config change per-row results when nothing is split? same 1000 tokens, padded, different configs
outs = {}
for cfg in ({"thread_k": 64, "thread_n": 128}, {"thread_k": 64, "thread_n": 256}, {"thread_k": 128, "thread_n": 64}):
    sp = SmartPad(cfg, cfg)
    try:
        outs[f"{cfg['thread_k']},{cfg['thread_n']}"] = run(ar(0, 1000), sp)
    except Exception as e:  # noqa
        print("cfg", cfg, "error", repr(e)[:120])
ks = list(outs)
res["config_dependence_rows_differ_of_1000"] = {f"{a} vs {b}": int((outs[a] != outs[b]).any(1).sum()) for i, a in enumerate(ks) for b in ks[i + 1:]}
print("config dependence", res["config_dependence_rows_differ_of_1000"], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
