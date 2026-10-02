"""Which Marlin MoE launch parameter makes the per-row result depend on M?  Patches ops.moe_wna16_marlin_gemm as called by
fused_marlin_moe to force thread_k / thread_n / blocks_per_sm, and optionally pads sorted_token_ids to a fixed allocation.
Same setup as census_marlin_moe.py (512 INT8 experts, 2560/640, top-10), M in {2032, 2040, 2048}, rows < 2032 compared.
"""
import json, sys, itertools, torch
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
from vllm import _custom_ops as ops
torch.manual_seed(0)
dev = "cuda"
E, H, I, TOPK = 512, 2560, 640, 10
MS = [2032, 2040, 2048]
NCMP = 2032
qt = scalar_types.uint8b128


def pack(K, N):
    qws, ss = [], []
    for e in range(E):
        w = (torch.randn(K, N, device=dev) / K ** 0.5).to(torch.bfloat16)
        _, qw, s, _, _, _ = marlin_quantize(w, qt, -1, False)
        qws.append(qw); ss.append(s)
    return torch.stack(qws), torch.stack(ss)


w1, s1 = pack(H, 2 * I)
w2, s2 = pack(I, H)
X = torch.randn(max(MS), H, device=dev).to(torch.bfloat16)
tw, ti = torch.topk(torch.softmax(torch.randn(max(MS), E, device=dev), -1), TOPK, -1)
tw = (tw / tw.sum(-1, keepdim=True)).float(); ti = ti.to(torch.int32)
orig = ops.moe_wna16_marlin_gemm
calls = []


def make_patch(kw, record):
    def patched(*a, **k):
        k.update(kw)
        if record is not None:
            record.append({"size_m": k.get("size_m"), "size_n": k.get("size_n"), "size_k": k.get("size_k"), "moe_block_size": k.get("moe_block_size"),
                           "sorted_len": int(a[11].shape[0]) if len(a) > 11 else None})
        return orig(*a, **k)
    return patched


def run(M, kw, record=None):
    mm.ops.moe_wna16_marlin_gemm = make_patch(kw, record)
    try:
        return mm.fused_marlin_moe(X[:M].clone(), w1, w2, None, None, s1, s2, tw[:M].clone(), ti[:M].clone(), quant_type_id=qt.id, global_num_experts=E)[:NCMP].clone()
    finally:
        mm.ops.moe_wna16_marlin_gemm = orig


def stat(kw):
    try:
        ys = {M: run(M, kw) for M in MS}
    except Exception as e:  # noqa
        return {"error": repr(e)[:160]}
    out = {}
    for M in MS[1:]:
        d = (ys[MS[0]] != ys[M])
        out[f"vsM{M}"] = {"elems": round(float(d.float().mean()), 6), "rows": round(float(d.any(1).float().mean()), 4)}
    return out


rec = []
run(2032, {}, rec); run(2040, {}, rec)
print("default launch args per GEMM (M=2032, then M=2040):", json.dumps(rec), flush=True)
res = {"default": stat({})}
for bps in (1, 2):
    res[f"blocks_per_sm={bps}"] = stat({"blocks_per_sm": bps})
for tk, tn in ((64, 256), (128, 128), (128, 64), (64, 128)):
    res[f"thread_k={tk},thread_n={tn}"] = stat({"thread_k": tk, "thread_n": tn})
    res[f"thread_k={tk},thread_n={tn},blocks_per_sm=1"] = stat({"thread_k": tk, "thread_n": tn, "blocks_per_sm": 1})
res["use_fp32_reduce=False"] = stat({"use_fp32_reduce": False})
for k, v in res.items():
    print(f"{k:42s} {json.dumps(v)}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
