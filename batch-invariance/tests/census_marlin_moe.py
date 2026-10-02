"""Batch-size (M) invariance of vLLM's Marlin MoE (W8A16, INT8 per-channel, uint8b128) at Qwen3.8-Flash-Next shapes.
512 experts, hidden 2560, intermediate 640, top-10. Same first 2032 tokens (inputs + routing), M in {2032, 2040, 2048}.
Compares output rows < 2032 bitwise. Also checks the shared-expert-free routed output only (that is what Marlin computes).
"""
import json, sys, torch
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
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


w1, s1 = pack(H, 2 * I)   # gate_up: [K=2560, N=1280]
w2, s2 = pack(I, H)       # down:    [K=640,  N=2560]
print("packed", w1.shape, s1.shape, w2.shape, s2.shape, flush=True)
X = torch.randn(max(MS), H, device=dev).to(torch.bfloat16)
logits = torch.randn(max(MS), E, device=dev)
tw, ti = torch.topk(torch.softmax(logits, -1), TOPK, -1)
tw = (tw / tw.sum(-1, keepdim=True)).float()
ti = ti.to(torch.int32)


def run(M):
    return fused_marlin_moe(X[:M].clone(), w1, w2, None, None, s1, s2, tw[:M].clone(), ti[:M].clone(), quant_type_id=qt.id,
                            global_num_experts=E)[:NCMP].clone()


def ulps(a, b):
    ai, bi = a.view(torch.int16).to(torch.int32), b.view(torch.int16).to(torch.int32)
    ai = torch.where(ai < 0, -32768 - ai, ai); bi = torch.where(bi < 0, -32768 - bi, bi)
    return (ai - bi).abs()


ys = {M: run(M) for M in MS}
res = {"device": torch.cuda.get_device_name(), "repeat_same_M_identical": bool(torch.equal(run(MS[0]), ys[MS[0]]))}
for M in MS[1:]:
    d = ulps(ys[MS[0]], ys[M])
    rel = ((ys[MS[0]].float() - ys[M].float()).norm(dim=1) / ys[MS[0]].float().norm(dim=1))
    res[f"M{MS[0]}vsM{M}"] = {"frac_elems_differ": round(float((d > 0).float().mean()), 5), "frac_rows_differ": round(float((d > 0).any(1).float().mean()), 4),
                             "max_ulps": int(d.max()), "median_ulps_of_differing": float(d[d > 0].float().median()) if (d > 0).any() else 0.0,
                             "rel_l2_row_diff_mean": float(rel.mean()), "rel_l2_row_diff_max": float(rel.max())}
# rounding reference: the same computation with the inputs perturbed by 1 bf16 ulp in 0.1% of elements (what a "rounding-level" change looks like)
print(json.dumps(res, indent=1))
if len(sys.argv) > 1:
    json.dump(res, open(sys.argv[1], "w"), indent=1)
