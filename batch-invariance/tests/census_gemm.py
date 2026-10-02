"""Batch-size (M) invariance census of the dense bf16 GEMMs of Qwen3.8-Flash-Next, as vLLM runs them (F.linear -> cuBLAS).
Same first 2032 input rows, M in {2032, 2040, 2048}; compare output rows < 2032 bitwise. Reports the fraction of differing
elements and the max difference in bf16 ulps, with torch's default allow_bf16_reduced_precision_reduction=True and False.
usage: python3 census_gemm.py [out.json]   (run in the serving image on one GPU)
"""
import json, sys, torch
import torch.nn.functional as F
torch.manual_seed(0)
dev = "cuda"
SHAPES = {  # name: (K, N)
    "gdn in_proj_qkvz": (2560, 16384), "gdn in_proj_ba": (2560, 96), "gdn out_proj": (6144, 2560),
    "moe router gate": (2560, 512), "shared expert gate_up": (2560, 1280), "shared expert down": (640, 2560),
    "shared_expert_gate": (2560, 1), "hc down+inject": (10240, 336), "hc up": (320, 10240),
    "ple key_proj": (2560, 10240), "attn q_proj": (2560, 12288), "attn k/v_proj": (2560, 512),
    "qsa index_qk_proj": (2560, 640), "attn o_proj": (6144, 2560), "lm_head": (2560, 248320),
}
MS = [2032, 2040, 2048]
NCMP = 2032


def ulps(a, b):
    ai, bi = a.view(torch.int16).to(torch.int32), b.view(torch.int16).to(torch.int32)
    # map bf16 bit patterns to a monotonic integer line
    ai = torch.where(ai < 0, -32768 - ai, ai); bi = torch.where(bi < 0, -32768 - bi, bi)
    return (ai - bi).abs()


res = {"device": torch.cuda.get_device_name(), "sm": torch.cuda.get_device_capability()}
for flag in (True, False):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
    out = {}
    for name, (K, N) in SHAPES.items():
        W = (torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16)
        X = torch.randn(max(MS), K, device=dev).to(torch.bfloat16)
        ys = {M: F.linear(X[:M], W)[:NCMP] for M in MS}
        r = {}
        for M in MS[1:]:
            d = ulps(ys[MS[0]], ys[M])
            r[f"M{MS[0]}vsM{M}"] = {"frac_elems_differ": round(float((d > 0).float().mean()), 6), "max_ulps": int(d.max()),
                                     "frac_rows_differ": round(float((d > 0).any(1).float().mean()), 4)}
        rep = F.linear(X[:MS[0]], W)[:NCMP]
        r["repeat_same_M_identical"] = bool(torch.equal(rep, ys[MS[0]]))
        out[name] = r
        del W, X, ys
        torch.cuda.empty_cache()
    res[f"allow_bf16_reduced_precision_reduction={flag}"] = out
print(json.dumps(res, indent=1))
if len(sys.argv) > 1:
    json.dump(res, open(sys.argv[1], "w"), indent=1)
