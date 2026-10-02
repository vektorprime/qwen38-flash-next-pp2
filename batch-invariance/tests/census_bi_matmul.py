"""vLLM batch-invariant matmul (Triton persistent, SM80 path) vs cuBLAS: invariance across M and speed, model shapes."""
import json, sys, time, torch
import torch.nn.functional as F
torch.manual_seed(0)
dev = "cuda"
SHAPES = {"gdn in_proj_qkvz": (2560, 16384), "gdn in_proj_ba": (2560, 96), "gdn out_proj": (6144, 2560), "moe router gate": (2560, 512),
          "shared expert gate_up": (2560, 1280), "shared expert down": (640, 2560), "shared_expert_gate": (2560, 1), "hc down+inject": (10240, 336),
          "hc up": (320, 10240), "attn q_proj": (2560, 12288), "lm_head": (2560, 248320)}
MS = [1, 4, 8, 16, 32, 64, 256, 1024, 2032, 2048]
def bench(f, n):
    for _ in range(3): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e6
res = {"device": torch.cuda.get_device_name()}
Ws = {n: ((torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16), torch.randn(2048, K, device=dev).to(torch.bfloat16)) for n, (K, N) in SHAPES.items()}
cublas = {n: {M: F.linear(X[:M], W) for M in MS} for n, (W, X) in Ws.items()}
tc = {n: {M: bench(lambda: F.linear(X[:M], W), 50) for M in (4, 32, 64, 2048)} for n, (W, X) in Ws.items()}
from vllm.model_executor.layers import batch_invariant as bi
bi.enable_batch_invariant_mode()
for name, (W, X) in Ws.items():
    outs = {M: F.linear(X[:M], W) for M in MS}
    groups = []
    for M in MS:
        for g in groups:
            if torch.equal(outs[g[0]][:1], outs[M][:1]): g.append(M); break
        else: groups.append([M])
    tb = {M: bench(lambda: F.linear(X[:M], W), 50) for M in (4, 32, 64, 2048)}
    rel = float(((outs[2048].float() - cublas[name][2048].float()).norm() / cublas[name][2048].float().norm()))
    res[name] = {"row0_groups_by_M": groups, "us_cublas": {M: round(v, 1) for M, v in tc[name].items()}, "us_batchinv": {M: round(v, 1) for M, v in tb.items()}, "rel_diff_vs_cublas": rel}
    print(f"{name:22s} groups {groups}  us cublas {res[name]['us_cublas']}  us inv {res[name]['us_batchinv']}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
