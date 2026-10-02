"""Is cuBLAS(Lt) batch-invariant on Ampere once split-K is disabled (no workspace), as vLLM assumes for SM90+?
Run with CUBLAS_WORKSPACE_CONFIG=:16:8 CUBLASLT_WORKSPACE_SIZE=1 (set before CUDA init). Row-0 groups by M + timings."""
import json, os, sys, time, torch
import torch.nn.functional as F
torch.manual_seed(0)
dev = "cuda"
if os.environ.get("LT") == "1":
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = os.environ.get("RPR", "1") == "1"
SHAPES = {"gdn in_proj_qkvz": (2560, 16384), "gdn in_proj_ba": (2560, 96), "gdn out_proj": (6144, 2560), "moe router gate": (2560, 512),
          "shared expert gate_up": (2560, 1280), "shared expert down": (640, 2560), "shared_expert_gate": (2560, 1), "hc down+inject": (10240, 336),
          "hc up": (320, 10240), "ple key_proj": (2560, 10240), "attn q_proj": (2560, 12288), "attn k/v_proj": (2560, 512), "qsa index_qk_proj": (2560, 640),
          "attn o_proj": (6144, 2560), "lm_head": (2560, 248320), "mtp fc": (2560, 2560)}
MS = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, 256, 1024, 2032, 2048]
def bench(f, n=50):
    for _ in range(3): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return round((time.perf_counter() - t0) / n * 1e6, 1)
res = {"env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "LT", "RPR")}}
ng = 0
for name, (K, N) in SHAPES.items():
    W = (torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16); X = torch.randn(2048, K, device=dev).to(torch.bfloat16)
    groups = []
    for M in MS:
        y = F.linear(X[:M], W)[:1]
        for g in groups:
            if torch.equal(g[0], y): g[1].append(M); break
        else: groups.append((y, [M]))
    gl = [g[1] for g in groups]; ng += len(gl) - 1
    res[name] = {"groups": gl, "us": {M: bench(lambda: F.linear(X[:M], W)) for M in (4, 32, 2048)}}
    print(f"{name:22s} {gl}  us {res[name]['us']}", flush=True)
print("TOTAL extra groups (0 = fully invariant):", ng)
json.dump(res, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
