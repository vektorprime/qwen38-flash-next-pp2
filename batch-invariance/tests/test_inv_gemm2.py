import sys, time, torch
sys.path.insert(0, "/w/hook3"); import inv_gemm
import torch.nn.functional as F
torch.manual_seed(0); dev = "cuda"
def gtime(fn, n=50):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); return round((time.perf_counter() - t0) / (5 * n) * 1e6, 1)
for name, (K, N) in {"gdn in_proj_ba": (2560, 96), "moe router gate": (2560, 512), "shared_expert_gate": (2560, 1), "hc down+inject": (10240, 336),
                     "attn k/v_proj": (2560, 512), "qsa index_qk_proj": (2560, 640), "hc up": (320, 10240), "gdn out_proj": (6144, 2560)}.items():
    W = (torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16); X = torch.randn(2048, K, device=dev).to(torch.bfloat16)
    ref = inv_gemm.linear(X[:4], W)
    inv = all(torch.equal(inv_gemm.linear(X[:M], W)[:4], ref) for M in (4, 5, 16, 17, 33, 64, 65, 300, 2048))
    rel = float((inv_gemm.linear(X, W).float() - F.linear(X, W).float()).norm() / F.linear(X, W).float().norm())
    x4, x32 = X[:4].clone(), X[:32].clone()
    print(f"{name:20s} splits {inv_gemm._splits(N, K):2d} invariant {inv}  rel {rel:.1e}  M4 cub {gtime(lambda: F.linear(x4, W)):6.1f} inv {gtime(lambda: inv_gemm.linear(x4, W)):6.1f} | M32 cub {gtime(lambda: F.linear(x32, W)):6.1f} inv {gtime(lambda: inv_gemm.linear(x32, W)):6.1f}", flush=True)
