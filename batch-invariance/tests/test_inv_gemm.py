"""Offline test of hook3/inv_gemm.py: invariance, correctness vs cuBLAS, torch.compile (dynamic shapes) + CUDA graph capture."""
import sys, time, torch
sys.path.insert(0, "/w/hook3")
import inv_gemm
torch.manual_seed(0)
dev = "cuda"
W = (torch.randn(16384, 2560, device=dev) / 50).to(torch.bfloat16); X = torch.randn(2048, 2560, device=dev).to(torch.bfloat16)
ref = inv_gemm.linear(X[:4], W)
print("invariant across M:", all(torch.equal(inv_gemm.linear(X[:M], W)[:4], ref) for M in (4, 5, 16, 17, 64, 65, 256, 257, 1000, 2048)))
rel = ((inv_gemm.linear(X, W).float() - torch.nn.functional.linear(X, W).float()).norm() / torch.nn.functional.linear(X, W).float().norm()).item()
print("rel diff vs cuBLAS:", rel)
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w1 = torch.nn.Parameter(W, requires_grad=False); self.w2 = torch.nn.Parameter((torch.randn(2560, 16384, device=dev) / 128).to(torch.bfloat16), requires_grad=False)
    def forward(self, x):
        return inv_gemm.linear(torch.nn.functional.silu(inv_gemm.linear(x, self.w1)), self.w2)
torch.set_grad_enabled(False)
m = Toy()
eager = {M: m(X[:M]) for M in (4, 2048)}
cm = torch.compile(m, dynamic=True)
comp = {M: cm(X[:M]) for M in (4, 37, 2048)}
print("compiled == eager (M=4, 2048):", torch.equal(comp[4], eager[4]), torch.equal(comp[2048], eager[2048]), "| compiled row-invariant:", torch.equal(comp[37][:4], comp[4]))
xs = X[:8].clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): cm(xs)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = cm(xs)
xs.copy_(X[:8]); g.replay(); torch.cuda.synchronize()
print("CUDA graph replay == compiled:", torch.equal(out[:4], comp[4]))
