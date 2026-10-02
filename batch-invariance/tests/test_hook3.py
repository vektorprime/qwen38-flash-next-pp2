"""Offline test of hook3 (run with PYTHONPATH=/w/hook3 and PLEFP8_INV_MOE/GEMM/QSA=1 set at startup)."""
import torch, math
torch.manual_seed(0); dev = "cuda"
# --- patches applied at import ---
import vllm.model_executor.layers.utils as lu
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
import vllm.models.qwen4_exp.nvidia.ops.qsa as qm
from vllm import _custom_ops as ops
print("gemm patched:", lu.default_unquantized_gemm.__module__ == "sitecustomize" or "inv" in lu.default_unquantized_gemm.__code__.co_filename,
      "| marlin align patched:", mm.moe_align_block_size.__name__ == "align", "| qsa patched:", hasattr(qm.qsa_sparse_paged_attention, "_plefp8_orig"))
# --- dense GEMM via vLLM's dispatch ---
W = (torch.randn(1280, 2560, device=dev) / 50).to(torch.bfloat16); X = torch.randn(300, 2560, device=dev).to(torch.bfloat16)
g = lu.dispatch_unquantized_gemm()
print("dispatch GEMM row-invariant:", all(torch.equal(g(None, X[:M], W)[:3], g(None, X[:3], W)) for M in (3, 8, 64, 300)))
# --- Marlin MoE through the patched module ---
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
qt = scalar_types.uint8b128; E, H, I = 512, 2560, 640
def pack(K, N):
    a, b = [], []
    for e in range(E):
        _, qw, s, _, _, _ = marlin_quantize((torch.randn(K, N, device=dev) / K ** 0.5).to(torch.bfloat16), qt, -1, False); a.append(qw); b.append(s)
    return torch.stack(a), torch.stack(b)
w1, s1 = pack(H, 2 * I); w2, s2 = pack(I, H)
Xm = torch.randn(4200, H, device=dev).to(torch.bfloat16)
tw, ti = torch.topk(torch.softmax(torch.randn(4200, E, device=dev), -1), 10, -1); tw = (tw / tw.sum(-1, keepdim=True)).float(); ti = ti.to(torch.int32)
run = lambda idx: mm.fused_marlin_moe(Xm[idx].contiguous(), w1, w2, None, None, s1, s2, tw[idx].contiguous(), ti[idx].contiguous(), quant_type_id=qt.id, global_num_experts=E)
ar = lambda a, b: torch.arange(a, b, device=dev)
ref4, ref1k = run(ar(0, 4)), run(ar(0, 1000))
bad = sum(int((run(torch.cat([ar(0, 4), ar(2048, 2048 + x)]))[:4] != ref4).any(1).sum()) for x in (0, 4, 28, 124, 252, 2040))
bad += sum(int((run(torch.cat([ar(0, 1000), ar(1100, 1100 + x)]))[:1000] != ref1k).any(1).sum()) for x in (1, 8, 40, 1000))
print("Marlin via hook: rows differing across batch compositions:", bad, "(0 = invariant)")
# --- QSA pinned profile: invariance across batch sizes + closeness to stock ---
Hq, Hkv, D, PAGE, CTX = 24, 2, 256, 64, 1500
nblk = math.ceil(CTX / PAGE) + 2
kc = torch.randn(nblk * 4, PAGE, Hkv, D, device=dev).to(torch.bfloat16); vc = torch.randn_like(kc)
bt = torch.arange(nblk * 4, device=dev, dtype=torch.int32).view(4, nblk)
def qsa(fn, T, req_of):
    torch.manual_seed(1)
    q = torch.randn(64, Hq, D, device=dev).to(torch.bfloat16)[:T]
    li = torch.arange(CTX, device=dev, dtype=torch.int32).repeat(T, 1)
    return fn(q.contiguous(), kc, vc, li.contiguous(), bt, torch.tensor(req_of[:T], device=dev, dtype=torch.int32))
req = [0, 1, 2, 3] * 16
pinned, stock = qm.qsa_sparse_paged_attention, qm.qsa_sparse_paged_attention._plefp8_orig
p1, p32 = qsa(pinned, 1, req), qsa(pinned, 32, req)
s1, s32 = qsa(stock, 1, req), qsa(stock, 32, req)
print("QSA pinned: token 0 identical in batch of 1 vs 32:", torch.equal(p1[0], p32[0]), "| stock:", torch.equal(s1[0], s32[0]),
      "| pinned vs stock rel diff:", float((p32.float() - s32.float()).norm() / s32.float().norm()))
