"""INT4 (AWQ-style uint4 + zero points, group 32) Marlin MoE: is hook3's invariant Marlin (inv_marlin.make with the CMP config)
valid and batch-invariant for the INT4 checkpoint too? Stock vs invariant, rows of a fixed request vs batch composition."""
import sys, os, torch
sys.path.insert(0, "/w/hook3")
import inv_marlin
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
from vllm import _custom_ops as ops
torch.manual_seed(0); dev = "cuda"
E, H, I, qt, G = int(os.environ.get("E", "128")), 2560, 640, scalar_types.uint4, 32
def pack(K, N):
    a, s, z = [], [], []
    for e in range(E):
        _, qw, sc, zp = awq_marlin_quantize((torch.randn(K, N, device=dev) / K ** 0.5).to(torch.bfloat16), qt, G); a.append(qw); s.append(sc); z.append(zp)
    return torch.stack(a), torch.stack(s), torch.stack(z)
w1, s1, z1 = pack(H, 2 * I); w2, s2, z2 = pack(I, H)
X = torch.randn(4200, H, device=dev).to(torch.bfloat16)
tw, ti = torch.topk(torch.softmax(torch.randn(4200, E, device=dev), -1), 10, -1); tw = (tw / tw.sum(-1, keepdim=True)).float(); ti = ti.to(torch.int32)
orig_align, orig_gemm = mm.moe_align_block_size, ops.moe_wna16_marlin_gemm
cfg = [int(v) for v in os.environ.get("CFG", "64,128,2,128,2").split(",")]
impl = inv_marlin.make(orig_align, orig_gemm, {"thread_k": cfg[0], "thread_n": cfg[1], "blocks_per_sm": cfg[2]},
                       {"thread_k": cfg[0], "thread_n": cfg[3], "blocks_per_sm": cfg[4]})
def moe(idx, impl):
    if impl: mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = impl
    try:
        return mm.fused_marlin_moe(X[idx], w1, w2, None, None, s1, s2, tw[idx], ti[idx], quant_type_id=qt.id, global_num_experts=E, w1_zeros=z1, w2_zeros=z2)
    finally:
        mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = orig_align, orig_gemm
ar = lambda a, b: torch.arange(a, b, device=dev)
def invariance(impl):
    ref4, ref1k = moe(ar(0, 4), impl), moe(ar(0, 1000), impl)
    bad = sum(int((moe(torch.cat([ar(0, 4), ar(2048, 2048 + x)]), impl)[:4] != ref4).any(1).sum()) for x in (0, 4, 12, 28, 60, 124, 252, 1020, 2040))
    bad += sum(int((moe(torch.cat([ar(0, 1000), ar(1100, 1100 + x)]), impl)[:1000] != ref1k).any(1).sum()) for x in (1, 8, 40, 1000))
    return bad
a, b = moe(ar(0, 300), None), moe(ar(0, 300), impl)
print("INT4 g32 zp: stock rows differing:", invariance(None), "| invariant rows differing:", invariance(impl),
      "| inv vs stock rel diff:", float((a.float() - b.float()).norm() / a.float().norm()), flush=True)
