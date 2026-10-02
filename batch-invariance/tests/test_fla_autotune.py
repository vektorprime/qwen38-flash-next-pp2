"""Is vLLM's GDN prefill (FLA chunk_gated_delta_rule) bit-identical across processes? Each process autotunes its kernels
(@triton.autotune benchmarks configs at first use). usage: python3 test_fla_autotune.py <out.pt>  (run several times;
with VLLM_TRITON_FORCE_FIRST_CONFIG=1 the first valid config is used instead)."""
import sys, hashlib, torch
import vllm  # noqa  (env_override installs the force-first-config patch when requested)
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
torch.manual_seed(0); dev = "cuda"
T, H, K, V = 2048, 16, 128, 128
q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16); k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, T, H, V, device=dev, dtype=torch.bfloat16)
g = -torch.rand(1, T, H, device=dev, dtype=torch.float32) * 0.1; beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
h0 = torch.zeros(1, H, K, V, device=dev, dtype=torch.float32)
cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
o, ht = chunk_gated_delta_rule(q, k, v, g, beta, scale=K ** -0.5, initial_state=h0, output_final_state=True, cu_seqlens=cu,
                               use_qk_l2norm_in_kernel=True)
torch.cuda.synchronize()
print("hash o:", hashlib.md5(o.float().cpu().numpy().tobytes()).hexdigest()[:12], "ht:", hashlib.md5(ht.float().cpu().numpy().tobytes()).hexdigest()[:12], flush=True)
