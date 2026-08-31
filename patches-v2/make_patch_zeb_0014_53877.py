import pathlib

# --- 0014: per-rank KV budget override in gpu_worker.py ---
p = pathlib.Path("/opt/vllm/vllm/v1/worker/gpu_worker.py")
src = p.read_text()
assert "VLLM_KV_CACHE_MEMORY_RANK" not in src, "0014 already applied"
anchor = "        maybe_apply_startup_plan(self)\n"
assert src.count(anchor) == 1, "0014 anchor not found"
ins = anchor + """
        # PATCH (zebgop 0014): per-rank KV budget override. --kv-cache-memory is a
        # single value for every worker, but under PP the ranks are
        # heterogeneous (PLE machinery on rank 0, the MTP drafter on the last
        # rank, different per-token KV costs), so one global value must fit
        # the tightest rank and strands memory everywhere else. Let each rank
        # take its own absolute budget from VLLM_KV_CACHE_MEMORY_RANK<i>.
        import os as _os
        _pp_rank = get_pp_group().rank_in_group
        _override = _os.getenv(f"VLLM_KV_CACHE_MEMORY_RANK{_pp_rank}")
        if _override and not self.cache_config.kv_cache_memory_bytes:
            self.cache_config.kv_cache_memory_bytes = int(_override)
            logger.info(
                "Per-rank KV budget override: pp_rank=%d takes %s bytes from "
                "VLLM_KV_CACHE_MEMORY_RANK%d",
                _pp_rank, _override, _pp_rank,
            )
"""
src = src.replace(anchor, ins, 1)
p.write_text(src)

import ast
ast.parse(src)
print("0014 applied to gpu_worker.py - syntax OK")

# --- #53877: fp32 GDN decode beta in fused_recurrent.py ---
p2 = pathlib.Path("/opt/vllm/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py")
src2 = p2.read_text()
old = "    beta_val = tl.sigmoid(b_val)\n"
new = "    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)\n"
assert src2.count(old) == 1, "53877 anchor not unique"
src2 = src2.replace(old, new, 1)
p2.write_text(src2)
ast.parse(src2)
print("53877 applied to fused_recurrent.py - syntax OK")
