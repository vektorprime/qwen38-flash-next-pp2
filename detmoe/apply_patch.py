# Produces the patched moe_align_block_size.py from the image's original (run on host).
# usage: python3 apply_patch.py <original.py> <patched.py>
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
assert "VLLM_MOE_DETERMINISTIC_ALIGN" not in s, "already patched"

imp = "import torch\n\nfrom vllm import _custom_ops as ops\nfrom vllm.triton_utils import triton\n"
assert s.count(imp) == 1
s = s.replace(
    imp,
    "import os\n\nimport torch\n\nfrom vllm import _custom_ops as ops\n"
    "from vllm.triton_utils import tl, triton\n",
    1,
)

HELPER = '''

# PATCH (detmoe, 2026-09-29): ops.moe_align_block_size places token slots into
# their expert's segment with GPU atomics, so the order inside a segment (and
# hence which block_size-row block a token lands in) changes from run to run
# once a batch has more than 16 tokens. Marlin MoE results depend on that
# placement at the rounding level, and MoE routing in later layers amplifies it
# (identical requests gave different logprobs). Rewrite each expert segment in
# ascending slot order: same segments, same padding, same expert_ids, same
# num_tokens_post_pad, deterministic order. Device-only (no host sync), so it
# is safe under CUDA-graph capture. Disable with VLLM_MOE_DETERMINISTIC_ALIGN=0.
_DETERMINISTIC_ALIGN = os.environ.get("VLLM_MOE_DETERMINISTIC_ALIGN", "1") != "0"
_CANON_SEARCH_ITERS = 24  # binary-search depth; covers up to 16M blocks
_CANON_BLOCK_N = 1024


@triton.jit
def _canonicalize_sorted_ids_kernel(
    topk_ids_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    numel,
    block_size: tl.constexpr,
    SEARCH_ITERS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per expert. Segments are laid out in ascending expert order,
    # so the segment start is lower_bound(expert) over the valid expert_ids.
    expert = tl.program_id(0)
    n_blocks = tl.load(num_tokens_post_pad_ptr) // block_size
    lo = n_blocks * 0
    hi = n_blocks
    for _ in range(SEARCH_ITERS):
        active = lo < hi
        mid = (lo + hi) // 2
        v = tl.load(expert_ids_ptr + mid, mask=active, other=0)
        lo = tl.where(active & (v < expert), mid + 1, lo)
        hi = tl.where(active & (v >= expert), mid, hi)
    found = lo < n_blocks
    v = tl.load(expert_ids_ptr + lo, mask=found, other=-1)
    if found & (v == expert):
        base = lo * block_size
        count = n_blocks * 0
        # Stream the flattened topk_ids in order; write this expert's slots
        # contiguously from the segment start (padding stays untouched).
        for start in range(0, numel, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            valid = offs < numel
            ids = tl.load(topk_ids_ptr + offs, mask=valid, other=-1)
            hit = (ids == expert) & valid
            rank = tl.cumsum(hit.to(tl.int32), axis=0)
            tl.store(
                sorted_ids_ptr + base + count + rank - 1,
                offs.to(tl.int32),
                mask=hit,
            )
            count += tl.sum(hit.to(tl.int32), axis=0)


def _canonicalize_sorted_ids_torch(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    # General fallback (any contiguous segment layout, e.g. local expert ids).
    num_ids = sorted_ids.numel()
    num_blocks = expert_ids.numel()
    device = sorted_ids.device
    block_idx = torch.arange(num_blocks, device=device, dtype=torch.int64)
    valid_block = block_idx < (num_tokens_post_pad.to(torch.int64) // block_size)
    experts = expert_ids.to(torch.int64)
    seg_start = torch.ones(num_blocks, device=device, dtype=torch.bool)
    seg_start[1:] = experts[1:] != experts[:-1]
    seg = torch.cummax(
        torch.where(seg_start, block_idx, torch.zeros_like(block_idx)), dim=0
    ).values
    seg = torch.where(valid_block, seg, torch.full_like(seg, num_blocks))
    key = seg.repeat_interleave(block_size)[:num_ids] * (1 << 32) + sorted_ids.to(
        torch.int64
    )
    return sorted_ids[torch.argsort(key, stable=True)]


def _canonicalize_sorted_ids(
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size: int,
    num_experts: int,
    global_expert_ids: bool,
) -> torch.Tensor:
    if sorted_ids.numel() == 0 or expert_ids.numel() == 0 or topk_ids.numel() == 0:
        return sorted_ids
    if not global_expert_ids or not sorted_ids.is_cuda:
        return _canonicalize_sorted_ids_torch(
            sorted_ids, expert_ids, num_tokens_post_pad, block_size
        )
    _canonicalize_sorted_ids_kernel[(num_experts,)](
        topk_ids.reshape(-1),
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_ids.numel(),
        block_size=block_size,
        SEARCH_ITERS=_CANON_SEARCH_ITERS,
        BLOCK_N=_CANON_BLOCK_N,
        num_warps=4,
    )
    return sorted_ids

'''

anchor = "\n\ndef moe_align_block_size("
assert s.count(anchor) == 1
s = s.replace(anchor, HELPER + "\ndef moe_align_block_size(", 1)

call_end = "        expert_map if ignore_invalid_experts else None,\n    )\n"
assert s.count(call_end) == 1
s = s.replace(
    call_end,
    call_end
    + "\n    if _DETERMINISTIC_ALIGN:\n"
    + "        # expert_ids are global (ascending) unless the kernel received\n"
    + "        # expert_map, in which case use the layout-agnostic fallback.\n"
    + "        sorted_ids = _canonicalize_sorted_ids(\n"
    + "            topk_ids,\n"
    + "            sorted_ids,\n"
    + "            expert_ids,\n"
    + "            num_tokens_post_pad,\n"
    + "            block_size,\n"
    + "            num_experts,\n"
    + "            global_expert_ids=not (ignore_invalid_experts and expert_map is not None),\n"
    + "        )\n",
    1,
)
open(dst, "w").write(s)
print("patched ->", dst)
