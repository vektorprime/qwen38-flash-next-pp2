"""Batch-invariant Marlin MoE (no kernel rebuild).

Marlin's scheduler (csrc/.../marlin_moe_wna16/marlin_template.h, "DP + two-tile stream-K"): with global = blocks * n_tiles
output tiles and grid = SMs * blocks_per_sm threadblocks, if global > grid the last part2 tiles (part2 = global % grid, plus
grid if that is <= grid/3) are stream-K split along K across threadblocks; all other tiles are computed whole. Which tiles
land in the split region, and where they are split, depends on the batch -> per-row results depend on the batch.

Fix: after moe_align_block_size, append P empty expert blocks (expert 0, sentinel token ids -> nothing is written), with P
the smallest value for which, for both MoE GEMMs, global > grid and P * n_tiles >= part2: the split region then lies
entirely in padding and every real tile is computed whole, in a fixed K order. thread_k must be the same for every block
size (it changes per-row results); thread_n and blocks_per_sm only change speed. P is computed on the device by one Triton
kernel that also writes the padded copies (graph-safe, no host sync).
"""
import torch
import triton
import triton.language as tl

N13, N2 = 1280, 2560   # Qwen3.8-Flash-Next: w13 = 2 x 640, w2 = hidden 2560


@triton.jit
def _inv_pad_kernel(sid_in, eid_in, ntpp_ptr, sid_out, eid_out, ntpp_out, n_sid_in, n_eid_in, n_sid_out, n_eid_out,
                    sentinel, bs, g, n1, n2, PMAX: tl.constexpr, PP2: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    ntpp = tl.load(ntpp_ptr)
    B = ntpp // bs
    ps = tl.arange(0, PP2)
    g1 = (B + ps) * n1
    r1 = g1 % g
    q1 = tl.where(3 * r1 > g, r1, r1 + g)
    g2 = (B + ps) * n2
    r2 = g2 % g
    q2 = tl.where(3 * r2 > g, r2, r2 + g)
    ok = (ps <= PMAX) & (g1 > g) & (ps * n1 >= q1) & (g2 > g) & (ps * n2 >= q2)
    P = tl.min(tl.where(ok, ps, PMAX))
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(sid_in + offs, mask=(offs < ntpp) & (offs < n_sid_in), other=0)
    tl.store(sid_out + offs, tl.where(offs < ntpp, v, sentinel), mask=offs < n_sid_out)
    e = tl.load(eid_in + offs, mask=(offs < B) & (offs < n_eid_in), other=0)
    tl.store(eid_out + offs, tl.where(offs < B, e, 0), mask=offs < n_eid_out)
    if pid == 0:
        tl.store(ntpp_out, ntpp + P * bs)


def make(orig_align, orig_gemm, cfg_small, cfg_large, on_first=None):
    """cfg_*: dict(thread_k, thread_n, blocks_per_sm) for block size <= 16 / > 16. Returns (align, gemm)."""
    assert cfg_small["thread_k"] == cfg_large["thread_k"], "thread_k must be identical for all block sizes"
    st = {"sms": None}

    def pick(bs):
        return cfg_small if bs <= 16 else cfg_large

    def align(topk_ids, block_size, num_experts, expert_map=None, *a, **k):
        sid, eid, ntpp = orig_align(topk_ids, block_size, num_experts, expert_map, *a, **k)
        if st["sms"] is None:
            st["sms"] = torch.cuda.get_device_properties(topk_ids.device).multi_processor_count
            if on_first:
                on_first(st["sms"])
        c = pick(block_size)
        g = st["sms"] * c["blocks_per_sm"]
        n1, n2 = N13 // c["thread_n"], N2 // c["thread_n"]
        pmax = -(-(4 * g) // (3 * min(n1, n2))) + 2
        sid2 = torch.empty(sid.numel() + pmax * block_size, dtype=sid.dtype, device=sid.device)
        eid2 = torch.empty(eid.numel() + pmax, dtype=eid.dtype, device=eid.device)
        nt2 = torch.empty_like(ntpp)
        BLOCK = 1024
        grid = (triton.cdiv(max(sid2.numel(), eid2.numel()), BLOCK),)
        _inv_pad_kernel[grid](sid, eid, ntpp, sid2, eid2, nt2, sid.numel(), eid.numel(), sid2.numel(), eid2.numel(),
                              topk_ids.numel(), block_size, g, n1, n2, PMAX=pmax, PP2=triton.next_power_of_2(pmax + 1),
                              BLOCK=BLOCK, num_warps=4)
        return sid2, eid2, nt2

    def gemm(*a, **k):
        return orig_gemm(*a, **{**k, **pick(k.get("moe_block_size", 64))})

    return align, gemm
