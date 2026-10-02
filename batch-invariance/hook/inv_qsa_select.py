"""Deterministic, batch-invariant QSA token selection.

qsa_select_paged_tokens scores the compressed key blocks (batch-invariant) and picks the top block_topk (512) blocks per query
row with torch.ops._C.persistent_topk. When a row sees more than 512 blocks (context > 2048 tokens) that op returns the selected
blocks in a run-dependent order, and with exact score ties at the cut-off even a run-dependent set; the sparse attention then
sums the selected tokens in that order -> results differ between identical requests. Fix (after persistent_topk, graph-safe):
the k-th largest score T (unique, whatever the tie-breaking) is read back from the op's own selection; one Triton program per
row then writes, in ascending block order, every block with score > T plus the lowest-index blocks with score == T until k are
chosen. Rows with <= 512 visible blocks keep 0..visible-1 ascending, exactly what persistent_topk returns for them."""
import torch
import triton
import triton.language as tl


@triton.jit
def _det_select_kernel(logits_ptr, vis_ptr, thr_ptr, blocks_ptr, stride_l, stride_b, C, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    vis = tl.minimum(tl.load(vis_ptr + row), C)
    T = tl.load(thr_ptr + row)
    over = vis > K
    n_gt = 0
    for c0 in range(0, vis, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        m = cols < vis
        x = tl.load(logits_ptr + row * stride_l + cols, mask=m, other=float("-inf"))
        n_gt += tl.sum((m & (x > T)).to(tl.int32), 0)
    need_eq = K - n_gt
    base = 0
    eq_seen = 0
    for c0 in range(0, vis, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        m = cols < vis
        x = tl.load(logits_ptr + row * stride_l + cols, mask=m, other=float("-inf"))
        gt = m & (x > T)
        eq = m & (x == T)
        eq_rank = eq_seen + tl.cumsum(eq.to(tl.int32), 0)
        sel = tl.where(over, gt | (eq & (eq_rank <= need_eq)), m)
        pos = base + tl.cumsum(sel.to(tl.int32), 0) - 1
        tl.store(blocks_ptr + row * stride_b + pos, cols, mask=sel & (pos < K))
        base += tl.sum(sel.to(tl.int32), 0)
        eq_seen += tl.sum(eq.to(tl.int32), 0)


def fix_selection(logits, visible_blocks, blocks, k, columns):
    """In place on blocks [rows, k] (int32), after persistent_topk(logits, visible_blocks, blocks, ws, k, columns)."""
    rows = blocks.shape[0]
    if rows == 0:
        return
    C = min(columns, logits.shape[1])
    sel = logits.gather(1, blocks.long().clamp_(0, max(C - 1, 0)))
    thr = torch.where(visible_blocks > k, sel.min(1).values, torch.full((rows,), float("-inf"), device=logits.device))
    _det_select_kernel[(rows,)](logits, visible_blocks, thr.contiguous(), blocks, logits.stride(0), blocks.stride(0), C,
                                K=k, BLOCK=1024, num_warps=4)
