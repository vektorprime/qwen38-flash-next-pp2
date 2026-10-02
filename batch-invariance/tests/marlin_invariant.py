"""Batch-invariant Marlin MoE without recompiling: pin the thread config and append P empty expert blocks after the real
ones, so Marlin's stream-K region (the last part2 <= 4/3 * grid output tiles) always falls on padding and every real
output tile is computed whole (data-parallel part, fixed K order). Padding blocks point at expert 0 and contain only the
sentinel token id, so they compute but write nothing.
Tests: (1) prefill M in {2032, 2040, 2048}, rows < 2032;  (2) decode-style: the same 4 tokens alone vs inside batches of
8 / 32 / 64 / 256 other tokens (different block_size_m, different expert mixes);  (3) timing vs stock Marlin.
"""
import json, sys, time, torch
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
from vllm import _custom_ops as ops
torch.manual_seed(0)
dev = "cuda"
E, H, I, TOPK = 512, 2560, 640, 10
qt = scalar_types.uint8b128
SMS = torch.cuda.get_device_properties(0).multi_processor_count


def pack(K, N):
    qws, ss = [], []
    for e in range(E):
        w = (torch.randn(K, N, device=dev) / K ** 0.5).to(torch.bfloat16)
        _, qw, s, _, _, _ = marlin_quantize(w, qt, -1, False)
        qws.append(qw); ss.append(s)
    return torch.stack(qws), torch.stack(ss)


w1, s1 = pack(H, 2 * I)
w2, s2 = pack(I, H)
NMAX = 2048 + 300
X = torch.randn(NMAX, H, device=dev).to(torch.bfloat16)
tw, ti = torch.topk(torch.softmax(torch.randn(NMAX, E, device=dev), -1), TOPK, -1)
tw = (tw / tw.sum(-1, keepdim=True)).float(); ti = ti.to(torch.int32)

orig_gemm, orig_align = ops.moe_wna16_marlin_gemm, mm.moe_align_block_size
CFG = {"thread_k": int(__import__("os").environ.get("TK", "64")), "thread_n": int(__import__("os").environ.get("TN", "128")), "blocks_per_sm": 1}   # pinned exec config
MIN_TN = 64                                                    # smallest thread_n -> most n-tiles; conservative P


def padded_align(topk_ids, block_size, num_experts, expert_map=None, *a, **k):
    sorted_ids, expert_ids, ntpp = orig_align(topk_ids, block_size, num_experts, expert_map, *a, **k)
    grid = SMS * CFG["blocks_per_sm"]
    n_tiles_min = 1280 // CFG["thread_n"] if CFG["thread_n"] else 1280 // MIN_TN   # w13 has the fewest n-tiles (N=1280)
    P = -(-(4 * grid) // (3 * max(n_tiles_min, 1))) + 2         # padding blocks: P * n_tiles >= 4/3 grid (+2 safety)
    sentinel = topk_ids.numel()
    nb = sorted_ids.numel() // block_size
    sid = torch.cat([sorted_ids, torch.full((P * block_size,), sentinel, dtype=sorted_ids.dtype, device=sorted_ids.device)])
    eid = torch.cat([expert_ids, torch.zeros(P, dtype=expert_ids.dtype, device=expert_ids.device)])
    # move the padding right behind the last real block (device-side, no host sync)
    pos = ntpp.to(torch.int64) + torch.arange(P * block_size, device=dev)
    sid.scatter_(0, pos, torch.full_like(pos, sentinel, dtype=sid.dtype))
    bpos = ntpp.to(torch.int64) // block_size + torch.arange(P, device=dev)
    eid.scatter_(0, bpos, torch.zeros(P, dtype=eid.dtype, device=dev))
    return sid, eid, ntpp + P * block_size


def pinned_gemm(*a, **k):
    k.update(CFG)
    return orig_gemm(*a, **k)


def run(idx, invariant):
    if invariant:
        mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = padded_align, pinned_gemm
    try:
        return mm.fused_marlin_moe(X[idx].contiguous(), w1, w2, None, None, s1, s2, tw[idx].contiguous(), ti[idx].contiguous(),
                                   quant_type_id=qt.id, global_num_experts=E)
    finally:
        mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = orig_align, orig_gemm


def same_rows(a, b, n):
    d = (a[:n] != b[:n])
    return {"rows_differ": int(d.any(1).sum()), "of": n, "elems_differ": int(d.sum())}


res = {"device": torch.cuda.get_device_name(), "sms": SMS, "cfg": CFG}
ar = torch.arange
for inv in (False, True):
    tag = "invariant" if inv else "stock"
    r = {}
    base = run(ar(2032, device=dev), inv)
    for M in (2040, 2048):
        r[f"prefill M2032 vs M{M}"] = same_rows(base, run(ar(M, device=dev), inv), 2032)
    four = ar(4, device=dev)
    alone = run(four, inv)
    for extra in (4, 28, 60, 252, 1020):
        r[f"decode: 4 tokens alone vs with {extra} others"] = same_rows(alone, run(torch.cat([four, ar(2048, 2048 + extra, device=dev) % NMAX]), inv), 4)
    one = run(ar(1, device=dev), inv)
    r["decode: 1 token alone vs with 7 others"] = same_rows(one, run(ar(8, device=dev), inv), 1)
    # timing
    for M in (1, 4, 32, 2040):
        idx = ar(M, device=dev)
        for _ in range(3): run(idx, inv)
        torch.cuda.synchronize(); t0 = time.perf_counter(); n = 20 if M > 64 else 100
        for _ in range(n): run(idx, inv)
        torch.cuda.synchronize(); r[f"time_ms M={M}"] = round((time.perf_counter() - t0) / n * 1e3, 3)
    # correctness vs stock (same math, different summation order): relative difference
    if inv:
        a, b = run(ar(2040, device=dev), False), run(ar(2040, device=dev), True)
        r["max_rel_diff_vs_stock"] = float(((a.float() - b.float()).norm(dim=1) / a.float().norm(dim=1)).max())
    res[tag] = r
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
