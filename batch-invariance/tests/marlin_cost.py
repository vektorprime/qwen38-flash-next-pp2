"""Cost breakdown + invariance of Marlin MoE variants: stock | pad-only (auto exec config) | pin-only | pin+pad.
Invariance check: the same tokens computed inside batches of different size/composition (rows compared bitwise)."""
import json, sys, time, os, torch
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
from vllm import _custom_ops as ops
torch.manual_seed(0)
dev = "cuda"; E, H, I, TOPK = 512, 2560, 640, 10; qt = scalar_types.uint8b128
SMS = torch.cuda.get_device_properties(0).multi_processor_count
def pack(K, N):
    qws, ss = [], []
    for e in range(E):
        w = (torch.randn(K, N, device=dev) / K ** 0.5).to(torch.bfloat16)
        _, qw, s, _, _, _ = marlin_quantize(w, qt, -1, False); qws.append(qw); ss.append(s)
    return torch.stack(qws), torch.stack(ss)
w1, s1 = pack(H, 2 * I); w2, s2 = pack(I, H)
NMAX = 4096
X = torch.randn(NMAX, H, device=dev).to(torch.bfloat16)
tw, ti = torch.topk(torch.softmax(torch.randn(NMAX, E, device=dev), -1), TOPK, -1)
tw = (tw / tw.sum(-1, keepdim=True)).float(); ti = ti.to(torch.int32)
orig_gemm, orig_align = ops.moe_wna16_marlin_gemm, mm.moe_align_block_size
PIN = {"thread_k": 64, "thread_n": 128, "blocks_per_sm": 1}
def make_align(P):
    def padded_align(topk_ids, block_size, num_experts, expert_map=None, *a, **k):
        sid, eid, ntpp = orig_align(topk_ids, block_size, num_experts, expert_map, *a, **k)
        sentinel = topk_ids.numel()
        sid = torch.cat([sid, sid.new_full((P * block_size,), sentinel)]); eid = torch.cat([eid, eid.new_zeros(P)])
        pos = ntpp.to(torch.int64) + torch.arange(P * block_size, device=dev); sid.scatter_(0, pos, sid.new_full((P * block_size,), sentinel))
        bpos = ntpp.to(torch.int64) // block_size + torch.arange(P, device=dev); eid.scatter_(0, bpos, eid.new_zeros(P))
        return sid, eid, ntpp + P * block_size
    return padded_align
def P_for(bps, tn_min):  # padding blocks so that P * n_tiles(w13) >= 4/3 * grid
    grid = SMS * bps; return -(-(4 * grid) // (3 * (1280 // tn_min))) + 1
VARIANTS = {"stock": (None, None), "pad-only(auto cfg)": (P_for(4, 64), None), "pin-only": (None, PIN), "pin+pad": (P_for(1, 128), PIN)}
def run(idx, var):
    P, pin = VARIANTS[var]
    if P: mm.moe_align_block_size = make_align(P)
    if pin: mm.ops.moe_wna16_marlin_gemm = lambda *a, **k: orig_gemm(*a, **{**k, **pin})
    try:
        return mm.fused_marlin_moe(X[idx].contiguous(), w1, w2, None, None, s1, s2, tw[idx].contiguous(), ti[idx].contiguous(), quant_type_id=qt.id, global_num_experts=E)
    finally:
        mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = orig_align, orig_gemm
ar = lambda a, b: torch.arange(a, b, device=dev)
res = {"device": torch.cuda.get_device_name(), "sms": SMS, "P": {k: v[0] for k, v in VARIANTS.items()}}
for var in VARIANTS:
    r = {"invariance": {}, "time_ms": {}}
    # the same 4 "decode" tokens and the same 1000 "prefill" tokens inside batches of many sizes
    probe4 = ar(0, 4); ref4 = run(probe4, var)
    bad = 0
    for extra in (0, 4, 12, 28, 60, 124, 252, 508, 1020, 2040):
        o = run(torch.cat([probe4, ar(2048, 2048 + extra)]), var)[:4]
        bad += int((o != ref4).any(1).sum())
    r["invariance"]["4 tokens in 10 batch sizes (rows differing, of 40)"] = bad
    probe = ar(0, 1000); ref = run(probe, var); bad = 0
    for extra in (1, 8, 40, 500, 1048):
        bad += int((run(torch.cat([probe, ar(2048, 2048 + extra)]), var)[:1000] != ref).any(1).sum())
    r["invariance"]["1000 tokens in 5 batch sizes (rows differing, of 5000)"] = bad
    for M in (1, 2, 4, 8, 16, 32, 64, 256, 2040):
        idx = ar(0, M)
        for _ in range(3): run(idx, var)
        n = 20 if M > 256 else 100
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): run(idx, var)
        torch.cuda.synchronize(); r["time_ms"][M] = round((time.perf_counter() - t0) / n * 1e3, 3)
    res[var] = r
    print(var, json.dumps(r), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
