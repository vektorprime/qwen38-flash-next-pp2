"""Speed tuning of the batch-invariant kernels on the CMP 170HX (any config tried here gives the same results; only speed differs).
Part A  Marlin MoE: thread_k fixed across block sizes (it changes per-row results); per class (block size <= 16 / > 16) thread_n and
        blocks_per_sm free; padding rule 'cover': on the device, the smallest number of empty expert blocks P such that for both
        GEMMs Marlin's stream-K region (last part2 tiles) lies entirely in padding, so every real tile is computed whole.
Part B  dense GEMM (inv_gemm Triton kernel): per shape, per M bucket, fastest tile config; per shape one fixed split count.
usage: python3 tune_cmp.py out.json   (one free GPU)"""
import json, math, sys, time, itertools, torch
import torch.nn.functional as F
sys.path.insert(0, "/w/hook3")
import inv_gemm
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
from vllm import _custom_ops as ops
torch.manual_seed(0); dev = "cuda"
SMS = torch.cuda.get_device_properties(0).multi_processor_count
import os
PART = os.environ.get("PART", "AB")
res = {"device": torch.cuda.get_device_name(), "sms": SMS}


def gtime(fn, n=30):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(4): g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / (4 * n) * 1e6


def etime(fn, n=8):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e6


if 'A' in PART:
    # ============================== Part A: Marlin MoE
    E, H, I, qt = 512, 2560, 640, scalar_types.uint8b128
    def pack(K, N):
        a, b = [], []
        for e in range(E):
            _, qw, s, _, _, _ = marlin_quantize((torch.randn(K, N, device=dev) / K ** 0.5).to(torch.bfloat16), qt, -1, False); a.append(qw); b.append(s)
        return torch.stack(a), torch.stack(b)
    w1, s1 = pack(H, 2 * I); w2, s2 = pack(I, H)
    X = torch.randn(4200, H, device=dev).to(torch.bfloat16)
    tw, ti = torch.topk(torch.softmax(torch.randn(4200, E, device=dev), -1), 10, -1); tw = (tw / tw.sum(-1, keepdim=True)).float(); ti = ti.to(torch.int32)
    orig_align, orig_gemm = mm.moe_align_block_size, ops.moe_wna16_marlin_gemm


    import inv_marlin
    def make(small, large):
        return inv_marlin.make(orig_align, orig_gemm, small, large)


    def moe(idx, impl):
        if impl: mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = impl
        try:
            return mm.fused_marlin_moe(X[idx], w1, w2, None, None, s1, s2, tw[idx], ti[idx], quant_type_id=qt.id, global_num_experts=E)
        finally:
            mm.moe_align_block_size, mm.ops.moe_wna16_marlin_gemm = orig_align, orig_gemm


    ar = lambda a, b: torch.arange(a, b, device=dev)
    def invariance(impl):
        ref4, ref1k = moe(ar(0, 4), impl), moe(ar(0, 1000), impl)
        bad = sum(int((moe(torch.cat([ar(0, 4), ar(2048, 2048 + x)]), impl)[:4] != ref4).any(1).sum()) for x in (0, 4, 12, 28, 60, 124, 252, 1020, 2040))
        bad += sum(int((moe(torch.cat([ar(0, 1000), ar(1100, 1100 + x)]), impl)[:1000] != ref1k).any(1).sum()) for x in (1, 8, 40, 1000))
        return bad


    def timing(impl):
        t = {M: round(gtime(lambda idx=ar(0, M): moe(idx, impl), 15), 1) for M in (1, 4, 8, 16, 32)}
        t[2040] = round(etime(lambda idx=ar(0, 2040): moe(idx, impl)), 1)
        return t


    A = {"stock": {"invariance_rows_differing": invariance(None), "us": timing(None)}}
    print("A stock", A["stock"], flush=True)
    cands = []
    for tk in (64, 128):
        for tn_s in (64, 128):
            for tn_l in (64, 128, 256):
                for bps_s in (1, 2):
                    for bps_l in (1, 2):
                        cands.append(({"thread_k": tk, "thread_n": tn_s, "blocks_per_sm": bps_s}, {"thread_k": tk, "thread_n": tn_l, "blocks_per_sm": bps_l}))
    for small, large in cands:
        name = f"k{small['thread_k']} s(n{small['thread_n']},b{small['blocks_per_sm']}) l(n{large['thread_n']},b{large['blocks_per_sm']})"
        impl = make(small, large)
        try:
            t = timing(impl)
        except Exception as e:  # noqa  (invalid config for some block size)
            A[name] = {"error": repr(e)[:120]}; print("A", name, "invalid", flush=True); continue
        A[name] = {"us": t, "small": small, "large": large}
        print("A", name, t, flush=True)
    valid = {k: v for k, v in A.items() if k != "stock" and "us" in v}
    score = lambda v: sum(v["us"][M] for M in (1, 4, 8, 16, 32)) + 0.05 * v["us"][2040]
    best = sorted(valid, key=lambda k: score(valid[k]))[:4]
    for k in best:
        A[k]["invariance_rows_differing"] = invariance(make(valid[k]["small"], valid[k]["large"]))
    A["best"] = best
    print("A best:", [(k, A[k]["us"], A[k]["invariance_rows_differing"]) for k in best], flush=True)
    res["marlin"] = A
    del w1, w2, X; torch.cuda.empty_cache()


if 'B' in PART:
    # ============================== Part B: dense GEMM tile configs
    SHAPES = {"gdn in_proj_qkvz": (2560, 16384), "gdn in_proj_ba": (2560, 96), "gdn out_proj": (6144, 2560), "moe router gate": (2560, 512),
              "shared expert gate_up": (2560, 1280), "shared expert down": (640, 2560), "shared_expert_gate": (2560, 1), "hc down+inject": (10240, 336),
              "hc up": (320, 10240), "ple key_proj": (2560, 10240), "attn q_proj": (2560, 12288), "attn k/v_proj": (2560, 512),
              "qsa index_qk_proj": (2560, 640), "attn o_proj": (6144, 2560), "lm_head": (2560, 248320), "mtp fc": (2560, 2560)}
    BUCKETS = {16: 4, 64: 32, 256: 160, 4096: 2048}   # bucket upper bound -> M used for timing
    def cands_for(M):
        out = []
        if M <= 16:
            out = [(16, bn, bk, 4, s) for bn in (32, 64, 128) for bk in (64, 128, 256) for s in (3, 4, 5)]
        elif M <= 64:
            out = [(bm, bn, bk, 4, s) for bm in (32, 64) for bn in (64, 128) for bk in (64, 128) for s in (3, 4)]
        elif M <= 256:
            out = [(bm, bn, bk, w, s) for bm in (64, 128) for bn in (64, 128) for bk in (32, 64) for w in (4, 8) for s in (3, 4)]
        else:
            out = [(bm, bn, bk, 8, s) for bm in (64, 128) for bn in (128, 256) for bk in (32, 64) for s in (3, 4, 5)] + [(128, 128, 64, 4, 3), (256, 128, 32, 8, 3)]
        return out
    Bres = {}
    import triton
    for name, (K, N) in SHAPES.items():
        W = (torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16); Xg = torch.randn(2048, K, device=dev).to(torch.bfloat16)
        r = {}
        for ub, M in BUCKETS.items():
            x = Xg[:M].clone(); y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            tf = gtime if M <= 256 else etime
            cand = cands_for(M)
            best = None
            for (bm, bn, bk, w, s) in cand:
                def run(bm=bm, bn=bn, bk=bk, w=w, s=s):
                    inv_gemm._inv_mm_kernel[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](x, W, y, M, N, K, x.stride(0), x.stride(1), W.stride(0), W.stride(1),
                                                                                      y.stride(0), y.stride(1), BM=bm, BN=bn, BK=bk, GROUP_M=8, num_warps=w, num_stages=s)
                try:
                    t = tf(run, 10) if tf is gtime else tf(run, 4)
                except Exception:  # noqa  (out of shared memory etc.)
                    continue
                if best is None or t < best[0]:
                    best = (t, (bm, bn, bk, w, s))
            cub = tf(lambda: F.linear(x, W), 10) if tf is gtime else tf(lambda: F.linear(x, W), 4)
            r[ub] = {"best_cfg": best[1], "best_us": round(best[0], 1), "cublas_us": round(cub, 1)}
        # split-K candidates (narrow shapes): fixed split count per shape, timed at every bucket
        if N <= 1024 and K >= 2048:
            sk = {}
            for S in (2, 4, 8, 16):
                inv_gemm._splits = lambda n, k, S=S: S
                sk[S] = {ub: round((gtime if M <= 256 else etime)(lambda M=M: inv_gemm.linear(Xg[:M], W), 10 if M <= 256 else 4), 1) for ub, M in BUCKETS.items()}
            r["splitk_us_by_S"] = sk
        Bres[name] = r
        print("B", name, json.dumps(r), flush=True)
        del W, Xg; torch.cuda.empty_cache()
    res["gemm"] = Bres

json.dump(res, open(sys.argv[1], "w"), indent=1)
print("done", flush=True)
