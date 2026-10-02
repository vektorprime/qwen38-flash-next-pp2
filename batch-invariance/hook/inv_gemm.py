"""Batch-invariant bf16 GEMM for vLLM's unquantized linear layers (y = x @ W^T [+ b]).
A plain Triton tiled GEMM: every output element is accumulated over K in the same order (16-wide MMA steps, K ascending)
whatever the tile shape, so the tile config can follow M (fast decode and fast prefill) without changing any result.
Verified bit-identical across 9 tile configs and M = 1..2048 on sm86 (batchinv/triton_mm_invariance.py).
Narrow GEMMs (N <= 1024, K >= 2048) use split-K with a split count fixed by the weight shape and an ordered fp32
reduction, which keeps them invariant (cuBLAS split-K is not: it picks the split from M).
Exposed as torch.ops.plefp8.inv_linear so torch.compile keeps it opaque (no cuBLAS lowering) and CUDA graphs capture it.
"""
import json
import os

import torch
import triton
import triton.language as tl

# Optional tuned table (written by batchinv/tune_cmp.py on the serving GPU): {"N,K": {"16": [BM,BN,BK,warps,stages], "64": ...,
# "256": ..., "4096": ..., "splits": S}}. Any entry gives identical results; the table only changes speed.
_TUNED = {}
_tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned_gemm.json")
if os.path.exists(_tp):
    _TUNED = json.load(open(_tp))


def _bucket(M):
    return "16" if M <= 16 else "64" if M <= 64 else "256" if M <= 256 else "4096"


@triton.jit
def _inv_mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbn, sbk, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    group = GROUP_M * num_pid_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsz = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsz
    pid_n = (pid % group) // gsz
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kk = k * BK + rk
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _inv_mm_splitk_kernel(a_ptr, b_ptr, p_ptr, M, N, K, KCHUNK, sam, sak, sbn, sbk,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # grid (m tiles, n tiles, splits); split s covers K range [s*KCHUNK, (s+1)*KCHUNK) - fixed by the weight shape only
    pid_m, pid_n, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = s * KCHUNK
    for k in range(0, tl.cdiv(KCHUNK, BK)):
        kk = k0 + k * BK + rk
        km = (kk < K) & (kk < k0 + KCHUNK)
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak, mask=(rm[:, None] < M) & km[None, :], other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn, mask=km[:, None] & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(p_ptr + s * M * N + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _inv_splitk_reduce_kernel(p_ptr, c_ptr, M, N, scm, scn, S: tl.constexpr, BLOCK: tl.constexpr):
    # sums the S fp32 partials in fixed order s = 0..S-1
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    msk = offs < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(S):
        acc += tl.load(p_ptr + s * M * N + offs, mask=msk, other=0.0)
    m, n = offs // N, offs % N
    tl.store(c_ptr + m * scm + n * scn, acc.to(c_ptr.dtype.element_ty), mask=msk)


def _splits(N, K):
    # split count depends on the weight shape only (never on M): narrow GEMMs get K-parallelism, wide ones none
    t = _TUNED.get(f"{N},{K}")
    if t and "splits" in t:
        return int(t["splits"])
    if N > 1024 or K < 2048:
        return 1
    return 16 if K >= 8192 else 4


def _cfg(M, N, K):
    # (BM, BN, BK, num_warps, num_stages); any choice gives identical results, only speed differs
    t = _TUNED.get(f"{N},{K}")
    if t and _bucket(M) in t:
        return tuple(t[_bucket(M)])
    if M <= 16:
        return (16, 64, 128, 4, 3) if N >= 4096 else (16, 64, 64, 4, 4)
    if M <= 64:
        return (32, 128, 64, 4, 3)
    if M <= 256:
        return (64, 128, 64, 4, 3)
    return (128, 128, 64, 8, 3) if N < 8192 else (128, 256, 64, 8, 3)


def _skcfg(M, N, K):
    # split-K tile config; any choice gives identical results (split boundaries and reduce order are fixed by S), only speed differs.
    # BK must divide the 64-aligned K chunk.
    t = _TUNED.get(f"{N},{K}")
    if t and "sk" + _bucket(M) in t:
        return tuple(t["sk" + _bucket(M)])
    return (16 if M <= 16 else (32 if M <= 64 else 64), 32 if N <= 512 else 64, 64, 4, 3)


_BANNER = [False]


@torch.library.custom_op("plefp8::inv_linear", mutates_args=())
def inv_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if not _BANNER[0]:   # runtime body of an opaque custom op: never traced by Dynamo
        _BANNER[0] = True
        print(f"[hook3] invariant Triton GEMM in use (pid {__import__('os').getpid()})", flush=True)
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    M, K = x2.shape
    N = w.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    S = _splits(N, K)
    if M > 0 and S > 1:
        kchunk = -(-K // S)
        kchunk = -(-kchunk // 64) * 64
        BM, BN, BK, nw, ns = _skcfg(M, N, K)
        p = torch.empty((S, M, N), device=x.device, dtype=torch.float32)
        _inv_mm_splitk_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), S)](
            x2, w, p, M, N, K, kchunk, x2.stride(0), x2.stride(1), w.stride(0), w.stride(1), BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
        BLOCK = 1024
        _inv_splitk_reduce_kernel[(triton.cdiv(M * N, BLOCK),)](p, y, M, N, y.stride(0), y.stride(1), S=S, BLOCK=BLOCK, num_warps=4)
    elif M > 0:
        BM, BN, BK, nw, ns = _cfg(M, N, K)
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _inv_mm_kernel[grid](x2, w, y, M, N, K, x2.stride(0), x2.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
                             BM=BM, BN=BN, BK=BK, GROUP_M=8, num_warps=nw, num_stages=ns)
    return y.reshape(*shp[:-1], N)


@inv_linear.register_fake
def _(x, w):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def applicable(x, w):
    return (x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.is_cuda and w.dim() == 2 and x.dim() >= 1
            and x.shape[-1] == w.shape[1])


def linear(x, w, bias=None):
    if not applicable(x, w):
        return torch.nn.functional.linear(x, w, bias)
    y = torch.ops.plefp8.inv_linear(x, w)
    return y if bias is None else y + bias
