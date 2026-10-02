"""Does a plain Triton GEMM give per-row results independent of its tile config (BLOCK_M/N/K, warps, stages) and of M?
If so, an M-adaptive config (small BLOCK_M for decode, large for prefill) is batch-invariant AND fast. Compares to cuBLAS."""
import json, sys, time, torch, triton, triton.language as tl
import torch.nn.functional as F
torch.manual_seed(0)
dev = "cuda"

@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbn, sbk, scm, scn,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM); num_pid_n = tl.cdiv(N, BN)
    group = GROUP_M * num_pid_n
    gid = pid // group; first_m = gid * GROUP_M; gsz = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsz; pid_n = (pid % group) // gsz
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kk = k * BK + rk
        a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak, mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn, mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))

def tmm(x, w, cfg):  # y = x @ w.T  (w: [N, K] like nn.Linear)
    M, K = x.shape; N = w.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    BM, BN, BK, nw, ns = cfg
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    mm_kernel[grid](x, w, y, M, N, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1), BM=BM, BN=BN, BK=BK, GROUP_M=8, num_warps=nw, num_stages=ns)
    return y

CFGS = [(16, 64, 64, 4, 4), (16, 128, 64, 4, 3), (32, 128, 64, 4, 3), (64, 128, 64, 4, 3), (128, 128, 64, 8, 3), (128, 256, 64, 8, 3),
        (16, 64, 128, 4, 3), (128, 128, 32, 4, 4), (64, 64, 32, 4, 4)]
SHAPES = {"gdn in_proj_qkvz": (2560, 16384), "gdn in_proj_ba": (2560, 96), "gdn out_proj": (6144, 2560), "moe router gate": (2560, 512),
          "shared_expert_gate": (2560, 1), "hc down+inject": (10240, 336), "hc up": (320, 10240), "attn q_proj": (2560, 12288), "lm_head": (2560, 248320)}
def bench(f, n=50):
    for _ in range(3): f()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return round((time.perf_counter() - t0) / n * 1e6, 1)
res = {"device": torch.cuda.get_device_name()}
for name, (K, N) in SHAPES.items():
    W = (torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16); X = torch.randn(2048, K, device=dev).to(torch.bfloat16)
    ref = tmm(X[:4], W, CFGS[0])
    variants = 0; checked = 0
    for cfg in CFGS:
        for M in (1, 4, 37, 256, 2048):
            try:
                y = tmm(X[:M], W, cfg)[:min(M, 4)]
            except Exception as e:  # noqa
                continue
            checked += 1; variants += int(not torch.equal(y, ref[:min(M, 4)]))
    small = min(CFGS[:3], key=lambda c: bench(lambda: tmm(X[:4], W, c)))
    large = min(CFGS[3:6], key=lambda c: bench(lambda: tmm(X[:2048], W, c), 10))
    t = {"triton_us_M4": bench(lambda: tmm(X[:4], W, small)), "cublas_us_M4": bench(lambda: F.linear(X[:4], W)),
         "triton_us_M32": bench(lambda: tmm(X[:32], W, small)), "cublas_us_M32": bench(lambda: F.linear(X[:32], W)),
         "triton_us_M2048": bench(lambda: tmm(X[:2048], W, large), 10), "cublas_us_M2048": bench(lambda: F.linear(X[:2048], W), 10)}
    res[name] = {"configs_x_M_checked": checked, "differing_from_reference": variants, "small_cfg": small, "large_cfg": large, **t}
    print(f"{name:20s} checked {checked:2d} differ {variants}  M4 tri {t['triton_us_M4']:7.1f} cub {t['cublas_us_M4']:7.1f} | M32 tri {t['triton_us_M32']:7.1f} cub {t['cublas_us_M32']:7.1f} | M2048 tri {t['triton_us_M2048']:8.1f} cub {t['cublas_us_M2048']:8.1f}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
