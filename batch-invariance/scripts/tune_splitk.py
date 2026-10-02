"""Split-K tile tuning for the narrow dense GEMMs (split count S fixed per shape from hook3/tuned_gemm.json).
Also checks that every tile config gives bit-identical rows to the default config, at M=2048 and for the first 4 rows at M=4.
usage: python3 tune_splitk.py out.json   (one free GPU; mount batchinv at /w)"""
import json, sys, time, torch, triton
sys.path.insert(0, "/w/hook3")
import inv_gemm
torch.manual_seed(0); dev = "cuda"
T = json.load(open("/w/hook3/tuned_gemm.json"))
BUCKETS = {16: 4, 64: 32, 256: 160, 4096: 2048}


def gtime(fn, n=10):
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


def cands(M):
    if M <= 16:
        return [(16, bn, bk, 4, s) for bn in (32, 64, 128) for bk in (64, 128, 256) for s in (2, 3, 4)]
    if M <= 64:
        return [(bm, bn, bk, 4, s) for bm in (32, 64) for bn in (32, 64, 128) for bk in (64, 128) for s in (2, 3, 4)]
    if M <= 256:
        return [(bm, bn, bk, w, s) for bm in (32, 64, 128) for bn in (32, 64, 128) for bk in (32, 64) for w in (4, 8) for s in (3, 4)]
    return [(bm, bn, bk, w, s) for bm in (64, 128, 256) for bn in (32, 64, 128) for bk in (32, 64) for w in (4, 8) for s in (2, 3, 4)]


out = {"device": torch.cuda.get_device_name()}
mism = 0
import os
KEYS = os.environ.get("KEYS")
for key, ent in T.items():
    S = ent.get("splits", 1)
    if S <= 1 or (KEYS and key not in KEYS.split(";")):
        continue
    N, K = map(int, key.split(","))
    W = (torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16); X = torch.randn(2048, K, device=dev).to(torch.bfloat16)
    inv_gemm._TUNED = {key: {"splits": S}}
    ref = inv_gemm.inv_linear(X, W)                     # default split-K tile config
    r = {"S": S}
    for ub, M in BUCKETS.items():
        x = X[:M].contiguous()
        best = None
        for c in cands(M):
            inv_gemm._TUNED = {key: {"splits": S, "sk" + str(ub): list(c)}}
            try:
                y = inv_gemm.inv_linear(x, W); torch.cuda.synchronize()
            except Exception:  # noqa  (out of shared memory etc.)
                continue
            if not torch.equal(y, ref[:M]):
                mism += 1; print("MISMATCH", key, M, c, flush=True); continue
            t = gtime(lambda: inv_gemm.inv_linear(x, W))
            if best is None or t < best[0]:
                best = (t, c)
        inv_gemm._TUNED = {key: {"splits": S}}
        t0 = gtime(lambda: inv_gemm.inv_linear(x, W))
        r[str(ub)] = {"cfg": list(best[1]), "us": round(best[0], 1), "default_us": round(t0, 1)}
    out[key] = r
    print(key, json.dumps(r), flush=True)
    del W, X; torch.cuda.empty_cache()
out["mismatches"] = mism
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("done; mismatches:", mism, flush=True)
