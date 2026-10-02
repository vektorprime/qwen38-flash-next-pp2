"""Analysis of the batch-invariance window (dt_window3.sh). Reads batchinv/results/s_*.npz, ctrl/route/*, decode_*.json,
bench_*.json, cmp_bench_cmp.json and prod references in plefp8/e2e/results. Writes results/dt3_analysis.json and prints a summary."""
import glob, json, os, re
from collections import defaultdict
import numpy as np

B = "/home/user/qwen3nextflash/batchinv"
R = B + "/results/"
PE = "/home/user/qwen3nextflash/plefp8/e2e/results/"
SKIP = 32
fc = lambda L: (L // 8) * 8 - 8   # first prefill step size prod uses for an L-token prompt (measured 2026-10-02)


def load(tag):
    p = f"{R}s_{tag}.npz"
    return dict(np.load(p)) if os.path.exists(p) else None


def cmp(X, Y, lo, hi):
    s = slice(lo, hi)
    K = min(X["ids"].shape[2], Y["ids"].shape[2])
    xi, xl = X["ids"][:, s, :K], X["logprobs"][:, s, :K].astype(np.float64)
    yi, yl = Y["ids"][:, s, :K], Y["logprobs"][:, s, :K].astype(np.float64)
    xa, ya = X["actual"][:, s].astype(np.float64), Y["actual"][:, s].astype(np.float64)
    flip = xi[..., 0] != yi[..., 0]
    out = {"positions": int(flip.size), "bit_identical_actual_logprob": bool(np.array_equal(xa, ya, equal_nan=True)),
           "frac_positions_actual_logprob_differs": round(float((xa != ya).mean()), 5), "top1_flip_rate": round(float(flip.mean()), 5)}
    if K >= 2:
        margin = xl[..., 0] - xl[..., 1]
        eq1 = yi == xi[..., 0][..., None]; eq2 = yi == xi[..., 1][..., None]
        ok = eq1.any(-1) & eq2.any(-1)
        delta = ((yl * eq1).sum(-1) - (yl * eq2).sum(-1) - margin)[ok]
        out.update(flips_margin_gt1=int((flip & (margin > 1)).sum()), flips_margin_gt2=int((flip & (margin > 2)).sum()),
                   pair_logit_noise_rms=round(float(np.sqrt((delta ** 2).mean())), 4) if delta.size else 0.0)
    da = ya - xa
    out.update(dactual_rms=round(float(np.sqrt(np.nanmean(da ** 2))), 4), frac_dactual_gt1=round(float(np.nanmean(np.abs(da) > 1)), 5),
               dnll_mean=round(float(-np.nanmean(da)), 5))
    return out


def routes(tag):
    per = defaultdict(list)
    for f in sorted(glob.glob(f"{B}/ctrl/route/{tag}/*.npy"), key=lambda p: int(re.search(r"_(\d{5})_M", p).group(1))):
        per[int(re.search(r"_L(\d+)_", os.path.basename(f)).group(1))].append(np.load(f))
    return per


def route_diff(a, b, n):
    out = {}
    for l in sorted(set(a) & set(b)):
        d = t = 0
        for x, y in zip(a[l], b[l]):
            m = min(n, len(x), len(y))
            d += int((np.sort(x[:m], 1) != np.sort(y[:m], 1)).any(1).sum()); t += m
        out[l] = round(d / max(t, 1), 4)
    return out


res = {}
LS = [2048, 2047, 2040, 2032, 1984, 1536, 1024]
for E in ("E1", "E2"):
    S = {L: load(f"{E}_L{L}") for L in LS}
    if S[2048] is None:
        continue
    g = {}
    for i, La in enumerate(LS):
        for Lb in LS[i + 1:]:
            if S[La] is None or S[Lb] is None:
                continue
            g[f"{La} vs {Lb}"] = {"first-step positions": cmp(S[La], S[Lb], 1, min(fc(La), fc(Lb))),
                                  "all common positions": cmp(S[La], S[Lb], 1, min(La, Lb))}
    res[f"{E} prompt-length invariance"] = g
    prod = dict(np.load(PE + "probe_rep_k20.npz"))
    res[f"{E} L2048 vs prod (stock kernels) L2048"] = cmp(prod, S[2048], SKIP, 2048)
    stock_2047 = dict(np.load(PE + "probe2_trunc2047.npz"))
    res["prod (stock) 2048 vs 2047, for reference"] = cmp(prod, stock_2047, SKIP, 2047)
# mechanism (E1): injected seeds, with and without routing replay
base = load("E1_L2048")
if base is not None:
    mech = {}
    for t in ("inj_D2_L0_f0.0005", "inj_D1_L0_f0.006", "inj_D4_L47_f0.006", "inj_D3_all_f0.006", "inj_D1_L0_f0.006_replay", "inj_D3_all_f0.006_replay"):
        X = load(f"E1_{t}")
        if X is not None:
            mech[t] = cmp(base, X, SKIP, 2048)
    res["E1 dose-response (vs clean E1 L2048)"] = mech
    rc = routes("E1_c2048")
    if rc:
        rr = {}
        for t in ("E1_c2047", "E1_inj_D2_L0_f0.0005", "E1_inj_D1_L0_f0.006", "E1_inj_D3_all_f0.006", "E1_inj_D4_L47_f0.006"):
            r2 = routes(t)
            if r2:
                rr[t] = route_diff(rc, r2, fc(2047) if t.endswith("2047") else 2040)
        res["E1 routing disagreement per layer vs clean 2048 (frac tokens with different top-10 set)"] = rr
for f in sorted(glob.glob(R + "decode_*.json")):
    res[os.path.basename(f)] = json.load(open(f))["res"]
for f in sorted(glob.glob(R + "bench_*.json")):
    res[os.path.basename(f)] = json.load(open(f))["res"]
if os.path.exists(R + "cmp_bench_cmp.json"):
    cb = json.load(open(R + "cmp_bench_cmp.json"))
    res["cmp bench"] = {"sms": cb["sms"], "marlin": cb["marlin_moe"], "dense gemm totals": cb["dense_gemm"]["per_forward_total_us (x layer counts)"],
                        "dense gemm invariance failures": cb["dense_gemm"]["invariance_failures"]}
json.dump(res, open(R + "dt3_analysis.json", "w"), indent=1)


def short(v):
    if not isinstance(v, dict):
        return v
    keys = ("bit_identical_actual_logprob", "frac_positions_actual_logprob_differs", "top1_flip_rate", "flips_margin_gt1", "dactual_rms", "dnll_mean")
    return {k: v[k] for k in keys if k in v} if "top1_flip_rate" in v else v


for k, v in res.items():
    print(f"\n### {k}")
    if isinstance(v, dict):
        for kk, vv in v.items():
            if isinstance(vv, dict) and "first-step positions" in vv:
                print(f"  {kk:14s} first-step {short(vv['first-step positions'])}\n  {'':14s} all-common {short(vv['all common positions'])}")
            elif isinstance(vv, dict) and all(isinstance(x, (int, float)) for x in vv.values()) and len(vv) > 10:
                print(f"  {kk}: " + " ".join(f"L{l:02d}:{x:.3f}" for l, x in vv.items()))
            else:
                print(f"  {kk}: {json.dumps(short(vv))[:600]}")
    else:
        print(" ", v)
