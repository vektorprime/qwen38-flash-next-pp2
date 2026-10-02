"""Check one stage against the prod baseline. usage: stage_check.py <TAG> [--base PROD_base] -> exit 0 = pass
Reads results/<TAG>_k20.npz + results/gen_<TAG>.json and the baseline's; appends to results/stages.json.
Gates
  kld_bit_identical      corpus prompt logprobs (77 x 2048, top-20) bitwise equal to the baseline (target untouched)
  greedy_identical       16 greedy outputs byte-identical to the baseline
  calibration_t1         prod sampling (top-k 20 / top-p 0.95): |z_nll|, |z_argmax| <= 4 and PIT KS p >= 1e-3 vs the
                         target's p' (top-k membership is not exactly reconstructible from top-20 logprobs, so outside
                         counts are reported only)
  calibration_exact      top_k off / top_p 0.95 (exactly reconstructible): 0 sampled tokens outside the top-p set, same z/PIT
  pad                    first-token '!' count not above max(2, 3 * expected)
KLD / PPL / top-1 / same-top-p-set metrics are computed with plefp8/e2e/compare_ab.pair (same definitions as before).
"""
import importlib.util, json, os, sys
import numpy as np

R = "/home/user/qwen3nextflash/mtp/results/"
tag = sys.argv[1]
base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "PROD_base"
spec = importlib.util.spec_from_file_location("cab", "/home/user/qwen3nextflash/plefp8/e2e/compare_ab.py")
cab = importlib.util.module_from_spec(spec); spec.loader.exec_module(cab)
out = {"tag": tag, "base": base, "gates": {}}


def load(n):
    return {k: v for k, v in np.load(R + n + "_k20.npz").items()}


if os.path.exists(R + tag + "_k20.npz"):
    a, b = load(base), load(tag)
    ident = bool(np.array_equal(a["ids"], b["ids"]) and np.array_equal(a["logprobs"], b["logprobs"]))
    p = cab.pair(a, b)[0]
    out["kld"] = {k: p[k] for k in ("kld_mean", "kld_p99", "kld_max", "top1_same", "sampling_set_identical",
                                    "sampling_same_token_prob", "ppl_ref", "ppl_test", "dnll_mean") if k in p}
    out["gates"]["kld_bit_identical"] = ident
g = json.load(open(R + f"gen_{tag}.json"))
gb = json.load(open(R + f"gen_{base}.json"))
if "greedy" in g:
    same = [x == y for x, y in zip(g["greedy"]["sha"], gb["greedy"]["sha"])]
    out["greedy_same"] = f"{sum(same)}/{len(same)}"
    out["gates"]["greedy_identical"] = all(same)
if "t1" in g:
    c = g["t1"]["calibration"]
    out["calibration"] = c
    out["gates"]["calibration_t1"] = bool(abs(c["z_nll"]) <= 4 and abs(c["z_argmax"]) <= 4 and c["pit_ks_p"] >= 1e-3)
if "cal" in g:
    cc = g["cal"]["calibration"]
    out["calibration_cal"] = {k: cc[k] for k in ("n_tokens", "n_evaluated", "outside_nucleus", "z_nll", "z_argmax", "pit_ks_p")}
    out["gates"]["calibration_exact"] = bool(cc["outside_nucleus"] == 0 and abs(cc["z_nll"]) <= 4 and abs(cc["z_argmax"]) <= 4
                                             and cc["pit_ks_p"] >= 1e-3)
    out["accept"] = {k: g["t1"].get(k) for k in ("all", "clean")}
if "pad" in g:
    pd = g["pad"]
    out["pad"] = {k: pd[k] for k in ("n", "first_is_bang", "expected_bang", "max_bang_run")}
    out["gates"]["pad"] = pd["first_is_bang"] <= max(2, 3 * pd["expected_bang"])
for part in ("t1", "cal", "pad", "greedy"):          # a missing part (crash) is a failure, not a skip
    out["gates"].setdefault("present_" + part, part in g)
if "pad" in g:
    out["gates"]["pad_no_errors"] = not g["pad"].get("errors") and g["pad"]["n"] > 0
out["pass"] = all(out["gates"].values())
allst = json.load(open(R + "stages.json")) if os.path.exists(R + "stages.json") else {}
allst[tag] = out
json.dump(allst, open(R + "stages.json", "w"), indent=1)
short = {k: v for k, v in out.items() if k not in ("calibration",)}
short["calib"] = {k: out.get("calibration", {}).get(k) for k in ("n_tokens", "n_evaluated", "outside_nucleus", "outside_strict", "z_nll", "z_argmax", "pit_ks_p")}
print(json.dumps(short, indent=1, default=str))
sys.exit(0 if out["pass"] else 1)
