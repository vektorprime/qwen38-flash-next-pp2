"""Build hook3/tuned_gemm.json from results/tune_cmp_B.json (CMP 170HX). Split count per narrow shape chosen for decode
(M<=64 dominates serving time); any table entry gives identical results, only speed differs."""
import json, sys
B = json.load(open("results/tune_cmp_B.json"))["gemm"]
SH = {"gdn in_proj_qkvz": (2560, 16384), "gdn in_proj_ba": (2560, 96), "gdn out_proj": (6144, 2560), "moe router gate": (2560, 512),
      "shared expert gate_up": (2560, 1280), "shared expert down": (640, 2560), "shared_expert_gate": (2560, 1), "hc down+inject": (10240, 336),
      "hc up": (320, 10240), "ple key_proj": (2560, 10240), "attn q_proj": (2560, 12288), "attn k/v_proj": (2560, 512),
      "qsa index_qk_proj": (2560, 640), "attn o_proj": (6144, 2560), "lm_head": (2560, 248320), "mtp fc": (2560, 2560)}
COUNT = {"gdn in_proj_qkvz": 36, "gdn in_proj_ba": 36, "gdn out_proj": 36, "moe router gate": 49, "shared expert gate_up": 49, "shared expert down": 49,
         "shared_expert_gate": 49, "hc down+inject": 98, "hc up": 98, "ple key_proj": 1, "attn q_proj": 13, "attn k/v_proj": 26, "qsa index_qk_proj": 13,
         "attn o_proj": 13, "lm_head": 2, "mtp fc": 2}
SK = {}   # optional split-K tile tuning files (tune_splitk.py): {"N,K": {"S": S, "16": {"cfg": [...], "us": t}, ...}}
for f in sys.argv[1:]:
    SK.update({k: v for k, v in json.load(open(f)).items() if "," in k})
table = {}; tot = {f"{w}{b}": 0.0 for w in ("cub", "inv") for b in ("16", "64", "256", "4096")}
for name, r in B.items():
    K, N = SH[name]; key = f"{N},{K}"
    ent = {b: r[b]["best_cfg"] for b in ("16", "64", "256", "4096")}
    t = {b: r[b]["best_us"] for b in ("16", "64", "256", "4096")}
    if "splitk_us_by_S" in r:
        opts = {1: t}
        for S, ts in r["splitk_us_by_S"].items():
            opts[int(S)] = {b: ts[b] for b in ("16", "64", "256", "4096")}
        cost = lambda o: o["16"] + o["64"] + 0.25 * o["256"] + 0.01 * o["4096"]
        S = min(opts, key=lambda s: cost(opts[s])); t = dict(opts[S]); ent["splits"] = S
        if key in SK and SK[key].get("S") == S:
            for b in ("16", "64", "256", "4096"):
                ent["sk" + b] = SK[key][b]["cfg"]; t[b] = SK[key][b]["us"]
    else:
        ent["splits"] = 1
    table[key] = ent
    for b in ("16", "64", "256", "4096"):
        tot["cub" + b] += r[b]["cublas_us"] * COUNT[name]; tot["inv" + b] += t[b] * COUNT[name]
    print(f"  {name:22s} S={ent['splits']:2d}  inv {t['16']:7.1f}/{t['64']:7.1f}/{t['256']:7.1f}/{t['4096']:8.1f}  "
          f"cuBLAS {r['16']['cublas_us']:7.1f}/{r['64']['cublas_us']:7.1f}/{r['256']['cublas_us']:7.1f}/{r['4096']['cublas_us']:8.1f}")
json.dump(table, open("hook3/tuned_gemm.json", "w"), indent=1)
print("per-forward dense GEMM us (M=4 | 32 | 160 | 2048):", {k: round(v) for k, v in tot.items()})
