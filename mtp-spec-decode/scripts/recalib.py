"""Recompute the T=1 calibration of a gen_suite run from its saved raw logprobs. usage: recalib.py TAG [TAG ...]"""
import json, math, sys
import numpy as np
src = open("/home/user/qwen3nextflash/mtp/gen_suite.py").read(); ns = {}
exec(src[src.index("def tok_id"):src.index("def part_t1():")], {"np": np, "math": math}, ns)
R = "/home/user/qwen3nextflash/mtp/results/"
for tag in sys.argv[1:]:
    z = np.load(R + f"gen_{tag}_tokens.npz")
    c = ns["calib_from_raw"](z["sid"], z["ids"], z["lps"])
    g = json.load(open(R + f"gen_{tag}.json")); g["t1"]["calibration"] = c
    json.dump(g, open(R + f"gen_{tag}.json", "w"), indent=1)
    print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k not in ("pit_deciles", "outside_examples")})
