"""Block until results/gen_<TAG>.json has its t1 part, then print acceptance vs the baseline. usage: wait_t1.py TAG"""
import json, os, sys, time
R = "/home/user/qwen3nextflash/mtp/results/"; tag = sys.argv[1]
while True:
    try:
        g = json.load(open(R + f"gen_{tag}.json"))
        if "t1" in g and g.get("tag") == tag:
            break
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    time.sleep(2)
b = json.load(open(R + "gen_PROD_base.json"))["t1"]["all"]; t = g["t1"]["all"]
for name, x in (("baseline (prod today)", b), (tag, t)):
    print(f"{name:24s} mean_accept_len={x['mean_accept_len']:.3f} per_pos={[round(v, 3) for v in x['per_pos']]} "
          f"accept_rate={x['accept_rate']:.3f} tok/s={x['tok_per_s']:.1f} tokens={x['tokens']}")
print(f"relative change in mean accept len: {100 * (t['mean_accept_len'] / b['mean_accept_len'] - 1):+.1f}%")
