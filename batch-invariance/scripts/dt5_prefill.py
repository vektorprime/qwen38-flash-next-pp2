"""Prefill latency: 6 distinct 2048-token prompts (KLD corpus windows 70..75, never otherwise sent as plain prompts),
max_tokens=1, sequential. usage: dt5_prefill.py <base_url> <tag>  -> results/prefill_<tag>.json"""
import json, statistics, sys, time, urllib.request
BASE, TAG = sys.argv[1], sys.argv[2]
W = [json.loads(l) for l in open("/home/user/qwen3nextflash/kld/corpus/windows.jsonl")][70:76]
ts = []
for w in W:
    p = {"model": "qwen38-flash-next-awq", "prompt": w["token_ids"][:2048], "max_tokens": 1, "temperature": 0}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/completions", json.dumps(p).encode(), {"Content-Type": "application/json"}), timeout=600).read())
    ts.append(time.time() - t)
res = {"tag": TAG, "latency_s": [round(x, 3) for x in ts], "median_s": round(statistics.median(ts), 3), "prefill_tok_per_s": round(2048 / statistics.median(ts), 1)}
json.dump(res, open(f"/home/user/qwen3nextflash/batchinv/results/prefill_{TAG}.json", "w"), indent=1)
print(json.dumps(res), flush=True)
