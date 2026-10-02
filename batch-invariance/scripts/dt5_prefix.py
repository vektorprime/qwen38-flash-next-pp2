"""Prefix-cache and chunk-boundary reproducibility of greedy generation. usage: dt5_prefix.py <base_url> <tag>
For 3 long prompts (3 KLD-corpus windows concatenated, 5000 tokens -> 3 prefill chunks): generate 48 tokens greedily with
logprobs=5, twice. Run 1 computes the prompt from scratch (no cached prefix); run 2 hits the prefix cache (mamba align mode
caches at block boundaries) and recomputes only the tail. With chunk-boundary invariance both must be bit-identical
(tokens and logprobs). Also run 3: the same prompt while 3 other long prompts are prefilling concurrently (chunk ends move
with the load). Writes results/prefix_<tag>.json."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE, TAG = sys.argv[1], sys.argv[2]
M = "qwen38-flash-next-awq"
W = [json.loads(l) for l in open("/home/user/qwen3nextflash/kld/corpus/windows.jsonl")][50:68]   # not used by earlier probes
def gen(toks, n=48):
    p = {"model": M, "prompt": toks, "max_tokens": n, "temperature": 0, "logprobs": 5}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/completions", json.dumps(p).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    ch = r["choices"][0]; lp = ch["logprobs"]
    cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"text": ch["text"], "token_logprobs": lp["token_logprobs"], "top": lp["top_logprobs"], "cached_tokens": cached}
res = {"tag": TAG, "cases": []}
prompts = [sum((W[i]["token_ids"] for i in (3 * j, 3 * j + 1, 3 * j + 2)), [])[:5000] for j in range(3)]
others = [sum((W[i]["token_ids"] for i in (9 + 3 * j, 10 + 3 * j, 11 + 3 * j)), [])[:4500 + 37 * j] for j in range(3)]
for j, P in enumerate(prompts):
    r1 = gen(P); r2 = gen(P)
    with ThreadPoolExecutor(4) as ex:   # same prompt (now fully cached up to the last block) + 3 concurrent long prefills
        fut = [ex.submit(gen, P)] + [ex.submit(gen, O[:len(O) - 11 * j], 8) for O in others]
        r3 = fut[0].result(); [f.result() for f in fut[1:]]
    same = lambda a, b: a["text"] == b["text"] and a["token_logprobs"] == b["token_logprobs"] and a["top"] == b["top"]
    c = {"prompt_len": len(P), "cached_tokens": [r1["cached_tokens"], r2["cached_tokens"], r3["cached_tokens"]],
         "run2_identical": same(r1, r2), "run3_identical": same(r1, r3),
         "max_abs_dlogprob_run2": max(abs(x - y) for x, y in zip(r1["token_logprobs"], r2["token_logprobs"])) if r1["text"] == r2["text"] else None,
         "text_run1": r1["text"][:120]}
    res["cases"].append(c); print(json.dumps(c), flush=True)
res["all_identical"] = all(c["run2_identical"] and c["run3_identical"] for c in res["cases"])
json.dump(res, open(f"/home/user/qwen3nextflash/batchinv/results/prefix_{TAG}.json", "w"), indent=1)
print("prefix/chunk repro", TAG, "all identical:", res["all_identical"], flush=True)
