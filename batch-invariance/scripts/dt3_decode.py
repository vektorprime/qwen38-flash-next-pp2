"""Greedy decode reproducibility under concurrency. usage: dt3_decode.py <base_url> <tag>
16 fixed prompts (int8top5/bench.py), 256 tokens, temperature 0, chat API, thinking off.
  seq1, seq2 : one request at a time (identical batches -> must match exactly)
  conc8      : 8 requests in flight at a time (decode batches of 1..8 sequences, mixed with other requests' prefills)
  conc4+load : 4 at a time plus 2 background streams generating long answers
Reports, vs seq1: identical outputs, and for differing ones the first differing character / token-ish position."""
import json, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE, TAG = sys.argv[1], sys.argv[2]
M = "qwen38-flash-next-awq"
P = eval(open("/home/user/qwen3nextflash/int8top5/bench.py").read().split("P=", 1)[1].split("\ndef metrics")[0])

def ask(p, max_tokens=256):
    body = json.dumps({"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=900))
    return r["choices"][0]["message"]["content"]

def run_seq():
    return [ask(p) for p in P]

def run_conc(n):
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(ask, P))

def run_load():
    stop = threading.Event()
    def bg(i):
        while not stop.is_set():
            try: ask(f"Write a very long, detailed essay (part {i}) about the history of mathematics.", 1500)
            except Exception: time.sleep(1)
    ts = [threading.Thread(target=bg, args=(i,), daemon=True) for i in range(2)]
    [t.start() for t in ts]; time.sleep(3)
    try:
        return run_conc(4)
    finally:
        stop.set()

def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n if len(a) != len(b) else None)

t0 = time.time()
runs = {"seq1": run_seq(), "seq2": run_seq(), "conc8": run_conc(8), "conc4+load": run_load()}
res = {"tag": TAG, "seconds": round(time.time() - t0)}
for k in ("seq2", "conc8", "conc4+load"):
    d = [first_diff(a, b) for a, b in zip(runs["seq1"], runs[k])]
    res[f"{k}_identical_to_seq1"] = sum(x is None for x in d)
    res[f"{k}_first_diff_chars"] = [x for x in d if x is not None]
json.dump({"res": res, "outputs": runs}, open(f"/home/user/qwen3nextflash/batchinv/results/decode_{TAG}.json", "w"), indent=1)
print(json.dumps(res))
