"""Probe the scheduler-padded first step (1-token prompt while 6 requests decode) for errors / NaN fields.
usage: pad_probe.py <base_url> <tag> [--n 12] [--nologprobs]
Prints, per trial, whether the request succeeded, the error text, and which response fields hold NaN (requests are
retried once with logprobs off when they fail, to localise the NaN). Writes results/padprobe_<tag>.json."""
import argparse, json, math, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("tag")
ap.add_argument("--n", type=int, default=12); ap.add_argument("--nologprobs", action="store_true")
A = ap.parse_args()
M = "qwen38-flash-next-awq"
corpus = [json.loads(l) for l in open("/home/user/qwen3nextflash/kld/corpus/windows.jsonl")]


def raw_post(body, timeout=1800):
    req = urllib.request.Request(A.url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return 200, urllib.request.urlopen(req, timeout=timeout).read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


stop = threading.Event()


def background(seed):
    while not stop.is_set():
        raw_post({"model": M, "prompt": corpus[(40 + seed) % len(corpus)]["token_ids"][:600], "max_tokens": 3000,
                  "temperature": 1.0, "seed": seed, "ignore_eos": True}, timeout=3600)
        seed += 6


bgs = [threading.Thread(target=background, args=(s,), daemon=True) for s in range(6)]
for b in bgs:
    b.start()
time.sleep(8)
out = []
for j in range(A.n):
    body = {"model": M, "prompt": corpus[j]["token_ids"][:1], "max_tokens": 6, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
            "seed": 77 + j}
    if not A.nologprobs:
        body.update(logprobs=20, return_tokens_as_token_ids=True)
    code, txt = raw_post(body)
    rec = dict(j=j, code=code, logprobs=not A.nologprobs)
    if code == 200:
        r = json.loads(txt)
        rec["text"] = r["choices"][0]["text"]; rec["metrics"] = r.get("metrics")
    else:
        rec["error"] = txt[:300]
        code2, txt2 = raw_post({k: v for k, v in body.items() if k not in ("logprobs", "return_tokens_as_token_ids")})
        rec["retry_without_logprobs"] = code2
        if code2 == 200:
            r2 = json.loads(txt2); rec["retry_text"] = r2["choices"][0]["text"]; rec["retry_metrics"] = r2.get("metrics")
        else:
            rec["retry_error"] = txt2[:300]
    out.append(rec)
    print(json.dumps(rec)[:400], flush=True)
stop.set()
json.dump(out, open(f"/home/user/qwen3nextflash/mtp/results/padprobe_{A.tag}.json", "w"), indent=1)
print("summary:", sum(r["code"] == 200 for r in out), "/", len(out), "ok")
