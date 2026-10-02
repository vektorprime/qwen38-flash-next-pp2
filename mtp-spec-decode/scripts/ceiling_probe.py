"""Greedy-drafting ceiling probe at prod sampling (T=1, top_k=20, top_p=0.95).

For every generated token, take the top-20 logprobs, apply top-k 20 + top-p 0.95
(vLLM order: top-k, renormalize, top-p), and record max p' = the acceptance
probability a perfect greedy drafter (draft = target argmax) would get.
Per request, also diff the server's spec-decode counters; a request is "clean"
when the server's generation-token delta equals our completion tokens (no other
traffic decoded in the window), so its real acceptance is comparable.
"""
import json, math, re, sys, time, urllib.request

B = "http://127.0.0.1:8001"
M = "qwen38-flash-next-awq"
BENCH = open("/home/user/qwen3nextflash/int8top5/bench.py").read()

PROMPTS = [
    "Implement an LRU cache in Python with full type hints, then write pytest tests for it.",
    "Here is a script:\n```python\n" + BENCH + "\n```\nRefactor it into functions with docstrings and argparse "
    "for the base URL, model and output path. Output the complete new file.",
    "Explain how the Raft consensus algorithm handles leader election and log replication, including edge cases.",
    "Find all positive integer solutions of x^2 - 5y^2 = 1 with x < 1000. Show your work.",
    "Write a bash script that rotates log files in a directory: gzip files older than 1 day, delete gz older than 30 days, "
    "with a dry-run flag and logging.",
    "Output a JSON array of 12 objects describing fictional servers: hostname, ip, cpu_cores, ram_gb, role, tags (list). Only JSON.",
    "Write a 400-word short story about a lighthouse keeper who finds a radio that receives tomorrow's weather.",
    "This C function should reverse a singly linked list but has bugs. Find and fix them, explain each:\n```c\n"
    "struct node { int v; struct node *next; };\nstruct node *rev(struct node *h) {\n  struct node *p = NULL, *n;\n"
    "  while (h->next) { n = h->next; h->next = p; p = h; h = n; }\n  return p;\n}\n```",
]


def get(path):
    return urllib.request.urlopen(B + path, timeout=30).read().decode()


def counters():
    t = get("/metrics")
    out = {}
    for k in ["spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
              "spec_decode_num_accepted_tokens_total", "generation_tokens_total", "num_requests_running"]:
        m = re.search(r"^vllm:%s\{[^}]*\} (\S+)" % k, t, re.M)
        out[k] = float(m.group(1)) if m else float("nan")
    for pos in range(3):
        m = re.search(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="%d"[^}]*\} (\S+)' % pos, t, re.M)
        out["pos%d" % pos] = float(m.group(1))
    return out


def processed(top):
    lps = sorted((x["logprob"] for x in top), reverse=True)[:20]
    ps = [math.exp(l) for l in lps]
    z = sum(ps)
    ps = [p / z for p in ps]  # top-k 20 renormalized
    keep, c = [], 0.0
    for p in ps:  # top-p 0.95 on the top-k distribution
        keep.append(p)
        c += p
        if c >= 0.95:
            break
    z = sum(keep)
    keep = [p / z for p in keep]
    ent = -sum(p * math.log(p) for p in keep if p > 0)
    return keep[0], ent, len(keep)


rows = []
for i, p in enumerate(PROMPTS):
    c0 = counters()
    body = {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": 1500,
            "logprobs": True, "top_logprobs": 20}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        B + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t
    c1 = counters()
    lp = r["choices"][0]["logprobs"]["content"]
    ntok = r["usage"]["completion_tokens"]
    stats = [processed(x["top_logprobs"]) for x in lp]
    # split reasoning vs answer at the </think> token if present
    toks = [x["token"] for x in lp]
    cut = next((j for j, s in enumerate(toks) if "</think>" in s), len(toks))
    d = {k: c1[k] - c0[k] for k in c0 if k != "num_requests_running"}
    clean = abs(d["generation_tokens_total"] - ntok) < 0.5 and c0["num_requests_running"] == 0
    row = {
        "i": i, "tokens": ntok, "secs": round(dt, 1), "clean": clean,
        "running_before": c0["num_requests_running"],
        "gen_delta": d["generation_tokens_total"],
        "oracle_greedy_maxp": sum(s[0] for s in stats) / len(stats),
        "oracle_greedy_maxp_reasoning": (sum(s[0] for s in stats[:cut]) / cut) if cut else None,
        "oracle_greedy_maxp_answer": (sum(s[0] for s in stats[cut + 1:]) / max(1, len(stats) - cut - 1)) if cut < len(stats) - 1 else None,
        "n_reasoning": cut,
        "mean_entropy_nats": sum(s[1] for s in stats) / len(stats),
        "frac_maxp_ge_0.9": sum(s[0] >= 0.9 for s in stats) / len(stats),
        "frac_maxp_lt_0.5": sum(s[0] < 0.5 for s in stats) / len(stats),
        "mean_nucleus_size": sum(s[2] for s in stats) / len(stats),
        "actual_pos0": d["pos0"] / d["spec_decode_num_drafts_total"] if d["spec_decode_num_drafts_total"] else None,
        "actual_pos1": d["pos1"] / d["spec_decode_num_drafts_total"] if d["spec_decode_num_drafts_total"] else None,
        "actual_pos2": d["pos2"] / d["spec_decode_num_drafts_total"] if d["spec_decode_num_drafts_total"] else None,
        "actual_accept_rate": d["spec_decode_num_accepted_tokens_total"] / d["spec_decode_num_draft_tokens_total"] if d["spec_decode_num_draft_tokens_total"] else None,
        "actual_mean_accept_len": 1 + d["spec_decode_num_accepted_tokens_total"] / d["spec_decode_num_drafts_total"] if d["spec_decode_num_drafts_total"] else None,
        "maxp_series": [round(s[0], 4) for s in stats],
    }
    rows.append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "maxp_series"}), flush=True)

json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "ceiling_probe.json", "w"))
