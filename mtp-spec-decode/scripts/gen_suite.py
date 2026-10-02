"""Generation-side validation for spec-decode changes (the sampling counterpart of KLD / PPL / same-top-p).

usage: python3 gen_suite.py <base_url> <tag> [--parts t1,pad,greedy] [--n-t1 24] [--max-tokens 1500]
writes results/gen_<tag>.json (summary) and results/gen_<tag>_tokens.npz (per-token data)

Parts
  t1      single-stream T=1 benchmark at prod sampling (top_k 20, top_p 0.95, thinking on), fixed prompts + seeds,
          top-20 logprobs. Acceptance / tok/s from the server's spec counters (clean only when nothing else ran),
          plus per-token calibration of the sampled token against the target's processed distribution p':
            outside_nucleus   sampled tokens with p' = 0 (must be 0)
            nll / exp_nll     mean -log p'(sampled) vs its expectation H(p') (generation "perplexity"), z-score
            argmax / exp      rate of sampling p' argmax vs mean max p', z-score
            pit               randomized PIT, uniform iff tokens are exact draws from p' (KS + decile chi2)
  pad     #58784 stress: 1-token prompts (the only "1 token left" case here: with MTP the prefix cache drops one block
          from every hit) sent while two background requests decode -> scheduler pads the step with placeholder drafts;
          top_p 1 / top_k -1 (no nucleus protection). Counts first tokens equal to id 0 ('!'), vs the target's p('!').
  greedy  the 16 int8top5/bench.py prompts at T=0, thinking off, 256 tokens; outputs must be byte-identical across
          stages (drafting is argmax at T=0 in every mode).
"""
import argparse, json, math, re, threading, time, urllib.error, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("tag")
ap.add_argument("--parts", default="t1,cal,pad,greedy")
ap.add_argument("--n-t1", type=int, default=24)
ap.add_argument("--max-tokens", type=int, default=1500)
ap.add_argument("--n-pad", type=int, default=24)
ap.add_argument("--model", default="qwen38-flash-next-awq")
A = ap.parse_args()
R = "/home/user/qwen3nextflash/mtp/results/"
M = A.model

T1_PROMPTS = [
    "Implement an LRU cache in Python with full type hints, then write pytest tests for it.",
    "Explain how the Raft consensus algorithm handles leader election and log replication, including edge cases.",
    "Find all positive integer solutions of x^2 - 5y^2 = 1 with x < 1000. Show your work.",
    "Write a bash script that rotates log files in a directory: gzip files older than 1 day, delete gz older than 30 days, with a dry-run flag.",
    "Output a JSON array of 12 objects describing fictional servers: hostname, ip, cpu_cores, ram_gb, role, tags. Only JSON.",
    "Write a 400-word short story about a lighthouse keeper who finds a radio that receives tomorrow's weather.",
    "This C function should reverse a singly linked list but has bugs. Find and fix them:\n```c\nstruct node { int v; struct node *next; };\n"
    "struct node *rev(struct node *h) {\n  struct node *p = NULL, *n;\n  while (h->next) { n = h->next; h->next = p; p = h; h = n; }\n  return p;\n}\n```",
    "Derive the closed form of the sum of k^2 for k=1..n and prove it by induction.",
    "Write a Rust function that parses a CSV line with quoted fields and escaped quotes. Include unit tests.",
    "Compare B-trees and LSM-trees for write-heavy workloads. Be concrete about amplification factors.",
    "Write a SQL query for the top 3 products by revenue per month, and explain window functions used.",
    "Explain the attention mechanism in transformers mathematically, including the scaling factor.",
    "A train leaves at 3:15 pm at 80 km/h, another at 4:00 pm at 110 km/h from the same station. When does the second catch up?",
    "Write a Python asyncio web crawler with a concurrency limit, retries and a max depth.",
    "Summarize the causes and consequences of the 2008 financial crisis.",
    "Write a Dockerfile and docker-compose.yml for a Flask app with Postgres and Redis.",
    "Explain how TCP congestion control works (slow start, congestion avoidance, fast recovery).",
    "Write a TypeScript React hook useDebouncedValue with tests using vitest.",
    "Prove that there are infinitely many primes of the form 4k+3.",
    "Translate this to French and German: 'The meeting has been moved to Thursday because the projector is broken.'",
    "Design a rate limiter for an API gateway. Discuss token bucket vs sliding window and give code.",
    "Explain Kalman filters with the predict/update equations and a 1D example with numbers.",
    "Write a Go program that watches a directory and prints file changes, with graceful shutdown.",
    "What are the tradeoffs of microservices versus a modular monolith? Give a decision checklist.",
]
GREEDY_PROMPTS = re.findall(r'"([^"]+)"', open("/home/user/qwen3nextflash/int8top5/bench.py").read().split("P=[", 1)[1].split("]", 1)[0])
assert len(GREEDY_PROMPTS) == 16, len(GREEDY_PROMPTS)


def post(path, body, timeout=1800):
    req = urllib.request.Request(A.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None


def counters():
    t = urllib.request.urlopen(A.url + "/metrics", timeout=30).read().decode()
    g = lambda k: float(re.search(r"^vllm:%s\{[^}]*\} (\S+)" % k, t, re.M).group(1))
    out = {k: g(k) for k in ["spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
                             "spec_decode_num_accepted_tokens_total", "generation_tokens_total"]}
    out["running"] = g("num_requests_running") + g("num_requests_waiting")
    for p in range(3):
        m = re.search(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="%d"[^}]*\} (\S+)' % p, t, re.M)
        out["pos%d" % p] = float(m.group(1)) if m else 0.0
    return out


def tok_id(t):
    return int(t.split(":", 1)[1]) if isinstance(t, str) and t.startswith("token_id:") else t


def raw_rows(lp_content):
    """Raw per-token record: sampled id, its logprob, top-20 ids/logprobs ordered by (logprob desc, id asc).
    vLLM always lists the sampled token first in top_logprobs (also inside exact ties), so the order is rebuilt."""
    sid, slp, ids, lps = [], [], [], []
    for x in lp_content:
        d = {}
        for t in x["top_logprobs"]:
            d.setdefault(tok_id(t["token"]), t["logprob"])
        items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        sid.append(tok_id(x["token"])); slp.append(x["logprob"])
        ids.append([i for i, _ in items] + [-1] * (20 - len(items)))
        lps.append([v for _, v in items] + [-1e9] * (20 - len(items)))
    return np.array(sid), np.array(slp), np.array(ids), np.array(lps)


def calib_from_raw(sid, ids, lps, k=20, top_p=0.95, seed=12345):
    """Calibration of sampled tokens against the target's processed distribution p' (top-k then top-p, vLLM rule).
    Only positions where p' is exactly reconstructible from the visible top-20 are evaluated: the top-p cut falls
    before rank 19 and the last kept token is not tied with the next one (bf16 logits give exact ties; the sampler's
    sort order decides which tied token survives). The filter depends on the target distribution only, never on the
    sampled token, so it cannot bias the statistics. Argmax counts any token tied at the max."""
    rng = np.random.default_rng(seed)
    rows, outside_ex = [], []
    n_skip = 0
    for n in range(len(sid)):
        lp = lps[n][:20]; tid = ids[n][:20]
        ps = np.exp(lp.astype(np.float64))
        if k is not None:
            ps = ps / ps.sum()                       # top-k renormalisation (approximate: rank-20 ties are invisible)
        csum_before = np.concatenate([[0.0], np.cumsum(ps)[:-1]])
        keep = csum_before < top_p
        last = int(np.nonzero(keep)[0].max())
        if k is not None:
            if last >= 18 or ps[last] == ps[last + 1]:
                n_skip += 1
                continue
        else:                                       # exact: raw logprobs are full-vocab normalised
            nxt = csum_before[last + 1] if last + 1 < 20 else 1.0
            if last >= 19 or ps[last] == ps[last + 1] or abs(nxt - top_p) < 1e-4 or abs(csum_before[last] - top_p) < 1e-4:
                n_skip += 1
                continue
        pk = np.where(keep, ps, 0.0); pk = pk / pk.sum()
        pmax = pk.max(); n_max = int((pk == pmax).sum())
        H = float(-(pk[pk > 0] * np.log(pk[pk > 0])).sum())
        where = np.nonzero(tid == sid[n])[0]
        if len(where) and keep[where[0]]:
            p_s = pk[where[0]]
            before = pk[(pk > p_s) | ((pk == p_s) & (tid < sid[n]))].sum()
            u = before + rng.random() * p_s
            out = 0.0
        else:
            p_s, u = 0.0, float("nan")
            # strict = not in the visible top-20 or >= 2 ranks past the cut; rank last+1 can be a legitimate keep
            # when tokens tied at rank 20 (invisible) change the top-k normalisation
            out = 2.0 if (not len(where) or where[0] > last + 1) else 1.0
            if len(outside_ex) < 50:
                outside_ex.append(dict(token=int(sid[n]), rank=int(where[0]) if len(where) else -1,
                                       cum_before=float(csum_before[where[0]]) if len(where) else None, last_kept=last))
        rows.append((p_s, pmax, H, u, out, float(p_s > 0 and p_s == pmax), pmax * n_max))
    a = np.array(rows)
    p_s, pmax, H, u, out, isarg, exparg = a.T
    ok = p_s > 0
    nll = -np.log(p_s[ok])
    z_nll = (nll.mean() - H[ok].mean()) / (nll.std() / math.sqrt(len(nll)))
    z_arg = (isarg.sum() - exparg.sum()) / math.sqrt((exparg * (1 - exparg)).sum())
    uu = np.sort(u[~np.isnan(u)])
    ks = float(np.max(np.abs(np.arange(1, len(uu) + 1) / len(uu) - uu)))
    hist = np.histogram(uu, bins=10, range=(0, 1))[0]
    return dict(n_tokens=int(len(sid)), n_evaluated=int(len(a)), n_not_reconstructible=int(n_skip),
                outside_nucleus=int((out > 0).sum()), outside_strict=int((out > 1).sum()),
                nll_mean=float(nll.mean()), exp_nll_mean=float(H[ok].mean()), z_nll=float(z_nll),
                argmax_rate=float(isarg.mean()), exp_argmax_rate=float(exparg.mean()), z_argmax=float(z_arg),
                pit_ks=ks, pit_ks_p=float(min(1.0, 2 * math.exp(-2 * len(uu) * ks * ks))),
                pit_decile_chi2_df9=float(((hist - len(uu) / 10) ** 2 / (len(uu) / 10)).sum()), pit_deciles=hist.tolist(),
                outside_examples=outside_ex)


def part_t1():
    reqs, raws = [], []
    c_start = counters()
    for i, p in enumerate(T1_PROMPTS[: A.n_t1]):
        c0 = counters(); t = time.time()
        r = post("/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": A.max_tokens,
                                          "seed": 1000 + i, "logprobs": True, "top_logprobs": 20,
                                          "return_tokens_as_token_ids": True})
        dt = time.time() - t; c1 = counters()
        ntok = r["usage"]["completion_tokens"]
        raws.append(raw_rows(r["choices"][0]["logprobs"]["content"]))
        d = {k: c1[k] - c0[k] for k in c0 if k != "running"}
        clean = c0["running"] == 0 and abs(d["generation_tokens_total"] - ntok) < 0.5
        reqs.append(dict(i=i, tokens=ntok, secs=dt, clean=clean, drafts=d["spec_decode_num_drafts_total"],
                         draft_tokens=d["spec_decode_num_draft_tokens_total"], accepted=d["spec_decode_num_accepted_tokens_total"],
                         pos=[d["pos0"], d["pos1"], d["pos2"]]))
        print(f"  t1 {i:2d} tok={ntok} {ntok / dt:5.1f} tok/s clean={clean} "
              f"acc_len={1 + d['spec_decode_num_accepted_tokens_total'] / max(1, d['spec_decode_num_drafts_total']):.2f}", flush=True)
    cl = [q for q in reqs if q["clean"]]
    agg = lambda qs, k: sum(q[k] for q in qs)
    res = {}
    for name, qs in (("all", reqs), ("clean", cl)):
        if not qs:
            continue
        dr, dt_, ac = agg(qs, "drafts"), agg(qs, "draft_tokens"), agg(qs, "accepted")
        res[name] = dict(n_req=len(qs), tokens=agg(qs, "tokens"), secs=agg(qs, "secs"),
                         tok_per_s=agg(qs, "tokens") / agg(qs, "secs"),
                         accept_rate=ac / dt_ if dt_ else None, mean_accept_len=1 + ac / dr if dr else None,
                         per_pos=[sum(q["pos"][j] for q in qs) / dr for j in range(3)] if dr else None)
    sid = np.concatenate([x[0] for x in raws]); slp = np.concatenate([x[1] for x in raws])
    ids = np.concatenate([x[2] for x in raws]); lps = np.concatenate([x[3] for x in raws])
    np.savez_compressed(R + f"gen_{A.tag}_tokens.npz", sid=sid, slp=slp, ids=ids, lps=lps,
                        req_len=np.array([len(x[0]) for x in raws]))
    res["calibration"] = calib_from_raw(sid, ids, lps)
    res["requests"] = reqs
    return res


def part_cal():
    """Exact sampler check: top_k disabled, top_p 0.95 -> the kept set is exactly reconstructible from raw logprobs."""
    raws = []
    for i, p in enumerate(T1_PROMPTS[:12]):
        r = post("/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": A.max_tokens,
                                          "seed": 2000 + i, "top_k": -1, "top_p": 0.95, "logprobs": True, "top_logprobs": 20,
                                          "return_tokens_as_token_ids": True})
        raws.append(raw_rows(r["choices"][0]["logprobs"]["content"]))
        print(f"  cal {i:2d} tok={len(raws[-1][0])}", flush=True)
    sid = np.concatenate([x[0] for x in raws]); ids = np.concatenate([x[2] for x in raws]); lps = np.concatenate([x[3] for x in raws])
    np.savez_compressed(R + f"gen_{A.tag}_cal_tokens.npz", sid=sid, slp=np.concatenate([x[1] for x in raws]), ids=ids, lps=lps)
    return dict(calibration=calib_from_raw(sid, ids, lps, k=None))


def part_pad():
    corpus = [json.loads(l) for l in open("/home/user/qwen3nextflash/kld/corpus/windows.jsonl")]
    stop = threading.Event()

    def background(seed):
        while not stop.is_set():
            try:
                post("/v1/completions", {"model": M, "prompt": corpus[(40 + seed) % len(corpus)]["token_ids"][:600], "max_tokens": 3000,
                                         "temperature": 1.0, "seed": seed, "ignore_eos": True}, timeout=3600)
            except Exception as e:  # noqa
                print("  bg error", e, flush=True); time.sleep(2)
            seed += 6
    bgs = [threading.Thread(target=background, args=(s,), daemon=True) for s in range(6)]  # 6 decoding: PP3 keeps 2 per micro-batch
    for b in bgs:
        b.start()
    time.sleep(8)
    trials, errors = [], []
    for j in range(A.n_pad):
        # A 1-token prompt is the only way to get "1 token left" in this deployment: with MTP the prefix cache drops
        # one 1600-token block from every hit (use_eagle), so repeats never leave a 1-token tail.
        ids = corpus[j]["token_ids"][:1]
        body = {"model": M, "prompt": ids, "max_tokens": 6, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "seed": 77 + j, "logprobs": 20, "return_tokens_as_token_ids": True}
        try:
            r = post("/v1/completions", body)
        except Exception as e:  # noqa: record and continue; the gate fails if trials error
            print(f"  pad {j:2d} ERROR {e}", flush=True)
            errors.append(dict(j=j, error=str(e)))
            continue
        cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        cached_w = None
        lp = r["choices"][0]["logprobs"]
        toks = [int(t.split(":")[1]) for t in lp["tokens"]]
        first_top = lp["top_logprobs"][0] or {}
        p_bang = math.exp(first_top["token_id:0"]) if "token_id:0" in first_top else 0.0
        trials.append(dict(j=j, tokens=toks, cached_tokens=cached, cached_tokens_warm=cached_w, first_is_bang=toks[0] == 0,
                           bang_run=int(np.argmin(np.array(toks + [1]) == 0)), p_bang_first=p_bang))
        print(f"  pad {j:2d} cached={cached} first={toks[:6]} p(!)={p_bang:.2e}", flush=True)
    stop.set()
    n_bang = sum(t["first_is_bang"] for t in trials)
    return dict(n=len(trials), errors=errors, first_is_bang=n_bang, cached_tokens=[t["cached_tokens"] for t in trials], expected_bang=sum(t["p_bang_first"] for t in trials),
                max_bang_run=max([t["bang_run"] for t in trials] or [0]), trials=trials)


def part_greedy():
    outs = []
    for p in GREEDY_PROMPTS:
        r = post("/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": 256,
                                          "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        outs.append(r["choices"][0]["message"]["content"])
    return dict(outputs=outs, sha=[__import__("hashlib").sha256(o.encode()).hexdigest()[:16] for o in outs])


import os
_path = R + f"gen_{A.tag}.json"
summary = json.load(open(_path)) if os.path.exists(_path) else {}
summary.update(dict(tag=A.tag, url=A.url, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
for part in A.parts.split(","):
    t = time.time()
    print(f"[{A.tag}] part {part}", flush=True)
    summary[part] = {"t1": part_t1, "cal": part_cal, "pad": part_pad, "greedy": part_greedy}[part]()
    summary[part + "_secs"] = time.time() - t
    json.dump(summary, open(R + f"gen_{A.tag}.json", "w"), indent=1)
s = {k: v for k, v in summary.get("t1", {}).items() if k != "requests"}
cal = {k: v for k, v in summary.get("cal", {}).get("calibration", {}).items() if k not in ("pit_deciles", "outside_examples")}
print(json.dumps(dict(tag=A.tag, t1=s, cal=cal, pad={k: v for k, v in summary.get("pad", {}).items() if k != "trials"}), indent=1))
