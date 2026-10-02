"""Score the 20 probe windows truncated to L tokens with K prompt logprobs. usage: dt3_score.py <base_url> <tag> <L> <K>
Writes results/s_<tag>.npz (2048-padded arrays like kld/score.py)."""
import json, sys, time, urllib.request
import numpy as np
base, tag, L, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
W = [json.loads(l) for l in open("/home/user/qwen3nextflash/plefp8/e2e/results/probe_windows.jsonl")]
def post(p):
    return json.loads(urllib.request.urlopen(urllib.request.Request(base + "/v1/completions", json.dumps(p).encode(), {"Content-Type": "application/json"}), timeout=900).read())
t0 = time.time(); I, LP, A = [], [], []
for w in W:
    t = w["token_ids"][:L]
    for a in range(4):
        try:
            pl = post({"model": "qwen38-flash-next-awq", "prompt": t, "max_tokens": 1, "temperature": 0, "prompt_logprobs": K})["choices"][0]["prompt_logprobs"]; break
        except Exception as e:  # noqa
            print("retry", a, e, flush=True); time.sleep(5)
    ids = np.zeros((2048, K), np.int32); lp = np.zeros((2048, K), np.float16); ac = np.full((2048,), np.nan, np.float32)
    for i in range(1, L):
        d = pl[i]
        for j, (tok, info) in enumerate(sorted(d.items(), key=lambda kv: kv[1]["logprob"], reverse=True)[:K]):
            ids[i, j] = int(tok); lp[i, j] = info["logprob"]
        if str(t[i]) in d:
            ac[i] = d[str(t[i])]["logprob"]
    I.append(ids); LP.append(lp); A.append(ac)
np.savez_compressed(f"/home/user/qwen3nextflash/batchinv/results/s_{tag}.npz", ids=np.stack(I), logprobs=np.stack(LP), actual=np.stack(A),
                    window_ids=np.array([w["id"] for w in W]), L=np.array([L]), k=np.array([K]))
print(f"score {tag}: L={L} K={K} {len(W)} windows {time.time() - t0:.0f}s", flush=True)
