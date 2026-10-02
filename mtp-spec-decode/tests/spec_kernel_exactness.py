"""Exactness of speculative sampling with this image's kernels (patched spec image), offline on one GPU.

Simulates k=3 MTP verification steps for N independent sequences until each has G=4 output tokens, using the real
draft-sampling path (DraftModelSpeculator._shape_draft_logits + gumbel_sample with the logits cache) and the real
rejection_sample kernel. Target/draft distributions are fixed per output position (adversarial pairs spread over
several vocab blocks), so under lossless sampling every output position must be an exact draw from its target p.
Reports per-position chi-square p-value and TV(empirical, p), pairwise independence of positions 0/1, and mean
accepted drafts per step.

docker run --rm --gpus '"device=2"' --memory 14g --entrypoint python3 -v $PWD:/w <spec image> /w/spec_kernel_exactness.py
"""
import json, math, sys
import numpy as np
import torch
import vllm.v1.worker.gpu.spec_decode.speculator as specmod
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as rsu
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

dev = "cuda"
V = 32768                    # 4 rejection vocab blocks (8192), 32 gumbel blocks (1024)
K = 3
G = 4
N = 4096
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SUP = torch.tensor([5, 9001, 17003, 30011, 12345, 24680], device=dev)   # support tokens in different blocks

# per-output-position target p (processed, i.e. already top-k/top-p truncated) and draft q (with extra tail mass on
# tokens outside the target support, which the draft top-k/top-p mask should remove)
P = [[.10, .45, .45, 0, 0, 0], [.20, .40, .40, 0, 0, 0], [.05, .50, .30, .15, 0, 0], [.60, .25, .15, 0, 0, 0],
     [.30, .30, .20, .20, 0, 0], [.90, .10, 0, 0, 0, 0], [.40, .35, .20, .05, 0, 0]]
Q = [[.70, .17, .01, 0, .07, .05], [.55, .27, .08, 0, .06, .04], [.62, .22, .04, .01, .06, .05], [.45, .30, .10, .05, .06, .04],
     [.25, .25, .25, .15, .06, .04], [.70, .20, .04, 0, .03, .03], [.30, .30, .30, .02, .05, .03]]
NP = len(P)


def to_logits(probs):
    lg = torch.full((V,), float("-inf"), device=dev)
    pr = torch.tensor(probs, device=dev, dtype=torch.float32)
    nz = pr > 0
    lg[SUP[nz]] = pr[nz].log()
    return lg


TL = torch.stack([to_logits(p) for p in P])                       # [NP, V] target processed logits (fp32)
QL = torch.stack([to_logits(q) for q in Q])                       # [NP, V] draft logits
QL = torch.where(torch.isinf(QL), torch.full_like(QL, -30.0), QL)  # draft has (tiny) mass everywhere, like a real head
QL = QL.to(torch.bfloat16)                                         # head dtype


def chi2_sf(x, k):
    return float(torch.special.gammaincc(torch.tensor(k / 2.0, dtype=torch.float64), torch.tensor(x / 2.0, dtype=torch.float64)))


class FakeSpec:
    pass


def run_variant(name, draft_mode, indep_noise, topkp, tau, block, top_k=3, top_p=0.9, fresh_step_seed=False, step_unique=False):
    specmod._DRAFT_TOPKP = topkp
    specmod._STEP_UNIQUE_RNG = step_unique
    rsu.STEP_UNIQUE_RNG = step_unique
    specmod._DRAFT_TAU = tau
    fs = FakeSpec()
    fs.top_k = torch.full((N,), top_k, dtype=torch.int32, device=dev)
    fs.top_p = torch.full((N,), top_p, dtype=torch.float32, device=dev)
    counts = np.zeros((G, len(SUP)), np.int64)
    pair = np.zeros((len(SUP), len(SUP)), np.int64)
    acc_tot, steps_tot = 0, 0
    sup_index = {int(t): i for i, t in enumerate(SUP.tolist())}
    for rep in range(REPS):
        g = torch.Generator(device=dev); g.manual_seed(1000 + rep)
        seeds = torch.randint(0, 2**62, (N,), device=dev, generator=g, dtype=torch.int64)
        draft_seeds = seeds ^ specmod._DRAFT_SEED_XOR if indep_noise else seeds.clone()
        temp = torch.ones(N, dtype=torch.float32, device=dev)
        idx = torch.arange(N, dtype=torch.int32, device=dev)
        emitted = torch.zeros(N, dtype=torch.int64, device=dev)
        out = torch.full((N, G + K + 1), -1, dtype=torch.int64, device=dev)
        base = 100_000 * (rep + 1)
        draft_cache = torch.zeros(N, K, V, dtype=torch.bfloat16, device=dev) if draft_mode == "prob" else None
        seeds0, dseeds0 = seeds.clone(), draft_seeds.clone()
        for step in range(G):
            if fresh_step_seed:   # diagnostic: unique RNG key per verification step
                seeds = seeds0 ^ ((step + 1) * 0x9E3779B97F4A7C15 % 2**62)
                draft_seeds = dseeds0 ^ ((step + 1) * 0x9E3779B97F4A7C15 % 2**62)
            c = emitted.clone()                                     # output position of the first draft
            pos_rows = base + c[:, None] + torch.arange(K + 1, device=dev)[None, :]   # [N, K+1] row input positions
            drafts = torch.empty(N, K, dtype=torch.int64, device=dev)
            for j in range(K):
                qi = ((c + j) % NP)
                lg = QL[qi]                                         # [N, V] bf16
                if draft_mode == "prob":
                    if topkp or tau != 1.0:
                        lg = specmod.DraftModelSpeculator._shape_draft_logits(fs, lg, idx.long(), temp, torch.bfloat16)
                    col = torch.tensor(j, device=dev, dtype=torch.int64)
                    drafts[:, j] = gumbel_sample(lg, idx, temp, draft_seeds, specmod.draft_noise_positions(pos_rows[:, j].contiguous(), col), apply_temperature=True,
                                                 logits_cache=draft_cache, logits_cache_col=col)
                else:
                    drafts[:, j] = lg.float().argmax(-1)
            ti = (c[:, None] + torch.arange(K + 1, device=dev)[None, :]) % NP           # [N, K+1]
            tlog = TL[ti.reshape(-1)].contiguous()                                          # [N*(K+1), V]
            ds = torch.cat([torch.full((N, 1), 7, dtype=torch.int64, device=dev), drafts], 1).reshape(-1).contiguous()
            cu = torch.arange(0, N * (K + 1) + 1, K + 1, dtype=torch.int32, device=dev)
            exp_idx = idx.repeat_interleave(K + 1)
            exp_pos = torch.arange(K + 1, dtype=torch.int32, device=dev).repeat(N)
            sampled, num_sampled = rejection_sample(tlog, draft_cache, ds, cu, pos_rows.reshape(-1).contiguous(), idx, exp_idx,
                                                    exp_pos, temp, seeds, K, use_block_verification=block)
            sampled = sampled.view(N, -1)
            ns = num_sampled.long()
            for t in range(K + 1):
                m = (t < ns) & (emitted + t < out.shape[1])
                rows = torch.nonzero(m).squeeze(1)
                out[rows, (emitted + t)[rows]] = sampled[rows, t].long()
            acc_tot += int((ns - 1).sum()); steps_tot += N
            emitted = emitted + ns
            del tlog
        o = out[:, :G].cpu().numpy()
        for gpos in range(G):
            for t, i in sup_index.items():
                counts[gpos, i] += int((o[:, gpos] == t).sum())
        bad = ~np.isin(o, list(sup_index))
        assert not bad.any(), f"{name}: tokens outside the support: {np.unique(o[bad])[:10]}"
        for a_, b_ in zip(o[:, 0], o[:, 1]):
            pair[sup_index[int(a_)], sup_index[int(b_)]] += 1
    rows = []
    for gpos in range(G):
        p = np.array(P[gpos % NP], dtype=np.float64)
        n = counts[gpos].sum()
        e = p * n
        nz = p > 0
        chi2 = float(((counts[gpos][nz] - e[nz]) ** 2 / e[nz]).sum())
        df = int(nz.sum()) - 1
        pval = chi2_sf(chi2, df) if df > 0 else 1.0
        tv = float(0.5 * np.abs(counts[gpos] / n - p).sum())
        rows.append(dict(pos=gpos, chi2=round(chi2, 2), df=df, p_value=round(pval, 4), tv=round(tv, 4)))
    # independence of positions 0 and 1
    n = pair.sum()
    e = np.outer(pair.sum(1), pair.sum(0)) / n
    nz = e > 0
    chi2i = float(((pair[nz] - e[nz]) ** 2 / e[nz]).sum())
    dfi = (int((pair.sum(1) > 0).sum()) - 1) * (int((pair.sum(0) > 0).sum()) - 1)
    pind = chi2_sf(chi2i, dfi)
    res = dict(variant=name, draft=draft_mode, indep_noise=indep_noise, topkp=topkp, tau=tau, block=block, step_unique=step_unique,
               sequences=int(N * REPS), mean_accepted_per_step=round(acc_tot / steps_tot, 4),
               min_p_value=min(r["p_value"] for r in rows), max_tv=max(r["tv"] for r in rows),
               indep_p01=round(pind, 4), positions=rows)
    print(json.dumps({k: v for k, v in res.items() if k != "positions"}), flush=True)
    return res


variants = [
    # name, draft, indep_noise, topkp, tau, block, top_k, top_p, fresh_step_seed, step_unique
    ("V0 prod today: greedy drafts, token-wise", "greedy", False, False, 1.0, False),
    ("V1 stock probabilistic (shared noise)", "prob", False, False, 1.0, False),
    ("V2 A: probabilistic + independent noise", "prob", True, False, 1.0, False),
    ("V3 A+B: + draft top-k/top-p", "prob", True, True, 1.0, False),
    ("V4 A+B: tau 0.7", "prob", True, True, 0.7, False),
    ("V5 A+B+C: block, stock RNG keys", "prob", True, True, 1.0, True),
    ("V6 A+B+C: block + step-unique RNG", "prob", True, True, 1.0, True, 3, 0.9, False, True),
    ("V7 A+B+C: tau 0.7, block + step-unique RNG", "prob", True, True, 0.7, True, 3, 0.9, False, True),
    ("V8 greedy + block + step-unique RNG", "greedy", False, False, 1.0, True, 3, 0.9, False, True),
    ("V9 A+B token-wise + step-unique RNG keys", "prob", True, True, 1.0, False, 3, 0.9, False, True),
]
if len(sys.argv) > 2 and sys.argv[2] == "diag":
    variants = [
        ("D1 greedy + block, fresh seed per step", "greedy", False, False, 1.0, True, 3, 0.9, True),
        ("D2 A+B+C, fresh seed per step", "prob", True, True, 1.0, True, 3, 0.9, True),
        ("D3 A+B (token-wise), fresh seed per step", "prob", True, True, 1.0, False, 3, 0.9, True),
        ("D4 greedy + block (stock keys, repeat)", "greedy", False, False, 1.0, True),
    ]
allres = [run_variant(*v) for v in variants]
json.dump(allres, open("/w/results/spec_kernel_exactness%s.json" % ("_diag" if len(sys.argv) > 2 else ""), "w"), indent=1)
