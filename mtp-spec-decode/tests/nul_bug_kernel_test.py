"""Offline repro of the vLLM #58784 mechanism with this image's rejection kernels.

A scheduler-padded row (new request, 1-token prompt tail, k placeholder drafts)
reaches rejection_sample with draft tokens = 0 (zeroed req_states.draft_tokens;
token 0 is '!' in this tokenizer) and, under probabilistic drafting, the slot's
stale draft_logits. We count how often '!' is emitted at the first output
position against the correct probability p'('!').

Run on an RTX 3080 inside the prod image (no model needed):
  docker run --rm --gpus '"device=2"' --memory 12g --entrypoint python3 \
    -v $PWD:/w qwen38-flash-next:ple-fp8-pp3-detmoe-inv-det /w/nul_bug_kernel_test.py
"""
import json
import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

torch.manual_seed(0)
dev = "cuda"
V = 248320           # Qwen3.8 vocab
K = 3                # num_speculative_tokens (prod)
N = 512              # rows (independent requests) per batch
REPS = 8             # batches per case -> 4096 rows
BANG = 0             # token id 0 == '!'


def target_logits(p_bang, truncate):
    """Raw target distribution: 20 'real' candidates + tiny tail; token 0 gets p_bang."""
    logp = torch.full((V,), -40.0, device=dev)           # tail
    cand = torch.randperm(V - 1, device=dev)[:20] + 1    # never token 0
    w = torch.tensor([0.40, 0.20, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01]
                     + [0.004] * 10, device=dev)
    w = w / w.sum() * (1 - p_bang)
    logp[cand] = w.log()
    logp[BANG] = torch.tensor(p_bang, device=dev).log()
    logp = torch.log_softmax(logp, 0)
    if truncate:  # vLLM order: top-k 20, then top-p 0.95 on the top-k distribution
        top = torch.topk(logp, 20)
        keep = torch.full_like(logp, float("-inf"))
        probs = torch.softmax(top.values, 0)
        csum = torch.cumsum(probs, 0)
        n = int((csum < 0.95).sum().item()) + 1
        keep[top.indices[:n]] = logp[top.indices[:n]]
        logp = torch.log_softmax(keep, 0)
    return logp


FIXED = False


def run_case(p_bang, truncate, draft_mode, block):
    """Returns (rate of '!' as first output token, mean accepted drafts, correct p'('!'))."""
    tl = target_logits(p_bang, truncate)
    p_correct = float(tl[BANG].exp())
    bang_first = 0
    accepted = 0
    for rep in range(REPS):
        num_logits = N * (K + 1)
        logits = tl.unsqueeze(0).expand(num_logits, V).contiguous()
        if draft_mode == "greedy":
            dl = None
        elif draft_mode == "stale_zero":        # slot never drafted: logits 0 -> q uniform
            dl = torch.zeros(N, K, V, dtype=torch.bfloat16, device=dev)
        elif draft_mode == "stale_prev":        # slot's previous occupant's draft logits
            dl = (torch.randn(N, K, V, device=dev) * 3).to(torch.bfloat16)
            dl[:, :, BANG] = -30.0              # '!' very unlikely there
        else:
            raise ValueError(draft_mode)
        # input_ids at logits rows: [last prompt token, d1, d2, d3] per request
        ds = torch.zeros(num_logits, dtype=torch.int64, device=dev)
        if FIXED:  # upstream #58784 behavior: padded draft rows -> -1 before rejection
            ds[:] = -1
        ds[0::K + 1] = 1234  # (kept even when FIXED: only draft slots matter)                      # last prompt token (irrelevant)
        base_pos = 5000
        pos = (torch.arange(K + 1, device=dev).repeat(N) + base_pos).to(torch.int64)
        cu = torch.arange(0, num_logits + 1, K + 1, dtype=torch.int32, device=dev)
        idx = torch.arange(N, dtype=torch.int32, device=dev)
        exp_idx = idx.repeat_interleave(K + 1)
        exp_pos = torch.arange(K + 1, dtype=torch.int32, device=dev).repeat(N)
        temp = torch.ones(N, dtype=torch.float32, device=dev)
        seed = torch.randint(0, 2**62, (N,), dtype=torch.int64, device=dev) + rep
        sampled, num_sampled = rejection_sample(
            logits, dl, ds, cu, pos, idx, exp_idx, exp_pos, temp, seed, K,
            use_block_verification=block)
        sampled = sampled.view(N, -1)
        bang_first += int((sampled[:, 0] == BANG).sum())
        accepted += int((num_sampled - 1).sum())
        del logits, dl
        torch.cuda.empty_cache()
    rows = N * REPS
    return bang_first / rows, accepted / rows, p_correct


cases = [
    # name, p('!') raw, truncate (top-k 20 / top-p 0.95), draft mode, block
    ("upstream-like: no top-k/top-p, p(!)=1e-7", 1e-7, False),
    ("ours: top-k20/top-p.95, '!' outside nucleus (raw 1e-3)", 1e-3, True),
    ("ours: top-k20/top-p.95, '!' inside nucleus (raw 0.10)", 0.10, True),
]
out = []
for name, pb, trunc in cases:
    for mode in ["greedy", "stale_zero", "stale_prev"]:
        for block in [False, True]:
            r, acc, pc = run_case(pb, trunc, mode, block)
            row = dict(case=name, draft=mode, block=block, bang_first=round(r, 4),
                       correct_p_bang=round(pc, 6), mean_accepted=round(acc, 3))
            out.append(row)
            print(json.dumps(row), flush=True)
FIXED = True
for name, pb, trunc in cases:
    for block in [False, True]:
        r, acc, pc = run_case(pb, trunc, "stale_prev", block)
        row = dict(case=name, draft="stale_prev+FIXED(-1 rows)", block=block, bang_first=round(r, 4),
                   correct_p_bang=round(pc, 6), mean_accepted=round(acc, 3))
        out.append(row)
        print(json.dumps(row), flush=True)
json.dump(out, open("/w/nul_bug_kernel_test.json", "w"), indent=1)
