"""Offline simulation of prefill chunking for a lone prompt with vLLM's real Scheduler methods (stock, hook3_rev1, hook3 = rev 2).
Fake scheduler state mirrors prod: scheduler block size 8 (hash granularity), MTP (use_eagle back-off), budget 2048,
num_prefill_lookahead 3. Prints chunk ends per prompt length; checks rev-2 ends are 64-aligned except the prompt end."""
import os, sys, types, importlib, importlib.util
which = sys.argv[1]
if which != "stock":
    os.environ["PLEFP8_ALIGN_CHUNKS"] = "64"
    sp = importlib.util.spec_from_file_location("plefp8_hook", f"/w/{which}/sitecustomize.py")
    hk = importlib.util.module_from_spec(sp); sp.loader.exec_module(hk)   # installs the import hook (image has its own sitecustomize)
import vllm.v1.core.sched.scheduler as m
S = m.Scheduler
def fake(bs=8, hb=8, partial=False):   # prod: hash granularity == scheduler block size -> no partial-tail stop
    return types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=bs), use_eagle=True, max_num_scheduled_tokens=2048,
                                 scheduler_config=types.SimpleNamespace(long_prefill_token_threshold=0), hash_block_size=hb,
                                 mamba_partial_cache_hit=partial, num_prefill_lookahead=3)
def chunks(L, budget=2048, bs=8):
    sf = fake(bs); r = types.SimpleNamespace(num_computed_tokens=0, num_tokens=L, num_prompt_tokens=L, shared_prefix_boundary=0)
    ends = []
    while r.num_computed_tokens < L:
        n = min(L - r.num_computed_tokens, budget)
        n = S._mamba_block_aligned_split(sf, r, n)
        n = S._reserve_prefill_lookahead(sf, r, r.num_computed_tokens, n)
        assert n > 0, (L, ends)
        r.num_computed_tokens += n; ends.append(r.num_computed_tokens)
    return ends
bad = 0
for L in (2048, 2047, 2040, 2001, 1985, 1984, 1983, 1600, 1100, 70, 65, 64, 9, 5000, 4097, 6000):
    e = chunks(L)
    ok = all(x % 64 == 0 for x in e[:-1])
    bad += not ok
    print(f"{which:5s} L={L:5d} chunk ends {e} {'' if ok else '<-- non-aligned'}")
for b in (100, 37, 1000):   # load: smaller budgets
    for L in (2048, 2001):
        e = chunks(L, b); ok = all(x % 64 == 0 for x in e[:-1]); bad += not ok
        print(f"{which:5s} budget {b} L={L} ends {e[:6]}{'...' if len(e) > 6 else ''} {'' if ok else '<-- non-aligned'}")
print(which, "non-aligned cases:", bad)
