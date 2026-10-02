import sys, time, torch, json
sys.argv = [sys.argv[0], "/tmp/x.json"]
src = open("/w/tune_cmp.py").read().split("A = {\"stock\"")[0]   # setup + helpers only
exec(src)
r = {"stock": {"inv": invariance(None), "us": timing(None)}}
for name, (s, l) in {"k64 s128 l256": ({"thread_k": 64, "thread_n": 128, "blocks_per_sm": 1}, {"thread_k": 64, "thread_n": 256, "blocks_per_sm": 1}),
                     "k64 s128 l128": ({"thread_k": 64, "thread_n": 128, "blocks_per_sm": 1}, {"thread_k": 64, "thread_n": 128, "blocks_per_sm": 1})}.items():
    impl = make(s, l); r[name] = {"inv": invariance(impl), "us": timing(impl)}
for k, v in r.items(): print(k, v, flush=True)
