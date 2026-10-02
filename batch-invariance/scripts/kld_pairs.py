"""compare_ab pair() metrics for arbitrary pairs of results/kld5/*.npz. usage: kld_pairs.py REF:TEST [REF:TEST ...] -> prints + results/kld5/pairs.json"""
import importlib.util, json, os, sys
import numpy as np
D = "/home/user/qwen3nextflash/batchinv/results/kld5/"
spec = importlib.util.spec_from_file_location("cab", "/home/user/qwen3nextflash/plefp8/e2e/compare_ab.py")
cab = importlib.util.module_from_spec(spec); spec.loader.exec_module(cab)
L = lambda n: {k: v for k, v in np.load(D + n + ".npz").items()}
out = json.load(open(D + "pairs.json")) if os.path.exists(D + "pairs.json") else {}
for pr in sys.argv[1:]:
    a, b = pr.split(":")
    p = cab.pair(L(a), L(b))[0]
    out[pr] = p
    print(pr, json.dumps({k: (round(p[k], 5) if isinstance(p[k], float) else p[k]) for k in ("kld_mean", "kld_median", "kld_p99", "top1_same", "sampling_set_identical", "sampling_same_token_prob", "dnll_mean", "bit_identical_logprobs")}), flush=True)
json.dump(out, open(D + "pairs.json", "w"), indent=1)
