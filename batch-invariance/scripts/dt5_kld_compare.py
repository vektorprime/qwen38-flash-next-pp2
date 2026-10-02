"""KLD + same-top-p comparison on the batch-invariant stack (window 5). Metrics = plefp8/e2e/compare_ab.py pair()
(copied from kld/diag/audit/extended.py). Reads results/kld5/{REF_a,REF_b,Q4,FP8}.npz (kld/score.py collect, 77 windows, K=200).
  REF_a, REF_b : INT8 experts + BF16 PLE (prod checkpoint), two separate boots -> noise floor (expected: bit-identical)
  Q4           : INT4 experts (AWQ g32, original hub checkpoint) + BF16 PLE      -> INT4 vs INT8
  FP8          : INT8 experts + FP8 PLE table                                     -> FP8 PLE vs BF16 PLE
Writes results/kld5/compare.json."""
import importlib.util, json, os, sys
import numpy as np
D = "/home/user/qwen3nextflash/batchinv/results/kld5/"
spec = importlib.util.spec_from_file_location("cab", "/home/user/qwen3nextflash/plefp8/e2e/compare_ab.py")
cab = importlib.util.module_from_spec(spec); spec.loader.exec_module(cab)
L = lambda n: {k: v for k, v in np.load(D + n + ".npz").items()} if os.path.exists(D + n + ".npz") else None
ref, ref2, q4, fp8 = L("REF_a"), L("REF_b"), L("Q4"), L("FP8")
res = {"setup": "PP3 CMP 170HX, prod serving config + hook3 batch-invariance fixes (MoE/GEMM/QSA/chunk alignment), 77x2048 corpus, K=200",
       "sampling": dict(temperature=1.0, top_k=cab.TOPK, top_p=cab.TOPP)}
KEYS = ("kld_mean", "kld_mean_ci95", "kld_median", "kld_p99", "top1_same", "top5_overlap", "sampling_set_identical",
        "sampling_same_token_prob", "ppl_ref", "ppl_test", "dnll_mean", "dnll_ci95", "bit_identical_logprobs")
for name, a, b in (("floor_REF_a_vs_REF_b", ref, ref2), ("INT4_vs_INT8 (ref INT8, both BF16 PLE)", ref, q4),
                   ("FP8PLE_vs_BF16PLE (ref BF16 PLE, both INT8)", ref, fp8)):
    if a is None or b is None:
        continue
    p = cab.pair(a, b)[0]
    res[name] = p
    print(name); print("   " + json.dumps({k: p[k] for k in KEYS}))
try:
    c = json.load(open("/home/user/qwen3nextflash/kld/results/v2/extended.json"))["pairs"]
    res["context_v2_stock_stack"] = {k: {x: c[k][x] for x in ("kld_mean", "top1_same", "sampling_set_identical", "sampling_same_token_prob") if x in c[k]} for k in c}
except Exception as e:  # noqa
    res["context_v2_stock_stack"] = repr(e)
json.dump(res, open(D + "compare.json", "w"), indent=1)
