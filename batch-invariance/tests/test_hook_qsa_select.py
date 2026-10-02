"""hook3 (rev 2) QSA selection patch through the real import hook: patched qsa_select_paged_tokens vs vLLM's original."""
import os, sys, importlib.util
os.environ["PLEFP8_INV_QSA"] = "1"
sp = importlib.util.spec_from_file_location("plefp8_hook", "/w/hook3/sitecustomize.py")
hk = importlib.util.module_from_spec(sp); sp.loader.exec_module(hk)
import torch
from vllm.models.qwen4_exp.nvidia.ops import qsa as Q
assert hasattr(Q.qsa_select_paged_tokens, "_plefp8_orig"), "not patched"
exec(open("/w/test_qsa_topk.py").read().split("def select(")[0].split("torch.manual_seed(0)")[1].replace('dev = "cuda"', ''), globals()) if False else None
torch.manual_seed(0); dev = "cuda"
H, D, CR, PAGE, TOPK = 4, 128, 4, 400, 2048
src = open("/w/test_qsa_topk.py").read()
exec(src[src.index("def setup("):src.index("def select(")])
def run(fn, args):
    q, kc, pt, t2r, qpos, sl = args
    return fn(q, kc, pt, t2r, qpos, sl, TOPK, CR)
ok = True
for lens in ([2048, 1500, 600], [5000, 5037, 5900, 4700], [12000, 2100, 2049, 64]):
    args = setup(lens)
    new = [run(Q.qsa_select_paged_tokens, args) for _ in range(20)]
    old = [run(Q.qsa_select_paged_tokens._plefp8_orig, args) for _ in range(20)]
    det_new = all(torch.equal(new[0], x) for x in new); det_old = all(torch.equal(old[0], x) for x in old)
    same_sets = torch.equal(new[0].sort(1).values, old[0].sort(1).values)
    small = [r for r, L in enumerate(lens) if L // CR <= TOPK // CR]
    same_small = all(torch.equal(new[0][r], old[0][r]) for r in small)
    ok &= det_new and same_sets and same_small
    print(lens, "patched deterministic:", det_new, "| stock deterministic:", det_old, "| same token sets:", same_sets, "| rows <=2048 identical to stock:", same_small, flush=True)
print("ALL OK" if ok else "FAILED")
# diagnose set differences: are they exact ties at the cut-off?
args = setup([12000, 2100, 2049, 64])
new = run(Q.qsa_select_paged_tokens, args); old = run(Q.qsa_select_paged_tokens._plefp8_orig, args)
q, kc, pt, t2r, qpos, sl = args
logits, vis = Q.qsa_mqa_paged(q, kc, pt, t2r, qpos, sl, CR)
for r in range(4):
    a, b = set(new[r].tolist()) - {-1}, set(old[r].tolist()) - {-1}
    if a != b:
        da = sorted(a - b); db = sorted(b - a)
        blk = lambda toks: sorted({t // CR for t in toks})
        sa = [float(logits[r, x]) for x in blk(da)][:6]; sb = [float(logits[r, x]) for x in blk(db)][:6]
        sel = sorted({t // CR for t in a if t // CR < int(vis[r])}); kth = min(float(logits[r, x]) for x in sel) if len(sel) else None
        print(f"row {r}: visible {int(vis[r])}: only patched {len(da)} tokens (block scores {sa}), only stock {len(db)} (block scores {sb}), k-th score {kth}")
