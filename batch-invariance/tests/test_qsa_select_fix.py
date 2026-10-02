"""Offline check of hook3/inv_qsa_select.fix_selection on synthetic QSA inputs (see test_qsa_topk.py):
determinism over 30 calls, batch invariance, exactness (= stable top-k: score desc, ties lowest index, output ascending),
and rows with <= k visible blocks unchanged vs persistent_topk."""
import sys
sys.path.insert(0, "/w/hook3")
exec(open("/w/test_qsa_topk.py").read().split("for L in (2048")[0])
import inv_qsa_select as F
def fixed(q, kc, pt, t2r, qpos, sl):
    logits, vis, b = select(q, kc, pt, t2r, qpos, sl)
    raw = valid(b, vis).clone()
    F.fix_selection(logits, vis, b, BT, logits.shape[1])
    return logits, vis, valid(b, vis), raw
def reference(logits, vis):   # stable exact top-k by (score desc, index asc), returned ascending
    out = []
    for r in range(logits.shape[0]):
        n = int(vis[r]); x = logits[r, :n].double().cpu()
        if n <= BT:
            out.append(list(range(n)) + [-7] * (BT - n)); continue
        order = sorted(range(n), key=lambda i: (-float(x[i]), i))[:BT]
        out.append(sorted(order))
    return torch.tensor(out, dtype=torch.int32, device=dev)
ok = True
for lens in ([2085], [2048, 2085, 2948, 1748], [5000, 5037, 5900, 4700], [12000, 3000, 2049, 600]):
    args = setup(lens)
    outs = [fixed(*args) for _ in range(30)]
    det = all(torch.equal(outs[0][2], o[2]) for o in outs)
    ref = reference(outs[0][0], outs[0][1])
    exact = torch.equal(outs[0][2], ref)
    small = [r for r in range(len(lens)) if lens[r] // CR <= BT]
    same_small = all(torch.equal(outs[0][2][r], outs[0][3][r]) for r in small)
    ok &= det and exact and same_small
    print(lens, "deterministic:", det, "| exact stable top-k:", exact, "| rows <= k unchanged:", same_small, flush=True)
L = 5000
q, kc, pt, t2r, qpos, sl = setup([L] + [L + 50 * i for i in range(1, 40)])
a = fixed(q[:1], kc, pt[:1], t2r[:1], qpos[:1], sl[:1])[2]
b = fixed(q, kc, pt, t2r, qpos, sl)[2]
inv = torch.equal(a[0], b[0]); ok &= inv
print("row 0 alone vs in a batch of 40: identical selection:", inv)
# ties: force many exact-zero scores via a zero query
args = list(setup([3000, 2100]))
args[0] = torch.zeros_like(args[0]); args[0][0, 0, :4] = 1.0
outs = [fixed(*args) for _ in range(20)]
det = all(torch.equal(outs[0][2], o[2]) for o in outs); exact = torch.equal(outs[0][2], reference(outs[0][0], outs[0][1]))
raw_det = all(torch.equal(outs[0][3].sort(1).values, o[3].sort(1).values) for o in outs)
ok &= det and exact
print("heavy ties: persistent_topk sets deterministic:", raw_det, "| fixed deterministic:", det, "| exact:", exact)
print("ALL OK" if ok else "FAILED")
