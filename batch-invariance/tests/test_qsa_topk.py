"""QSA indexer selection determinism (offline, synthetic). qsa_mqa_paged scores -> persistent_topk -> block ids.
(1) run-to-run: same inputs, 30 calls -> identical outputs? (exact order and as sets)  (2) visible <= k vs > k
(3) batch: row 0 alone vs in a batch of 40 rows (tiles_per_program 1 vs 8; different max columns)."""
import torch
from vllm.models.qwen4_exp.nvidia.ops import qsa as Q
torch.manual_seed(0); dev = "cuda"
H, D, CR, PAGE, TOPK = 4, 128, 4, 400, 2048
BT = TOPK // CR
def setup(seq_lens, rows_per_req=1):
    R = len(seq_lens)
    pages_per = [-(-(s // CR) // PAGE) + 1 for s in seq_lens]
    width = max(pages_per)
    k_cache = (torch.randn(sum(pages_per) + 1, PAGE, 1, D, device=dev) / 4).to(torch.bfloat16)
    pt = torch.zeros(R, width, dtype=torch.int32, device=dev); p = 1
    for r, n in enumerate(pages_per):
        pt[r, :n] = torch.arange(p, p + n, device=dev); p += n
    tok2req = torch.arange(R, device=dev, dtype=torch.int32).repeat_interleave(rows_per_req)
    qpos = torch.tensor([s - 1 for s in seq_lens], device=dev, dtype=torch.int32).repeat_interleave(rows_per_req)
    q = torch.randn(len(tok2req), H, D, device=dev).to(torch.bfloat16)
    return q, k_cache, pt, tok2req, qpos, torch.tensor(seq_lens, device=dev, dtype=torch.int32)
def select(q, kc, pt, t2r, qpos, sl):
    logits, vis = Q.qsa_mqa_paged(q, kc, pt, t2r, qpos, sl, CR)
    blocks = torch.empty((q.shape[0], BT), dtype=torch.int32, device=dev)
    ws = torch.empty((1 << 20,), dtype=torch.uint8, device=dev)
    torch.ops._C.persistent_topk(logits, vis, blocks, ws, BT, logits.shape[1])
    return logits, vis, blocks
def valid(blocks, vis):   # only the first min(visible, k) entries are used (expand_qsa_block_indices); the rest is scratch
    n = torch.clamp(vis, max=BT).long()
    col = torch.arange(BT, device=dev)[None, :]
    return torch.where(col < n[:, None], blocks, torch.full_like(blocks, -7))
for L in (2048, 1500, 5000, 12000):
    args = setup([L, L + 37, L + 900, L - 300])
    outs = []
    for _ in range(30):
        _, vis, b = select(*args); outs.append(valid(b, vis).clone())
    exact = all(torch.equal(outs[0], o) for o in outs)
    sets = all(torch.equal(outs[0].sort(1).values, o.sort(1).values) for o in outs)
    nv = min(L // CR, BT)
    asc = bool((outs[0][0, 1:nv] > outs[0][0, :nv - 1]).all())
    print(f"L={L:6d} visible blocks row0={L // CR:5d} (k={BT}): 30 calls identical order: {exact} | identical sets: {sets} | row0 ascending: {asc}", flush=True)
L = 5000
q, kc, pt, t2r, qpos, sl = setup([L] + [L + 50 * i for i in range(1, 40)])
la, va, ba = select(q[:1], kc, pt[:1], t2r[:1], qpos[:1], sl[:1])
lb, vb, bb = select(q, kc, pt, t2r, qpos, sl)
nv = int(va[0])
print("visible", nv, int(vb[0]), "| scores row0 alone vs batch of 40 (visible columns) bit-identical:", torch.equal(la[0, :nv], lb[0, :nv]),
      "| selected sets equal:", torch.equal(valid(ba, va)[0].sort().values, valid(bb, vb)[0].sort().values))
