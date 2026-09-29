# GPU unit test for the deterministic moe_align patch. Runs inside the patched image.
import time

import torch

import vllm.model_executor.layers.fused_moe.moe_align_block_size as m

torch.manual_seed(0)
dev = "cuda"


def skewed_topk(M, topk, E):
    # realistic skew: a few hot experts picked by most tokens
    bias = torch.linspace(4, 0, E, device=dev)[torch.randperm(E, device=dev)]
    logits = torch.randn(M, E, device=dev) + bias
    return torch.topk(logits, topk, dim=-1).indices.to(torch.int32)


def align(ids, flag, block=8, E=512):
    m._DETERMINISTIC_ALIGN = flag
    return m.moe_align_block_size(ids, block, E, ignore_invalid_experts=True)


def check(M, topk=10, E=512, block=8, reps=30):
    ids = skewed_topk(M, topk, E)
    raw = [align(ids, False) for _ in range(reps)]
    fix = [align(ids, True) for _ in range(reps)]
    n = int(raw[0][2].item())
    raw_distinct = len({r[0][:n].cpu().numpy().tobytes() for r in raw})
    fix_distinct = len({f[0][:n].cpu().numpy().tobytes() for f in fix})
    ok = True
    for r, f in zip(raw[:5], fix[:5]):
        ok &= torch.equal(r[1][: n // block], f[1][: n // block]) and torch.equal(r[2], f[2])
        e = r[1][: n // block].repeat_interleave(block)
        for ex in torch.unique(e).tolist():
            sel = e == ex
            # same multiset of slots per expert segment
            ok &= torch.equal(torch.sort(r[0][:n][sel]).values, torch.sort(f[0][:n][sel]).values)
            # ascending valid slots, padding (== numel) at the segment end
            seg = f[0][:n][sel]
            val = seg[seg < ids.numel()]
            ok &= bool((val[1:] > val[:-1]).all()) and bool((seg[len(val):] == ids.numel()).all())
    maxload = int(torch.bincount(ids.flatten().long(), minlength=E).max())
    print(f"M={M:5d} max_tokens_per_expert={maxload:4d} "
          f"raw_distinct_orders={raw_distinct:2d}/{reps} patched_distinct={fix_distinct}/{reps} valid={ok}")
    return fix_distinct == 1 and ok


res = [check(M) for M in (1, 4, 16, 17, 24, 64, 256, 2048, 8192)]

# CUDA graph capture + replay with the patch on
ids = skewed_topk(2048, 10, 512)
m._DETERMINISTIC_ALIGN = True
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        m.moe_align_block_size(ids, 8, 512, ignore_invalid_experts=True)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = m.moe_align_block_size(ids, 8, 512, ignore_invalid_experts=True)
g.replay()
torch.cuda.synchronize()
a = out[0].clone()
g.replay()
torch.cuda.synchronize()
eager = m.moe_align_block_size(ids, 8, 512, ignore_invalid_experts=True)
n = int(eager[2].item())
graph_ok = torch.equal(a[:n], out[0][:n]) and torch.equal(a[:n], eager[0][:n])
print("cuda-graph capture/replay ok and equal to eager:", graph_ok)

for M in (32, 2048):
    ids = skewed_topk(M, 10, 512)
    for flag in (False, True):
        m._DETERMINISTIC_ALIGN = flag
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(200):
            m.moe_align_block_size(ids, 8, 512, ignore_invalid_experts=True)
        torch.cuda.synchronize()
        print(f"M={M} patched={flag} {1e6 * (time.perf_counter() - t) / 200:.1f} us/call")
print("ALL PASS" if all(res) and graph_ok else "FAIL")

# Triton fast path vs torch fallback on the same raw alignment
xok = True
for M in (3, 17, 64, 2048, 8192):
    ids = skewed_topk(M, 10, 512)
    m._DETERMINISTIC_ALIGN = False
    s, e, n = m.moe_align_block_size(ids, 8, 512, ignore_invalid_experts=True)
    a = m._canonicalize_sorted_ids_torch(s.clone(), e, n, 8)
    b = m._canonicalize_sorted_ids(ids, s.clone(), e, n, 8, 512, True)
    nv = int(n.item())
    xok &= torch.equal(a[:nv], b[:nv])
print("triton path == torch fallback:", xok)
print("FINAL", "PASS" if all(res) and graph_ok and xok else "FAIL")
