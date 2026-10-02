import sys
sys.argv = ["x"]
exec(open("/w/test_qsa_topk.py").read().split("for L in (2048")[0])
for lens in ([2048, 2085, 2948, 1748], [2048], [2052], [2085], [2048, 2048]):
    args = setup(lens)
    outs = []
    for _ in range(30):
        _, vis, b = select(*args); outs.append(valid(b, vis).clone())
    per_row = [all(torch.equal(outs[0][r].sort().values, o[r].sort().values) for o in outs) for r in range(len(lens))]
    per_row_order = [all(torch.equal(outs[0][r], o[r]) for o in outs) for r in range(len(lens))]
    print(lens, "visible", [int(x) for x in vis], "sets identical per row:", per_row, "order identical per row:", per_row_order)
    if not all(per_row):
        r = per_row.index(False); a = set(outs[0][r].tolist())
        for o in outs[1:]:
            b = set(o[r].tolist())
            if a != b:
                print("   row", r, "only in call0:", sorted(a - b)[:8], "only in other:", sorted(b - a)[:8]); break
