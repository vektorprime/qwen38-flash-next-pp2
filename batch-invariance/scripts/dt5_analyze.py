"""Window-5 analysis: prompt-length invariance on ALL positions (chunk-boundary residual), prefix/chunk repro, decode repro,
bench, and the KLD comparisons. usage: dt5_analyze.py <tag> [<tag2> ...]   (tags of s_<tag>_L<L>.npz sets, e.g. E3)
Writes results/dt5_analysis_<tag>.json."""
import glob, json, os, re, sys
import numpy as np
R = "/home/user/qwen3nextflash/batchinv/results/"
chunk1 = lambda L: (min(L, 2048) // 8 * 8 - 8) // 64 * 64   # first prefill chunk with PLEFP8_ALIGN_CHUNKS=64 (lone prompt)
out = {}
for tag in sys.argv[1:]:
    S = {}
    for f in glob.glob(f"{R}s_{tag}_L*.npz"):
        m = re.search(r"_L(\d+)\.npz$", f)
        if m: S[int(m.group(1))] = dict(np.load(f))
    Ls = sorted(S, reverse=True)
    pairs, n_ok = {}, 0
    for i, a in enumerate(Ls):
        for b in Ls[i + 1:]:
            hi = min(a, b)
            X, Y = S[a], S[b]
            same_ids = np.array_equal(X["ids"][:, 1:hi], Y["ids"][:, 1:hi])
            same_lp = np.array_equal(X["logprobs"][:, 1:hi], Y["logprobs"][:, 1:hi])
            diffpos = np.nonzero((X["logprobs"][:, 1:hi] != Y["logprobs"][:, 1:hi]).any(-1).any(0))[0] + 1
            ok = same_ids and same_lp
            n_ok += ok
            pairs[f"{a} vs {b}"] = {"chunk_boundaries": [chunk1(a), chunk1(b)], "positions": hi - 1, "bit_identical_all_positions": bool(ok),
                                    "n_positions_differing": int(diffpos.size), "first_differing_position": int(diffpos[0]) if diffpos.size else None,
                                    "top1_flips": int((X["ids"][:, 1:hi, 0] != Y["ids"][:, 1:hi, 0]).sum())}
    out[tag] = {"lengths": Ls, "pairs_bit_identical": f"{n_ok}/{len(pairs)}", "pairs": pairs}
    print(f"{tag}: prompt lengths {Ls}: {n_ok}/{len(pairs)} pairs bit-identical on all common positions")
    for k, v in pairs.items():
        if not v["bit_identical_all_positions"]:
            print(f"   {k}: chunk1 {v['chunk_boundaries']} {v['n_positions_differing']} positions differ from {v['first_differing_position']}, flips {v['top1_flips']}")
    for kind in ("prefix", "decode"):
        p = f"{R}{kind}_{tag}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            out[tag][kind] = d if kind == "prefix" else d.get("res", d)
            if kind == "prefix":
                print(f"{tag}: prefix/chunk repro all identical: {d['all_identical']}; cached tokens {[c['cached_tokens'] for c in d['cases']]}")
            else:
                print(f"{tag}: decode repro: {json.dumps(out[tag][kind])[:300]}")
    for p in sorted(glob.glob(f"{R}bench_{tag}*.json")):
        b = json.load(open(p))["res"]; out[tag].setdefault("bench", []).append(b)
        print(f"{tag}: bench {b['tag']}: {b['tok_per_s']:.1f} tok/s, acceptance {b['acceptance']:.3f}")
json.dump(out, open(f"{R}dt5_analysis_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
