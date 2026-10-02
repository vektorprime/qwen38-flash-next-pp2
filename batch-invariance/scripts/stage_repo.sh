#!/bin/bash
# Copy the batch-invariance work into the notes repo (github.com/vektorprime/qwen38-flash-next-pp2, dir batch-invariance/).
# Code, scripts, docs and small result summaries only: no logprob arrays (npz), no corpus text, no secrets.
set -eu
B=/home/user/qwen3nextflash/batchinv
R=/home/user/gh/qwen38-flash-next-pp2/batch-invariance
rm -rf $R; mkdir -p $R/hook $R/scripts $R/tests $R/results
cp $B/hook3/sitecustomize.py $B/hook3/inv_gemm.py $B/hook3/inv_marlin.py $B/hook3/inv_qsa_select.py $B/hook3/tuned_gemm.json $R/hook/
cp $B/deploy/Dockerfile $R/Dockerfile
cp $B/deploy_det/Dockerfile $R/Dockerfile.det
{ echo '> Copied from the working notes (`batchinv/NOTES.md` on the host). Paths like `hook3/`, `results/`, `dt_window*.sh` refer to'; echo '> this directory as `hook/`, `results/`, `scripts/`; tests are in `tests/`.'; echo; cat $B/NOTES.md; } > $R/README.md
cp $B/dt_window3.sh $B/dt_window4.sh $B/dt_window5.sh $B/dt_window6.sh $B/dt_window7.sh $B/dt_window8.sh $B/dt_window9.sh $B/stage_repo.sh $R/scripts/
cp $B/dt3_score.py $B/dt3_decode.py $B/dt3_bench.py $B/dt3_analyze.py $B/dt3_run_cmd.py $B/dt3_cmp_bench.py \
   $B/dt5_run_cmd.py $B/dt5_prefix.py $B/dt5_prefill.py $B/dt5_analyze.py $B/dt5_kld_compare.py \
   $B/tune_cmp.py $B/tune_splitk.py $B/build_tuned_table.py $R/scripts/
cp $B/test_hook3.py $B/test_inv_marlin_int4.py $B/test_sched_align.py $B/test_qsa_topk.py $B/test_qsa_topk2.py $B/test_qsa_select_fix.py \
   $B/test_hook_qsa_select.py $B/test_fla_autotune.py $B/triton_mm_invariance.py $B/test_fused_pad.py $B/test_inv_gemm.py $B/test_inv_gemm2.py \
   $B/census_gemm.py $B/census_gemm_small.py $B/census_gemm_nosplitk.py $B/census_marlin_moe.py $B/census_bi_matmul.py \
   $B/marlin_invariant.py $B/marlin_knobs.py $B/marlin_cost.py $B/marlin_smartpad.py $R/tests/
cp $B/*_sm86.json $R/tests/ 2>/dev/null || true
cd $B/results
cp tune_cmp_A.json tune_cmp_B.json tune_splitk_cmp_a.json tune_splitk_cmp_b.json tune_splitk_3080.json dt3_analysis.json cmp_bench_cmp.json $R/results/ 2>/dev/null || true
cp kld5/compare.json $R/results/kld5_compare_window5.json
cp kld5/pairs.json $R/results/kld5_pairs.json
cp $B/kld_pairs.py $R/scripts/
[ -s kld5/compare_det.json ] && cp kld5/compare_det.json $R/results/kld5_compare_det.json || true
python3 - $R/results <<'EOF'
import glob, json, os, sys
out = sys.argv[1]
src = "/home/user/qwen3nextflash/batchinv/results/"
# decode/bench: keep the summary only (outputs are model text for the fixed bench prompts: fine, but large)
for p in glob.glob(src + "decode_*.json") + glob.glob(src + "bench_*.json"):
    d = json.load(open(p)); json.dump(d.get("res", d), open(os.path.join(out, os.path.basename(p)), "w"), indent=1)
for p in glob.glob(src + "prefill_*.json"):
    json.dump(json.load(open(p)), open(os.path.join(out, os.path.basename(p)), "w"), indent=1)
# prefix test and analyses: drop generated text (continuations of the private corpus)
def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items() if k != "text_run1"}
    if isinstance(o, list):
        return [scrub(v) for v in o]
    return o
for p in glob.glob(src + "prefix_*.json") + glob.glob(src + "dt5_analysis_*.json"):
    json.dump(scrub(json.load(open(p))), open(os.path.join(out, os.path.basename(p)), "w"), indent=1)
EOF
grep -rlE --exclude=stage_repo.sh "hf_[A-Za-z0-9]{20,}|HUGGING_FACE_HUB_TOKEN=[^<]" $R && { echo "SECRET FOUND"; exit 1; } || echo "no secrets"
grep -rl "text_run1" $R/results && { echo "CORPUS TEXT FOUND"; exit 1; } || echo "no corpus text"
du -sh $R; find $R -type f | wc -l
