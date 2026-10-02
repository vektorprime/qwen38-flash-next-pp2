#!/bin/bash
# Copy the MTP spec-decode work into the notes repo (github.com/vektorprime/qwen38-flash-next-pp2, dir mtp-spec-decode/).
# Code, scripts, docs and small result summaries only: no logprob/token arrays (npz), no corpus text, no secrets
# (the container inspect dumps and docker-run commands carry the HF token and are never copied).
set -eu
Q=/home/user/qwen3nextflash
M=$Q/mtp
R=/home/user/gh/qwen38-flash-next-pp2/mtp-spec-decode
rm -rf $R; mkdir -p $R/files $R/scripts $R/tests $R/results $R/logs
{ echo '> Copied from the working notes (`140-mtp-acceptance-options.md` on the host). Paths like `mtp/deploy_spec/`,'
  echo '> `mtp/results/`, `mtp/logs/`, `mtp/*.py|sh` refer to this directory as `./` (Dockerfile, `files/`, `spec.diff`),'
  echo '> `results/`, `logs/`, `scripts/`; offline kernel tests are in `tests/`.'
  echo; cat $Q/140-mtp-acceptance-options.md; } > $R/README.md
cp $M/deploy_spec/Dockerfile $R/Dockerfile
cp $M/deploy_spec/files/*.py $R/files/
# readable diff of the five patched vLLM files against the live -inv-det image copies
: > $R/spec.diff
for f in speculator rejection_sampler rejection_sampler_utils model_runner scheduler; do
  diff -u --label a/$f.py --label b/$f.py $M/deploy_spec/orig/$f.py $M/deploy_spec/files/$f.py >> $R/spec.diff || true
done
cp $M/gen_suite.py $M/stage_check.py $M/recalib.py $M/wait_log.py $M/wait_t1.py $M/pad_probe.py $M/ceiling_probe.py \
   $M/baseline_prod.sh $M/dt_window_spec.sh $M/deploy_spec.sh $M/stage_repo.sh $R/scripts/
cp $M/spec_kernel_exactness.py $M/nul_bug_kernel_test.py $M/gumbel_coupling_mc.py $R/tests/
cp $M/results/stages.json $M/results/spec_kernel_exactness.json $M/results/spec_kernel_exactness_diag.json \
   $M/results/padprobe_prod_stock.json $M/nul_bug_kernel_test.json $M/ceiling_probe.json $R/results/
cp $M/speclog.txt $R/results/speclog_prod_metrics.txt
for t in PROD_base S0_stock S1_A S2_AB_t10 S2_AB_t08 S3_ABC PROD_new PROD_new2; do
  cp $M/results/gen_$t.json $R/results/
done
cp $M/logs/dt_window_spec.log $M/logs/dt_window_spec_attempt1.log $M/logs/deploy_spec.log $M/logs/deploy_spec_attempt1.log \
   $M/logs/exactness_full.log $M/logs/baseline_prod.log $M/logs/baseline_prod_t1_v2.log $M/logs/baseline_prod_t1_v3.log \
   $M/logs/baseline_prod_cal.log $M/ceiling_probe.log $R/logs/
grep -rlE --exclude=stage_repo.sh "hf_[A-Za-z0-9]{20,}|HUGGING_FACE_HUB_TOKEN=[^<]" $R && { echo "SECRET FOUND"; exit 1; } || echo "no secrets"
find $R -name "*.npz" | grep . && { echo "NPZ FOUND"; exit 1; } || echo "no npz"
grep -rl "text_run1\|\"token_ids\"" $R/results && { echo "CORPUS TEXT FOUND"; exit 1; } || echo "no corpus text"
du -sh $R; find $R -type f | wc -l
