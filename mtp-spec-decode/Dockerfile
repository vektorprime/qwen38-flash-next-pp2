# Prod image + local spec-decode patches (all gated by env, default = stock behaviour):
#   VLLM_SPEC_REJECT_UNPROPOSED=1  backport of vllm#58784 (reject scheduler-padded placeholder drafts)
#   VLLM_SPEC_DRAFT_INDEP_NOISE=1  independent Gumbel noise for probabilistic drafting (fixes shared-noise bias)
#   VLLM_SPEC_DRAFT_TOPKP=1        request top-k/top-p applied to the draft distribution
#   VLLM_SPEC_DRAFT_TAU=<t>        draft temperature scale
#   VLLM_SPEC_STEP_UNIQUE_RNG=1    step-unique RNG keys for block verification (local fix; requires INDEP_NOISE)
#   VLLM_SPEC_PAD_NEW_REQS=0       scheduler does not pad new 1-token requests to the spec shape (fixes NaN/garbage
#                                  output for fresh 1-token prompts on this hybrid model; default 1 = stock)
#   + vllm#59355 OOB guard in the block-verification residual-mass kernel (unconditional, inert in our config)
# Files are the live prod files (incl. PP patches 0011/0012 in model_runner.py) with the edits in ../../140 doc §6-7.
FROM qwen38-flash-next:ple-fp8-pp3-detmoe-inv-det
COPY files/speculator.py /opt/vllm/vllm/v1/worker/gpu/spec_decode/speculator.py
COPY files/rejection_sampler.py /opt/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
COPY files/rejection_sampler_utils.py /opt/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
COPY files/model_runner.py /opt/vllm/vllm/v1/worker/gpu/model_runner.py
COPY files/scheduler.py /opt/vllm/vllm/v1/core/sched/scheduler.py
RUN python3 -m py_compile /opt/vllm/vllm/v1/worker/gpu/spec_decode/speculator.py \
      /opt/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py \
      /opt/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py \
      /opt/vllm/vllm/v1/worker/gpu/model_runner.py /opt/vllm/vllm/v1/core/sched/scheduler.py
