# V2-runner-era bugfix patches (2026-08-31)

Applied to qwen38-flash-next:pp2-p0 via docker cp + docker commit (image c234557b+).

## patch_53919 — [Bugfix] Await the accepted-token copy before moving batch rows
Upstream: https://github.com/vllm-project/vllm/pull/53919
Targets V1 runner (vllm/v1/worker/gpu_model_runner.py):
1. _update_states waits on num_accepted_tokens_event BEFORE the first row
   move (add_request/condense permute the pinned accepted-counts buffer).
2. Removes the async double-permute gather in _prepare_inputs.
NOTE: this fork runs the V2 runner (see boot log "Using V2 Model Runner"),
so this patch is currently INERT but kept for any V1-path use.
Applying script re-derives anchors; safe on the a5530b9 base.

## patch_54436 — [Bugfix][PP] Never drop a decoding request from the sampled-token broadcast
Upstream: https://github.com/vllm-project/vllm/pull/54436
Targets V2 pp_utils.compute_need_sampled_mask: drops the not_finishing
guard (num_computed can overrun prompt+max_tokens under spec decode while
the request still runs; dropping the row froze the earlier stage's
last_sampled_tokens/draft_tokens -> degenerate repeating output).
req_idx_gen in get_prev_sampled_outputs already covers truly-finished reqs.

## Apply
python3 make_patch_53919.py && python3 make_patch_54436.py inside the
container (paths are absolute /opt/vllm/...), then docker commit.
