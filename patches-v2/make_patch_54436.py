import pathlib

p = pathlib.Path("/opt/vllm/vllm/v1/worker/gpu/pp_utils.py")
src = p.read_text()
assert "FIX 54436" not in src, "already patched"

old = """    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    max_seq_len = input_batch.max_seq_len_np
    assert max_seq_len is not None  # always populated under PP
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # Exclude requests that we know are finished.
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None
"""
new = """    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # FIX 54436: drop the ``not_finishing`` exclusion. Whether a request stops
    # is only known AFTER sampling; under spec decode num_computed_tokens can
    # overrun prompt_len + max_tokens while the scheduler still runs the
    # request, so the old predicate dropped still-decoding rows from the
    # sampled-token broadcast. The earlier stages' last_sampled_tokens and
    # draft_tokens then freeze while the last rank advances -- degenerate,
    # repeating output, and KV written at positions the last stage never used.
    # Requests that really finish are already handled by the req_idx_gen check
    # in get_prev_sampled_outputs, so this guard bought nothing.
    # The non-final-chunked-prefill exclusion is the only sound one: a
    # non-final chunk provably produces no sample, and both ranks advance
    # num_computed_tokens identically without a broadcast.
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    need_sampled_mask = produces_sample
    return need_sampled_mask if need_sampled_mask.any() else None
"""
assert src.count(old) == 1, "anchor not unique"
src = src.replace(old, new, 1)
p.write_text(src)

import ast
ast.parse(src)
print("patched compute_need_sampled_mask: 54436 - syntax OK")
