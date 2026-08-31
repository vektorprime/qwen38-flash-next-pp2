import pathlib

p = pathlib.Path("/opt/vllm/vllm/v1/worker/gpu_model_runner.py")
src = p.read_text()
assert "FIX 53919" not in src, "already patched"

# --- Part 1: wait on the accepted-counts event BEFORE the first row move in _update_states.
anchor1 = "    def _update_states(self, scheduler_output: \"SchedulerOutput\") -> Callable | None:\n"
assert src.count(anchor1) == 1, "anchor1 not unique"
ins1 = anchor1 + """        # FIX 53919: the accepted-token counts copied D2H (non-blocking) by
        # step N's mamba postprocess land in a pinned buffer that this step's
        # row moves (add_request/swap_states/condense) permute. Wait BEFORE
        # the first row move so the DMA has landed; syncing on an unrecorded
        # event is a no-op, so non-hybrid models are unaffected.
        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
"""
src = src.replace(anchor1, ins1, 1)

# --- Part 2: drop the async double-permute gather in _prepare_inputs.
old2 = """            # Async mode: condense() reordered indices, use prev_positions mapping
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # Non-async mode: use values directly
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
"""
new2 = """            # FIX 53919 (part 2): the prev_positions double-permute gather is
            # removed. The event wait at the top of _update_states guarantees
            # the copy landed before condense()/swap_states ran, so the buffer
            # is already in CURRENT row order and the direct read applies in
            # every mode. The old gather re-permuted already-correct values
            # whenever the DMA landed early, handing one request another
            # request's accepted count (-> wrong GDN snapshot selector ->
            # cross-request state bleed via the SSM cache).
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
"""
assert src.count(old2) == 1, "anchor2 not unique"
src = src.replace(old2, new2, 1)

p.write_text(src)

import ast
ast.parse(src)
print("patched: event wait before row moves + gather removed - syntax OK")
