# 95 — Patch 0012: parity-ring align correction + exact-mirror mode

**SHIPPED LIVE (2026-09-01 00:41 UTC, final image v7 commit `124df21c4393`):**
container `qwen38-flash-next` running from it, boot clean (`Application
startup complete`), smoke completion OK, TTFT 856 ms. Boot config unchanged
(PP2, MTP k=2, seqs=1, sync, PLE offload).
**VALIDATED (2026-09-01 01:38-01:41 UTC):** recreation at the previously
bleeding config (seqs=8 + `--async-scheduling` + `VLLM_PP_EXACT_MIRROR=1`,
MTP k=2, PLE offload): light soak (4x3) x3 rounds + heavy soak (8x6) x2
rounds = **130 requests, bleed_turns=0** (unpatched bisection: 1-2/run
async, 1-23/run seqs=8 sync-soak). Control seqs=1 pre/post patch: 0.
During integration four 0012 defects surfaced and were fixed in-image
(F1c fixes, all included in the canonical diff below):
- F1c-i: model_runner was missing `ring_zero_` / `pp_peek_rej_correction`
  imports → NameError at warmup.
- F1c-ii: `postprocess_num_computed_tokens` still passed removed
  `_pp_advance_gpu`; now passes `self._pp_advance_ring[self._pp_step % 2]`.
- F1c-iii: V2 `run_fused_postprocess_align` call passed `num_reqs`
  positionally into `advance_ptr`'s slot AND `advance_ptr=` as kwarg →
  TypeError. Fixed: `advance_gpu` positional, no kwarg. (First fix attempt
  hit the wrong call site — run_fused_precopy — caught as NameError;
  corrected with unique anchors.)
- F1c-iv: exact-mirror pop path referenced placeholder `_ZERO_INT32` →
  NameError, exact-mirror boot only. Fixed:
  `torch.zeros_like(outputs["num_rejected"])`.
CANONICAL DIFF for future rebuilds:
`patches/0011-0012-combined-pp2-p1.diff` (pp2-p0 → v7 live tree, all 6
files, `git apply --check` DRYRUN-OK on fresh pp2-p0). The stacked
0011+0012 pair below remains the design record.
LESSON: AST/py_compile/import checks miss call-shape and env-path runtime
errors; all four defects were caught only by boot warmup. The exhaustive
undefined-name AST scan (lambdas false-positive) is in this session's
notes; run it plus a warmup boot before declaring a patched image good.

Stacked on 0011 (`patches-align/0012-pp-parity-ring-fix.diff`, applies AFTER
`0011-f1-pp-align-skew-fix.diff` (in patches-align/); chain validated `git apply` +
AST on a fresh copy of `src/vllm`). Env-gated; default OFF.

## Why 0012 exists (0011 flaw found during implementation)
0011's single-slot `prev_qlen` ring is read at decision em(S) expecting
q(S-1), but it is written at st(T) and NOT cleared per step. When a
request is skipped at S-1 (PP cadence gate
`next_decode_eligible_step` scheduler.py:546 — active whenever
AsyncScheduler is in use, budget defer, or any non-scheduling step), the
slot still holds q(S-2) → WRONG subtraction (e.g. -3 when skew is 0) →
align decision shifted the other way. Simulation: decision-mode residual
errors on every skipped step.

## 0012a — Parity ring (decision mode; env unchanged, active whenever 0012 applied)
- `ring[2][max_num_reqs] int32` replaces the single slot.
- em(S) head: `ring_zero_(ring[S%2])` (real steps only, main stream).
- st(T): optimistic `+q(T)` also scatters q(T) into ring[T%2]
  (0011's `advance_out` arg, retargeted to the parity slice).
- Decision em(S) (serves step S-2): subtract `ring[(S-1)%2][req]`.
  - scheduled at S-1: skew = q(S-1) = ring value. skipped: skew 0 = 0.
  - freed rows: filtered (-1) at pop and peek; stale entries die with the
    ≤2-step parity rotation before any live row can read them.
- Simulation (sync / cadence-2 / random-defer / variable-qlen, k=2):
  decision residual 0 in ALL modes.

## 0012b — Exact-mirror peek mode (VLLM_PP_EXACT_MIRROR=1, async only)
At em(S) head (before the pop), PEEK queue[-1] = P(S-1) (received at
st(S-1), consumed at em(S+1) — one step before the pop needs it):
`wait_event(recv)`, then kernel: `mirror[req] -= rej(S-1)` and
`ring[(S-1)%2][req] -= rej(S-1]`. The pop later passes zeroed
num_rejected (no double subtract). peek is read-only; no protocol change.
- GPU mirror at forward(S) becomes EXACTLY truth(S-1) = last rank →
  fixes Consequence B (KV position displacement, the looping signature)
  and Consequence C (rank-0 preprocess early boundary crossing) too.
- Decision subtracts net(S-1) (ring) from truth(S-1) mirror → truth(S-2).
- Gated on async_scheduling: with AsyncScheduler+V2 the worker always
  receives when the request decodes; sync-mode P(S-1) may legitimately
  be None (mask skip) → decision mode only there (safe default).
- Simulation: decision residual 0 AND positions-mirror residual 0 in all
  scheduling patterns (vs 66% positions-error residual in decision-only).

## Files touched (0012 diff, 226 lines)
- `pp_utils.py`: `peek_pending_correction()` — filtered idx (-1 for
  freed/unneeded), num_rejected GPU tensor, recv event; queue[-1] peek.
- `input_batch.py`: `ring_zero_`, `_pp_peek_rej_correction_kernel` /
  `pp_peek_rej_correction`.
- `model_runner.py`: ring[2] init + step counter + zero-fill/peek at
  em head, `_zero_num_rejected` stand-in, advance_out → parity slice,
  postprocess_state advance_gpu → `ring[(S-1)%2]`.

## Deployment
`git apply 0011.diff && git apply 0012-stacked.diff`; run decision-mode
first (no env), soak seqs=8; then VLLM_PP_EXACT_MIRROR=1 + async soak.
0011+0012 decision mode supersedes bare 0011 — do NOT ship 0011 alone
(stale-ring flaw above).
