> Copied from the working notes (`140-mtp-acceptance-options.md` on the host). Paths like `mtp/deploy_spec/`,
> `mtp/results/`, `mtp/logs/`, `mtp/*.py|sh` refer to this directory as `./` (Dockerfile, `files/`, `spec.diff`),
> `results/`, `logs/`, `scripts/`; offline kernel tests are in `tests/`.

# 140 — MTP acceptance: measurements and options

2026-10-02. Research only. Prod (`qwen38-flash-next-pp3-int8all`) was not restarted or reconfigured. The only load on it was ~8 minutes of ordinary chat requests for the probe in §1.
Data and scripts: `mtp/` (`ceiling_probe.py/.json/.log`, `speclog.txt`, `gumbel_coupling_mc.py`).

## 1. Where we are (measured)

| Source | Per-token acceptance | Mean acceptance length (k=3) | Per-position (unconditional) |
|---|---|---|---|
| Prod `/metrics`, 4 h real traffic (T=1, top_p 0.95, top_k 20, thinking xhigh) | **60.0 %** | **2.80** | 0.751 / 0.588 / 0.461 |
| `int8top5/bench.py` (T=0, thinking off) on the same stack | 72–74 % | ≈3.2 | not recorded |

- Conditional acceptance is flat: 0.78 at position 2 given position 1, and 0.78 at position 3 given position 2. Drafts do not get worse with depth.
- **`bench.py` is greedy (T=0), so it cannot see any change to draft sampling.** At T=0 every draft method reduces to argmax. Every option below needs a T=1 benchmark instead.

**Greedy-drafting ceiling probe** (`mtp/ceiling_probe.py`): 8 prompts × 1500 tokens at prod sampling, top-20 logprobs, with top-k 20 → top-p 0.95 applied as vLLM does. For every token, `max p'` is the acceptance a *perfect* greedy drafter would get (draft = target argmax).

| | Position-1 acceptance | Mean length k=3 | Mean length k=4 |
|---|---|---|---|
| Perfect greedy drafter, all 11.5k probe tokens | 0.750 (0.60–0.94 per prompt) | 2.83 | 3.25 |
| Clean request (#3, no other traffic in window): perfect greedy | 0.793 | 2.99 | 3.45 |
| Clean request #3: **actual MTP** | **0.711** | **2.63** | — |
| Any drafter with q = p' and sampled drafting (theoretical max) | 1.0 | 4.0 | 5.0 |

On this text, wrong drafter argmaxes cost ~8 points at position 1. Greedy drafting at T=1 costs ~21 points, even with a perfect drafter. Probe text: 27 % of tokens have max p' < 0.5, and the mean nucleus is 5 tokens. **Most of the headroom is in how drafts are sampled, not in drafter quality.** Caveat: only one probe request ran with no other traffic. The other seven overlapped live traffic, so their *actual* numbers are mixed and were not used. The ceilings come from target logprobs and are valid for all eight.

Checked and ruled out:
- QSA index sharing across draft steps is already off: `index_share_for_mtp_iteration` is absent from config, so it defaults to False.
- Block verification with greedy drafts gives zero gain. With a fixed draft, P(accept ≥ i) ≤ ∏ p(x_j), and token-wise verification already reaches that bound. The paper's h_i formula telescopes to the same value; the research brief confirms this.
- MTP weights are BF16 already.
- Target fidelity is already harvested: INT4 71.4 % vs INT8 74.2 % on the greedy bench.

## 2. Correctness finding: probabilistic drafting in this image is lossy

The image's vLLM (2026-08-29 nightly) samples drafts with Gumbel noise keyed by `(seed, positions + 1)`. That is the same key the rejection sampler uses to resample from the residual after a rejection (`speculator.py` sample_draft; `rejection_sampler_utils.py` `_resample_kernel` → `gumbel_block_argmax`). The comment says this matching is intentional. After a rejection, the residual sample is therefore conditioned on the noise that produced the rejected draft.

Monte Carlo (`mtp/gumbel_coupling_mc.py`, 4M samples per case):

| p (target) / q (draft) | TV(output, p), shared noise | TV, fresh noise |
|---|---|---|
| (.1,.45,.45) / (.8,.19,.01) | 0.062 | 0.0002 |
| (.2,.4,.4) / (.6,.3,.1) | 0.045 | 0.0004 |
| (.05,.5,.3,.15) / (.7,.25,.04,.01) | 0.065 | 0.0003 |
| drafter close to target, or residual on 1 token | ≤ 0.0006 (noise level) | ≤ 0.0006 |

The bias appears when the residual covers ≥ 2 tokens with very different q. The research brief reports upstream measured output TV 0.0125 (vs 0.0014 floor) and fixed it in #47386 / #54282; those PR claims are unverified here. **Current prod (greedy drafts) is unaffected**: a greedy draft uses no noise, so the residual sample is independent of it. **Do not set `draft_sample_method: probabilistic` on this image without a fix.**

## 3. Options

Literature numbers come from the research brief. The lit-review PR and issue numbers were not re-verified.

### A. Sampled drafting with independent noise (small patch + config) — recommended first
- **Change:** `"draft_sample_method":"probabilistic"`, plus a patch so draft Gumbel noise uses a key independent of the target's (e.g. a derived seed in `sample_draft`).
- **Evidence:**
  - vLLM #56724: Qwen3.5-9B MTP, k=3, T=1, top_p .95, top_k 20 (our exact sampling). Mean length 2.39 → 2.65 (+11 %).
  - SGLang, Qwen3.8-27B MTP, T=1: 2.99 → 3.18 (+6 %).
  - llama.cpp, Qwen3.5-9B: +4 %.
  - Li et al. 2026: sampled drafting won in 23/24 Qwen3.5–3.7 model/task pairs.
  - For us, ≈ 2.80 → 2.95–3.15. Single-stream decode time per step barely changes, so tok/s should track this.
- **Risk:** it can lose where the target is peaked and the drafter is over-dispersed; only a measurement settles that. It is lossless after the fix. The draft-logits cache is 8 × 3 × 248320 × 2 B ≈ 12 MB on GPU3. Draft sampling, the logit cache and rejection all run on the last PP rank; the PP relay only forwards token IDs.
- **Cost:**
  - An offline kernel test on an RTX 3080: synthetic p/q through the image's `rejection_sample`, chi-square on output frequencies, with and without the fix. No prod impact.
  - Then one maintenance window. The setting is not in the compile hash, so it is a warm restart.

### B. Draft-side top-k/top-p and a draft temperature τ (patch on top of A)
- **Change:** vLLM's draft sampling ignores top_k/top_p (by design comment). At top-k 20 / top-p 0.95, every draft sampled outside the target's ~5-token nucleus is a certain rejection. Mask the draft logits with the request's top-k/top-p and optionally scale them by 1/τ, *before* they are cached. That keeps it lossless: the ratio test uses the same q that was sampled.
- τ gives a sweepable knob: τ→0 = greedy (today), τ=1 = option A. The best τ is at least as good as either end.
- **Evidence:** #56724: 2.65 → 2.74 from the masking (+15 % over greedy in total). No published τ sweep exists.
- **Cost:** ~30 lines in `sample_draft`, CUDA-graph safe; the same offline test as A.

### C. Block verification (config, only after A)
- **Change:** `"rejection_sample_method":"block"`.
- **Evidence:** +0–4 % length with sampled drafts at k=3–4 (Sun et al.: +8.7 % block efficiency at T=1, γ=8; SGLang #36516: 2.61 → 2.69 at γ=4). Zero with greedy drafts.
- **Risk:** the brief cites a bug (#58784) where block + probabilistic at T>0 accepted all drafts and emitted NUL tokens. Our image predates it. Include it in the offline kernel test.

### D. Retune k (after A/B; affects throughput, not acceptance)
- Flat 0.78 conditional acceptance means a 4th draft adds ≈ +0.36 tokens/step at today's rates. The perfect-greedy ceiling goes 2.83 → 3.25.
- **Cost:** verify width goes 4 → 5 tokens, which means ~25 % more distinct experts read per MoE layer. Plus one more MTP step, each reading the 1.27 GB lm_head on the slow GPU3.
- In the PP2 era, k=1 beat k=3 at much lower acceptance. Measure only.

### E. Fine-tune the MTP head (self-distillation on the INT8 target's own T=1 traces)
- **Evidence:**
  - FastMTP / Speculators on Qwen3-Next: positions 0.897/0.719/0.476 → 0.912/0.776/0.616 (≈ +7 % length, greedy) from ~5k samples.
  - AngelSpec: 51 → 63 % acceptance at T=0.9.
  - A TV-loss objective adds +3–8 pts on top of sampled drafting.
- **Cost:**
  - Capture target multi-stream hidden states (4 × 2560 per token), which needs downtime or a capture hook.
  - Port the trainer to the HC/PLE architecture.
  - Train a 512-expert BF16 layer, realistically on a rented 80 GB GPU; the CMP cards are poor for training.
- **Upside:** under greedy drafting it is capped at ~8 pts (§1). It is larger after A/B.

### F. Hybrid n-gram/suffix + MTP for agentic traffic
- **Evidence:**
  - SuffixDecoding hybrid: 7.5 tok/step on AgenticSQL.
  - SGLang HYBRID_SUFFIX_MTP: 3.2× vs 1.9× on SWE-bench at low concurrency, but worse than MTP on chat at high concurrency.
- **Status:** not in vLLM's V2 runner (#24344 still a draft). That means real implementation work, including the PP draft relay.
- **Sizing:** prod's 90 % prefix-cache hit rate suggests copy-heavy agentic traffic. It can be sized offline, with no prod impact, by measuring how many logged output tokens extend an n-gram match into the prompt (`--enable-log-requests/--enable-log-outputs` are on).

### G. Cheap diagnostics (no restart, need an idle server)
1. **Concurrency check.** In the 10 s log windows, acceptance with 2 running requests (0.51) is below 1 running (0.56). That is confounded by content: those windows are mostly my high-entropy probes. With the batch-invariant stack, a T=0 request's accepted count should be identical alone vs. batched with another. A difference would indicate a batch-dependent draft or relay problem. About 10 minutes.
2. **Context-length dependence.** Upstream reports Qwen3.6-27B per-position acceptance falling from 0.94/0.83/0.72 at 2k context to 0.72/0.51/0.40 at 30k. Prod contexts are 100k+. Measure acceptance against prompt length from the 10 s logs and request log timing.

### Not recommended
- **Lowering temperature:** project rule.
- **Typical acceptance:** lossy; a newer study reports MATH −3.4 pts and AIME −6.7 pts with EAGLE-3.
- **DFlash / DSpark:** no checkpoint exists for this model.
- **`enable_adaptive_verification`:** DSpark-only and incompatible with PP.

## 5. NUL-token bug (vLLM #58784) — does it affect us? (investigated 2026-10-02)

**Upstream bug.** The scheduler pads a newly admitted request that has exactly one token left to compute with k `-1` placeholder drafts, which keeps the batch at the uniform 1+k shape (#45237, #55126). MRV2 overwrites those slots from `req_states.draft_tokens`, which `add_request` zeroed. The rejection sampler then verifies k drafts of token 0. With probabilistic drafting, it uses the slot's stale `draft_logits` as q. When q(0) is tiny, p/q ≫ 1, so all k drafts are accepted. Upstream saw this in 71/80 P/D responses and 0/80 co-located. The fix (#58784, merged 2026-09-28, after our 2026-08-29 image) sets those rows back to `-1` before rejection. It touches one file (+78/−2).

**Our image (live files md5-checked against the mirror):**
- The padding is unconditional, not P/D-only (`scheduler.py:934-950`). It fires when spec decode is on, other requests are running, no prefill was scheduled earlier in the step, and `num_tokens − num_computed == 1`.
- `add_request` zeroes `draft_tokens`, and `combine_sampled_and_draft_tokens` copies them into the draft slots (`states.py:117`, `input_batch.py`). The sampler reads drafts from `input_ids`, so the `-1` never reaches it. This matches upstream before the fix.
- Token 0 is `!` in our tokenizer, so the symptom would be `!` runs, not NUL.
- The related hybrid-state bug (#58434) does **not** apply. Our `mamba_hybrid.py` classifies spec rows by `num_scheduled == drafts + 1`, and GDN uses only `num_decode_draft_tokens_cpu >= 0`. So a padded tail runs the spec-decode GDN kernels with rollback. `num_accepted_tokens` is reset to 1 on add.

**Kernel test (`mtp/nul_bug_kernel_test.py`).** Prod image's `rejection_sample` on an RTX 3080; 4,096 padded rows per case; T=1; k=3.

| Target sampling | Greedy drafts (prod today) | Probabilistic, stale logits (std or block) | Probabilistic + `-1` rows (fix) | Correct p'(`!`) |
|---|---|---|---|---|
| no top-k/top-p, p(`!`)=1e-7 | 0 % | **100 %, 3/3 accepted** | 0 % | ≈0 |
| top-k 20 / top-p 0.95, `!` outside nucleus | 0 % | 0 % | 0 % | 0 |
| top-k 20 / top-p 0.95, `!` inside nucleus | 10.4 % | **100 %** | 9.9–10.1 % | 10.5 % |

Block verification makes no difference; probabilistic drafting alone is enough. With greedy drafts, a padded row is lossless and only wastes draft slots. The block kernels in our image already guard against `-1`.

**Frequency in prod.** (Superseded by §7: with MTP the only trigger is a 1-token prompt.) A padded row needs a 1-token tail. In align mode, prefix hits are multiples of 1600 and capped at `num_tokens − 1`, so that requires one of:
- an exact-repeat prompt (fully cached) whose length − 1 is a multiple of 1600, or
- a preemption resume at such a length.

Either must also coincide with another running request. Since boot (~5 h): 489 requests, 0 preemptions, ≥2 running in ~12 % of 10 s windows, and 94 % of prompt tokens cached (new-turn tails are almost always > 1 token). Expected occurrences ≪ 1 per day. Rare, but not impossible.

**Verdict:**
- Prod today (greedy) is not affected.
- Enabling A (with or without C) exposes us. When a padded row occurs, the first output token becomes `!` with near-certainty whenever `!` is inside the target nucleus. That is uncommon at the start of a thinking response, but common mid-text in code (`!=`, `#!`, `![`) on a preemption resume.
- A client request with top_p=1 / top_k=-1 removes the nucleus protection and gives upstream-style `!!!` prefixes.
- **Backport #58784 (or disable scheduler padding) as part of the A/B/C patch set, together with the Gumbel-noise fix in §2.**

## 6. Implementation and offline validation (2026-10-02)

**Image:** `qwen38-flash-next:ple-fp8-pp3-detmoe-inv-det-spec` = prod image + four patched files (`mtp/deploy_spec/`). The files were taken from the live container, so they include PP patches 0011/0012. Every change is behind an env switch that defaults to off, so with no switches set the image is the stock prod code path.

| Switch | What it does | File |
|---|---|---|
| `VLLM_SPEC_REJECT_UNPROPOSED=1` | Backport of vllm#58784. Draft rows of a step that starts inside the prefill go to `-1`. A diagnostic counter is logged as `spec-fix(#58784)`. | `rejection_sampler.py` |
| `VLLM_SPEC_DRAFT_INDEP_NOISE=1` | Draft Gumbel noise from `seed ^ const`, independent of the target's residual-resample noise (§2). | `speculator.py` |
| `VLLM_SPEC_DRAFT_TOPKP=1` | Draft logits masked with the request's top-k/top-p, computed at the request temperature. Graph-safe: static top-64 candidates, cumsum, scatter. | `speculator.py`, `model_runner.py` (hands over sampling states) |
| `VLLM_SPEC_DRAFT_TAU=<t>` | Draft temperature scale. | `speculator.py` |
| `VLLM_SPEC_STEP_UNIQUE_RNG=1` | Block-verification RNG fix, see below. Requires INDEP_NOISE. | `rejection_sampler_utils.py`, `speculator.py` |
| (always on) | vllm#59355 OOB guard. Inert in our config, because the last row is always a bonus or position-0 row. | `rejection_sampler_utils.py` |

B's masked/scaled logits are cast to the draft-cache dtype before sampling, so the cached q is bitwise the distribution that was sampled from.

**New finding: block verification (option C) is biased in this image and in upstream main.**
- In block mode the kernel draws `u = rand(seed, absolute_pos)` at *every* draft row, because all rows are examined to find the longest acceptable prefix. Every draft token is examined too.
- The next step restarts right after the accepted prefix and reuses the (seed, position) keys of the rejected tail, for both the uniforms and the draft Gumbel noise. Those draws were conditioned on the previous decision.
- Token-wise verification never draws past the first rejection, so it is unaffected.
- Fix (`VLLM_SPEC_STEP_UNIQUE_RNG`): block-mode uniforms are keyed by `seed ^ (first_pos_of_step × 0x2545F491)`, and draft noise by `first_pos × 8 + draft_index` in the INDEP seed domain.

**Kernel exactness test** (`mtp/spec_kernel_exactness.py`, RTX 3080, the image's real kernels):
- Setup: 32,768 sequences per variant, k=3 multi-step generation of 4 tokens, adversarial target/draft pairs spread across vocab blocks.
- Measured: chi-square per output position and TV vs the target (noise floor ≈0.005), plus pos-0/pos-1 independence.

| Variant | min p | max TV | accepted/step |
|---|---|---|---|
| prod today (greedy, token-wise) | 0.089 | 0.005 | 0.391 |
| stock probabilistic (shared noise) | **0.000** | **0.060** | 1.056 |
| A (indep noise) | 0.339 | 0.004 | 1.053 |
| A+B (draft top-k/top-p) | 0.153 | 0.005 | 1.095 |
| A+B, tau 0.7 | 0.461 | 0.003 | 0.989 |
| A+B+C, stock RNG keys | **0.000** | **0.046** (indep p = 0) | 1.077 |
| A+B+C + step-unique RNG | 0.649 | 0.003 | 1.147 |
| A+B+C tau 0.7 + step-unique RNG | 0.513 | 0.003 | 1.041 |
| greedy + block + step-unique RNG | 0.031 (min of 4) | 0.005 | 0.393 |

The synthetic accepted/step values only show direction; real-model gains are measured in the window. The #58784 mechanism and fix were validated separately (§5, `mtp/nul_bug_kernel_test.py`).

**Window plan** (`mtp/dt_window_spec.sh`, gates in `mtp/stage_check.py`, generation suite `mtp/gen_suite.py`, baseline `mtp/baseline_prod.sh`):
- Prod baseline (live, no restart): KLD corpus top-20, plus the generation suite.
  - The suite runs T=1 acceptance with calibration of every sampled token against the target's top-k/top-p distribution: tokens outside the nucleus, NLL vs entropy, argmax rate, PIT.
  - Plus padded-row `!` trials and greedy byte-identity.
- Stages S0 (switches off) → S1 fixes+A → S2 A+B (tau 1.0, 0.8) → S3 A+B+C, each in a test container on :8002.
- Then deploy the furthest stage that passed, and re-validate prod.

## 7. Window results and deployment (2026-10-02, times CDT)

Prod was down 12:24–2:26 PM for stages S0–S3 in a test container, then 2:44–2:50 PM (rollback) and 2:53–2:59 PM (final deploy). **Live since 2:59 PM, validated 3:16 PM.**

Live config: image `-spec`, `--speculative-config {"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}`, env:
- `VLLM_SPEC_REJECT_UNPROPOSED=1`
- `VLLM_SPEC_DRAFT_INDEP_NOISE=1`
- `VLLM_SPEC_DRAFT_TOPKP=1`
- `VLLM_SPEC_DRAFT_TAU=1.0`
- `VLLM_SPEC_PAD_NEW_REQS=0`

The rollback container is `qwen38-flash-next-pp3-int8all-prespec`.

T=1 suite (24 fixed prompts and seeds, ~31k tokens, prod sampling, logprobs on). Every config is compared to the same baseline:

| Stage | accept len | per position | tok/s | gates |
|---|---|---|---|---|
| prod before (greedy drafts) | 2.354 | .626/.424/.305 | 52.5 | baseline |
| S0 patched image, switches off | 2.354 (identical tokens) | same | 52.5 | all pass |
| S1 fixes + A | 2.772 | .764/.573/.435 | 61.6 | all pass |
| S2 A+B tau 1.0 | 2.806 | .769/.586/.451 | 61.8 | all pass |
| S2 A+B tau 0.8 | 2.763 | .758/.570/.435 | 60.6 | test crashed in pad part (bug 3) |
| S3 A+B+C (block + step-unique RNG) | 2.815 | .760/.591/.464 | 61.4 | exact calibration pass; pad part hit bug 3; owner skipped C |
| **deployed: A+B tau 1.0 + no new-request padding** | **2.806** | .769/.586/.451 | **61.9** | **all pass** |

Paired per-prompt differences:
- A vs baseline: +0.375 ± 0.052 (21/24 prompts improved).
- B vs A: +0.033 ± 0.034.
- tau 0.8 vs 1.0: −0.051 ± 0.036.
- C vs A+B: +0.000 ± 0.047.

**Every stage kept the target bit-identical:** 77×2048 corpus at top-20, KLD 0, same top-1 / same top-p set 100 %, PPL 3.926888 both, greedy 16/16.

Sampling exactness:
- Exact calibration (top_k off): 0 tokens outside the top-p set in all stages, |z| ≤ 1.6.
- Prod-settings calibration: |z| ≤ 1.8, PIT p ≥ 0.16.

**Bug 3 (pre-existing in the old prod): fresh 1-token prompts.**
- When a new request has exactly 1 token left and other requests are running, the scheduler pads it to the 1+k spec shape.
- The hybrid model state then runs the row as a spec-decode row. GDN starts from a state slot that was never initialised for this request.
- Result: NaN logprobs (HTTP 400 "Out of range float values are not JSON compliant: nan") or garbage text ("ductductduct…").
- Old prod, 12 trials while 6 requests decode: 11/12 failed (`results/padprobe_prod_stock.json`).
- Upstream main pads the same way (`scheduled_running_reqs or num_computed_tokens > 0`).
- Fix: `VLLM_SPEC_PAD_NEW_REQS=0` (scheduler). Deployed prod: 24/24 trials OK.

**Correction to §5:** with MTP, `use_eagle()` makes the prefix cache drop one block (1600 tokens) from every hit. Repeats and preemption resumes therefore never leave a 1-token tail, and the only padding trigger here is a 1-token prompt. The #58784 counter fired only in the deployed attempt-1 run (72 rows = 24 trials × 3). With padding off it cannot fire.

Possible upstream reports (not filed):
- block-verification RNG reuse (§6);
- padded fresh 1-token prompts on hybrid models (bug 3).

Harness gaps found and fixed during the window:
- a leftover line crashed the first S0 run, which was rolled back;
- string-based token matching and bf16 ties in the calibration;
- `pgrep -f` self-matching;
- the checker passed stages with missing parts; a missing part is now a failure.

## 4. Suggested sequence
1. Offline, on an RTX 3080: kernel test of the noise fix (A), masking + τ (B) and block verification (C). Also build a T=1 benchmark: fixed prompts and seeds, thinking on, ~40k tokens per arm, using the clean-window check from `ceiling_probe.py`.
2. One maintenance window (owner's OK): baseline, then A, A+B (τ ∈ {0.6, 0.8, 1.0}), and A+B+C, each with the T=1 benchmark. Restore prod afterwards.
3. Decide on k (D) with the winning sampler. E and F only if the result is still short of the target.
