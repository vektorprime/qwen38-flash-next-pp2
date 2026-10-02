> Copied from the working notes (`batchinv/NOTES.md` on the host). Paths like `hook3/`, `results/`, `dt_window*.sh` refer to
> this directory as `hook/`, `results/`, `scripts/`; tests are in `tests/`.

# Batch invariance: why equivalent runs disagree 4–6% on top-1, and the fix (2026-10-02)

Owner's challenge: "95–96% same top-1 between equivalent runs is not normal — prove what is going on and fix it."

**Status (2026-10-02, morning):** see §0. All measurements are on prod's own stack: PP3 on CMP 170HX
(GPUs 0, 1 = 64 GB; GPU 3 = 40 GB), image `qwen38-flash-next:ple-fp8-pp3-detmoe`, serve args of prod (MTP k=3, prefix
caching, async scheduling, max-num-seqs 8, 524288 context with YaRN ×2).

## 0. Summary
* **Cause (proven):** two independent sources of 1-ulp differences, both amplified by MoE routing into 3–6% top-1 flips
  (KLD ≈ 0.014; §2).
  1. **Batch/chunk shape:** a request's logprobs depended on what else was in the same forward step and on where its
     prompt was split into steps. Seeds:
     - the Marlin MoE stream-K split;
     - cuBLAS kernel choice by M;
     - the QSA split-K profile;
     - prefill chunk boundaries that move with prompt length;
     - above 2048 tokens of context, the QSA indexer's top-k order.
  2. **Container instance:** every new container re-autotunes vLLM's GDN prefill kernels by timing. One of them
     (`chunk_fwd_kernel_o`) picked BK=64 in one container and BK=128 in another, which is a different summation order.
     So any two separately started containers differed by KLD 0.014 (§4a).
* **Fix:**
  - `hook3/` (rev 2), an env-gated `sitecustomize` hook with no vLLM rebuild (§3).
  - Plus `VLLM_TRITON_FORCE_FIRST_CONFIG=1`, `TORCHINDUCTOR_DETERMINISTIC=1` and
    `--compilation-config '{"inductor_compile_config":{"combo_kernels":true,"benchmark_combo_kernel":false}}'` (§4a).
* **Validated on prod's stack:**
  - The same request gives the same logprobs whatever the load, on every position (36/36 prompt-length pairs; decode
    16/16 under concurrency; 5000-token prompts 3/3).
  - Two **separately compiled containers** are bit-identical on the whole 77×2048 KLD corpus.
  - Decode speed is unchanged (73.0 vs 72.5 tok/s stock). 2048-token prefill is +6% (0.580 vs 0.546 s).
* **Quality on that stack (§5):**
  - Floor: exactly 0.
  - **INT4 vs INT8:** KLD 0.040, top-1 93.4%, same-token chance 94.1%.
  - **FP8 PLE vs BF16 PLE:** KLD 0.016, top-1 96.0%, same-token chance 96.4%; perplexity unchanged.
  - Highest fidelity: INT8 experts + BF16 PLE, the prod checkpoint.
* **Deployed (§6):** prod runs INT8 + BF16 PLE on the deterministic invariant stack, image
  `qwen38-flash-next:ple-fp8-pp3-detmoe-inv-det`. The previous containers are kept stopped for rollback.

## 1. What changes between "equivalent" runs: the batch shape of a forward step (proven)
* Live prod, prompt lengths 2024–2048: logprobs are **bit-identical exactly when the first forward step has the same size**.
  Prod runs a lone L-token prompt as a first step of floor8(L)−8 tokens plus a small second step. The groups were
  {2040..2047}→step 2032, {2032..2039}→2024 and {2048}→2040. A different step size means every position differs, with
  6% top-1 flips.
* Kernel census (offline, `census_*.py`, `marlin_*.py`; RTX 3080, and in-window on the CMP 170HX):
  - **Marlin MoE** (INT8/INT4 experts) is never batch-invariant.
    - Its scheduler (`marlin_template.h`, "DP + two-tile stream-K") splits the K-reduction of the last ⅓–1⅓ grid of
      output tiles across threadblocks.
    - Which tiles are split, and where, depends on the total tile count, i.e. on the batch.
    - Result: 1-ulp differences in ~0.6% of rows per MoE layer. Source: `marlin_src/`.
  - **cuBLAS dense GEMMs** are invariant around M≈2040, but they pick different kernels at other M (decode sizes 1/2–16/24–64,
    and 1024 vs 2048). So they are not invariant across decode batch sizes or prefill chunk sizes. Disabling split-K does
    not fix this on SM80.
  - **QSA attention** chooses its split-K profile from the number of tokens in the batch, so it is not invariant at decode.
  - **Chunk boundaries** (found 2026-10-02 in window 5):
    - Where a prompt is split into prefill steps depends on its length L. The scheduler stops a chunk at the last
      cacheable position floor8(L)−8, and at floor8 of the token budget.
    - A GDN layer continued from a cached state at a position that is not a multiple of its chunk size (64) is not
      bit-identical to computing straight through that position.
  - **QSA indexer top-k** (found 2026-10-02 in window 5):
    - Once a query sees more than 512 compressed blocks (context > 2048 tokens), `torch.ops._C.persistent_topk` returns
      the selected blocks in a run-dependent order.
    - With exact score ties at the cut-off it also returns a run-dependent *set*. ReLU scores make exact ties at 0.0
      common.
    - Sparse attention sums the tokens in that order, so two identical 5000-token requests produced different text.
    - At ≤ 2048 tokens of context it returns 0..n−1 in ascending order, which is deterministic.

## 2. Why a 1-ulp seed becomes 4–6% top-1 flips: MoE routing amplification (measured)
E1 (invariant Marlin, fixed batch), 1-ulp injections into the MoE output (`hook3` inject), 20 hardest windows:

| seed | top-1 flips | flips >1 nat | positions changed |
|---|---|---|---|
| 32 values (~2 tokens), layer 0 | 2.72% | 81 | 51% |
| 240 values (~15 tokens), layer 0 | 4.08% | 120 | 76% |
| same, last layer (47) only | 0.00% | 0 | 0.7% |
| same, every layer (≈ stock Marlin) | 4.16% | 132 | 76% |
| layer-0 seed + routing replayed from clean run | 1.56% | 7 | — |
| reference: stock prod, prompt 2048 vs 2047 | 5.96% | 206 | 99.8% |

* **Routing capture** (stock, 2048 vs 2047): 0% of tokens are re-routed at layer 0, 1.3% at layer 1, and ~45% per layer
  from layer 30 on.
* **Replay:** replaying the clean run's expert choices removes 94% of confident flips.
* **What this explains:** every earlier comparison. Eager vs compiled, FP8 vs BF16 PLE and 2047 vs 2048 all land at
  ~4–6% because any seed is amplified the same way.

## 3. The fix (`hook3/`, rev 2; env-gated `sitecustomize`, no vLLM rebuild)

| flag | what | how | cost |
|---|---|---|---|
| `PLEFP8_INV_MOE=1` | Marlin MoE | `inv_marlin.py` | decode within noise; prefill = stock |
| `PLEFP8_INV_GEMM=1` | all unquantized linears, incl. router gate and lm_head | `inv_gemm.py` | dense GEMMs per forward: decode 9.0 ms vs cuBLAS 10.0 (M=4) / 11.3 vs 10.6 (M=32); prefill 171 vs 142 ms per 2048 tokens |
| `PLEFP8_INV_QSA=1` | QSA attention + indexer selection | split profile pinned; `inv_qsa_select.py` | none measurable |
| `PLEFP8_ALIGN_CHUNKS=64` | scheduler | every non-final prefill chunk ends on a multiple of 64 | ≤ 63 extra tokens in the last chunk |
| `VLLM_TRITON_FORCE_FIRST_CONFIG=1` (vLLM) | every `@triton.autotune` kernel, incl. the GDN prefill kernels | first valid config instead of a timed choice (BK changes the summation order) | 1–6 µs per GDN kernel call; prefill +1% |
| `TORCHINDUCTOR_DETERMINISTIC=1` + `benchmark_combo_kernel: false` | torch.compile (Inductor) | no timing-based choices that affect numerics | none measured |

How each fix works:

* **`PLEFP8_INV_MOE` (Marlin MoE):**
  - thread_k=64 for every block size. thread_k changes per-row results; thread_n and blocks_per_sm do not.
  - Empty expert blocks are appended on the device so the stream-K region lies only in padding. Every real tile is then
    computed whole, in a fixed K order. This is one Triton kernel, graph-safe.
  - Config `PLEFP8_MARLIN_CFG=64,128,2,128,2` (CMP-tuned).
* **`PLEFP8_INV_GEMM` (dense linears):**
  - Each output element is accumulated over K in 16-wide MMA steps, ascending, whatever the tile config. So tiles can
    follow M without changing any result.
  - Narrow shapes (N ≤ 1024, K ≥ 2048) use split-K with S fixed per weight shape and a fixed-order reduction.
  - Tiles were tuned on the CMP (`tune_cmp.py`, `tune_splitk.py` → `tuned_gemm.json`).
  - It runs as an opaque custom op `plefp8::inv_linear`. It needs a fresh torch.compile cache.
* **`PLEFP8_INV_QSA` (QSA attention + indexer selection):**
  - The QSA split profile is pinned (block_n 64, 8 splits), and rows are processed 256 at a time.
  - Indexer selection is made deterministic (rev 2). The op's own selection gives the k-th largest score T, which is
    unique. One Triton program per row then writes, in ascending block order, every block with score > T, plus ties
    ≤ T by lowest index.
  - Rows with ≤ 512 visible blocks are unchanged.
* **`PLEFP8_ALIGN_CHUNKS=64` (scheduler):**
  - Wraps `_mamba_block_aligned_split` (rev 2) and `_reserve_prefill_lookahead`.
  - A non-aligned stop is rounded down. If that leaves nothing to compute, the chunk runs on to the prompt end.
  - 64 is the GDN chunk size. A continuation from a cached state at a multiple of 64 is bit-identical to computing
    through it (measured: positions 1920–1976 in the 2048-vs-1985 pair).
  - Skipping a cache stop only means that state is not cached.

Rev 1 (window 5 E3) lacked the mamba-split alignment and the QSA selection fix. Rev 2 adds both (§4: deploy validation).

## 4. Validation

| | stock prod | E3 (rev 1, window 5) | prod rev 2 (window 6) | **final prod (window 9)** |
|---|---|---|---|---|
| prompt lengths 1100…2048 (9 lengths, 20 windows), ALL common positions bit-identical | groups only | 10/36 pairs (rest: last 7–14 positions) | 36/36 | **36/36** |
| greedy, 16 prompts: repeat / 8 concurrent / 4 concurrent + load, identical | — / 7/16 / 7/16 | 16 / 16 / 16 | 16 / 16 / 16 | **16 / 16 / 16** |
| 5000-token prompt: repeat, and with 3 concurrent long prefills (3 prompts, 48 tokens + logprobs) | — | 0/3 (QSA top-k order) | 3/3 | **3/3** |
| separately compiled container of the same config bit-identical (20×2048 probe; KLD corpus) | — (same mechanism) | no (vs prod rev 2: 99% of positions differ) | no (D1 vs D2: KLD 0.014) | **yes** (vs D3; D3 vs D4 also on the full corpus) |
| decode tok/s (bench.py), MTP acceptance | 72.5–72.8, 0.728 | 72.1 / 72.7, 0.728 | 73.6 / 73.2, 0.731 | **73.6 / 73.2, 0.7395** (= D3's, to the digit) |
| 2048-token prefill, median of 6 | 0.546 s | 0.584 s | 0.575 s | 0.580 s (+6%) |
| context | 524288 | 524288 | 524288 | 524288 (after one warm restart) |

Rev 1 → rev 2 changed the mamba-split alignment and the QSA selection. Rev 2 → final added the instance switches (§4a).

## 4a. Compile/process-instance variance (found 2026-10-02 08:30–09:30, windows 7–8)

* **Symptom.** Deployed prod vs E3 differ at 99% of positions (3.3% top-1 flips), although rev 2 changes nothing for
  ≤ 2048-token prompts except the last 7 positions. E3 vs E3b (`docker restart`, same container filesystem and compile
  cache) were bit-identical.
* **Measured on the 77-window KLD corpus**, same config (INT8 + BF16 PLE), two fresh containers D1 and D2, each with its
  own empty compile cache:
  - **KLD 0.0140**, top-1 96.30%, same-token chance 96.71%.
  - This is the old "≈0.014 equivalence floor". It was not batch noise only: any two separately started containers
    differ by this much.
* **Cause, from diffing the two caches.**
  - Inductor generated identical code (the same 355 Triton kernels). Its own autotune differences were pointwise
    `XBLOCK` only, which does not affect numerics.
  - vLLM's GDN prefill kernels (FLA, `@triton.autotune`, re-tuned by timing in every new container) chose differently:
    `chunk_fwd_kernel_o` took **BK=64 in D1 and BK=128 in D2**.
  - FLA accumulates `b_o += tl.dot(b_q, b_h)` and `b_A += tl.dot(b_q, b_k)` per K block, so BK changes the summation
    order. That gives 1-ulp seeds, amplified by MoE routing (§2).
  - `chunk_scaled_dot_kkt` has the same kind of BK knob (`b_A += tl.dot(...)`); here both instances happened to pick
    BK=32.
  - Within one container the choice is cached (vLLM sets `TRITON_CACHE_AUTOTUNING=1`), which is why restarts matched.
* **Consequence for §5:** the window-5 INT4 and FP8 runs each started from a fresh cache, so their KLD contains this
  0.014 of instance noise. FP8 PLE's 0.0169 is therefore not distinguishable from noise without this fix. INT4's 0.041
  is.
* **Fix (window 8):**
  - `VLLM_TRITON_FORCE_FIRST_CONFIG=1`, vLLM's own switch. Every `@triton.autotune` kernel uses its first valid config
    instead of a timed choice.
  - Plus Inductor's deterministic mode, `TORCHINDUCTOR_DETERMINISTIC=1`, with
    `--compilation-config '{"inductor_compile_config":{"combo_kernels":true,"benchmark_combo_kernel":false}}'`.
    It costs no speed (D1: 72.2 / 72.7 tok/s) and removes the remaining timing-based Inductor choices that could affect
    reductions.
  - Cost of the first configs: 1–6 µs per GDN prefill kernel call (e.g. `chunk_fwd_kernel_o` 19.5 vs 13.3 µs at the
    tuned size).
  - Criterion: two fresh containers must give bit-identical KLD-corpus files.
* **Result (window 8):** D3 and D4 are two fresh containers, each with its own empty compile cache, under the fix.
  - **Bit-identical** on the 20×2048 probe **and on the whole 77×2048 KLD corpus (K=200)**.
  - vLLM logged all 6 GDN prefill kernels as pinned to config 0.
  - Speed is unchanged: decode 73.0 / 72.9 tok/s (MTP acceptance 0.739); 2048-token prefill 0.580 s (rev-2 prod 0.575 s).

## 5. Quality on the deterministic invariant stack: INT4 vs INT8, FP8 PLE vs BF16 PLE (window 8)

**Setup:**
- Prod's serving config + hook rev 2 + `TORCHINDUCTOR_DETERMINISTIC=1` + `VLLM_TRITON_FORCE_FIRST_CONFIG=1` +
  `benchmark_combo_kernel: false`, `--max-logprobs 200`.
- Every config ran in its own fresh container with its own empty compile cache.
- Corpus: the frozen 77×2048 KLD corpus (`kld/corpus`), scored with `kld/score.py collect`, K=200.
- Metrics are those of `kld/diag/audit/extended.py` (via `plefp8/e2e/compare_ab.py`), on positions 32–2047 (155,232 positions).
- Sampling = prod's sampler (top-k 20, top-p 0.95, T=1). 95% CIs are bootstrapped over windows.

| | KLD mean [95% CI] | median / p99 | top-1 same | top-5 overlap | sampling set identical | same-token chance | ppl | ΔNLL [95% CI] |
|---|---|---|---|---|---|---|---|---|
| **floor**: INT8+BF16 PLE, two fresh containers (D3 vs D4) | **0** (bit-identical) | 0 / 0 | **100%** | 100% | 100% | **100%** | 3.927 / 3.927 | 0 |
| **INT4 vs INT8** (both BF16 PLE; INT4 = original AWQ g32 hub checkpoint) | **0.0404** [0.0338, 0.0479] | 0.0065 / 0.538 | **93.41%** [92.83, 93.95] | 87.8% | 63.4% | **94.09%** [93.65, 94.52] | 3.927 / 3.906 | −0.0053 [−0.0087, −0.0022] |
| **FP8 PLE vs BF16 PLE** (both INT8 experts) | **0.0163** [0.0132, 0.0200] | 0.0022 / 0.207 | **95.98%** [95.54, 96.36] | 92.2% | 71.4% | **96.38%** [96.08, 96.67] | 3.927 / 3.931 | +0.0009 [−0.0025, +0.0046] |
| *for scale:* same config, two fresh containers **without** the instance fix (D1 vs D2) | 0.0140 [0.0112, 0.0175] | 0.0019 / 0.167 | 96.30% | 92.7% | 72.5% | 96.71% | 3.924 / 3.929 | +0.0013 [−0.0017, +0.0043] |

By domain (KLD / top-1 same / same-token chance):

| domain | INT4 vs INT8 | FP8 PLE vs BF16 PLE |
|---|---|---|
| A arXiv | 0.027 / 94.0% / 94.6% | 0.011 / 96.4% / 96.8% |
| B code | 0.029 / 95.4% / 95.6% | 0.011 / 97.2% / 97.3% |
| C wiki-en | 0.043 / 92.4% / 93.1% | 0.016 / 95.5% / 95.9% |
| D multilingual | 0.028 / 93.0% / 94.0% | 0.009 / 96.2% / 96.6% |
| E GitHub issues | 0.116 / 88.8% / 90.6% | 0.055 / 92.1% / 93.6% |

Reading it:
* **The floor is exactly zero.** Two separately started and separately compiled containers give bit-identical logprobs
  on all 155k scored positions. Every non-zero number above is the quantization's own effect.
* **INT4 vs INT8:** KLD 0.040, top-1 93.4%, same-token chance 94.1%.
  - This is in line with the 2026-09-29 eval (`kld/results/v2`, FP8 PLE on both sides, batch-1 server with patch 0015):
    KLD 0.0413, 93.27%, 94.01%.
  - INT4 has *lower* perplexity (−0.5% NLL, CI excludes 0), mostly on GitHub issues. As in kld-eval, perplexity is not a
    fidelity measure here.
* **FP8 PLE vs BF16 PLE: a real but minimal shift.**
  - KLD 0.016, top-1 96.0%, same-token chance 96.4%. Perplexity is unchanged (ΔNLL CI includes 0).
  - The size equals what *any* 1-ulp-class seed produces through MoE routing: a different GDN tile config alone gives
    0.014 (last row).
  - So FP8 PLE moves the output distribution, but by about the smallest amount a numeric change can on this model. It
    is not measurably worse on next-token likelihood.
  - It was not distinguishable from noise before tonight. The ~0.014 "floor" earlier in the day was the instance noise
    of the last row.
* **Highest fidelity of the three:** INT8 experts + BF16 PLE, prod's checkpoint `ple-bf16-int8all-38f`. It is the
  closest to the original weights in both comparisons (INT8 experts 1.5% expert error vs INT4 14%; BF16 PLE is the
  original table). This is what is deployed (§6).

Superseded measurements (window 5, rev 1, a fresh compile per config, so the noise in the last row above was included):
INT4 vs INT8 KLD 0.0413 / top-1 93.31%; FP8 vs BF16 PLE KLD 0.0169 / top-1 95.93%. REF twice from one compile cache:
bit-identical.

Speed (bench.py: 16 prompts, greedy, 256 tokens, single stream; prefill = median of six 2048-token prompts):

| | decode tok/s | MTP acceptance | 2048-token prefill |
|---|---|---|---|
| stock prod (INT8 + BF16 PLE, no fixes) | 72.5–72.8 | 0.728 | 0.546 s |
| hook rev 1 (E3) | 72.1 / 72.7 | 0.728 | 0.584 s |
| hook rev 2 (deployed in window 6) | 73.6 / 73.2 | 0.731 | 0.575 s |
| hook rev 2 + instance fix (D3) | 73.0 / 72.9 | 0.739 | 0.580 s |
| INT4 + BF16 PLE, same stack (Q4f) | 70.9 | 0.719 | — |
| INT8 + FP8 PLE, same stack (FP8f) | 72.0 | 0.736 | — |

## 6. Deployment (window 9, 2026-10-02 10:37–11:00 UTC)

* **Live:** container `qwen38-flash-next-pp3-int8all`, restart=always, port 8001.
  - Image `qwen38-flash-next:ple-fp8-pp3-detmoe-inv-det`. It is built from `deploy/Dockerfile` (the hook in
    `/opt/plefp8_hook`, PLEFP8_* flags, TRITON_CACHE_DIR on the cache volume) plus `deploy_det/Dockerfile`
    (TORCHINDUCTOR_DETERMINISTIC=1, VLLM_TRITON_FORCE_FIRST_CONFIG=1).
  - Serve args, env, GPUs and volumes are prod's, plus `--compilation-config '{"inductor_compile_config":{"combo_kernels":true,"benchmark_combo_kernel":false}}'`.
  - Compile cache volume: `vllm-cache-prod-det`. Checkpoint unchanged: `ple-bf16-int8all-38f` (INT8 experts, BF16 PLE).
* **Rollback containers** (stopped, restart=no):
  - `qwen38-flash-next-pp3-int8all-rev2`: hook rev 2 without the instance switches; image `-inv`, volume vllm-cache-inv.
  - `qwen38-flash-next-pp3-int8all-stock`: no hook, image `qwen38-flash-next:ple-fp8-pp3-detmoe`, volume vllm-cache.
  - `qwen38-flash-next-pp3-int8all-fp8ple`: older.
  - To roll back: `docker stop` + `docker rename` the live container away, rename the chosen container to
    `qwen38-flash-next-pp3-int8all`, `docker start` it and `docker update --restart=always` (`dt_window9.sh` rollback()).
* **First boot on a fresh compile cache** auto-fits the context to ~445k. One warm restart gives 524288; the window
  scripts do this.
* **To turn a fix off:** set its env var to 0 in a recreated container. A changed graph needs a fresh compile volume.

## 7. Notes, residuals, observations
* A prefix-cache hit that lands inside previously *generated* text continues from a state computed by the decode path
  (recurrent GDN). Recomputing the same text by prefill (chunked GDN) is numerically different, so a multi-turn request
  that hits the cache inside the previous answer can differ slightly from the same conversation sent fresh. This is
  inherent to hybrid recurrent models. Prefill-to-prefill continuation is exact (§3).
* Observed (stock and all variants): on this config the prefix cache reported `cached_tokens = 0` for an identical 5000-token prompt sent twice.
  The attention blocks are 1600 tokens and mamba states exist only at chunk ends, so hits are rare. This is the same on
  stock and is not investigated here.
* Marlin launch configs are GPU-specific for validity. 64,128,2,128,2 is invalid on an RTX 3080 (49.6 KB of shared
  memory per block at 2 blocks/SM), but valid and invariant on the CMP 170HX for INT8 and for INT4 (AWQ g32, zero
  points).

## Files
* `hook3/` (rev 2): the fix. `hook3_rev1/` is the version used in window 5.
  - `sitecustomize.py`: the patches and env gates.
  - `inv_marlin.py`, `inv_gemm.py`, `inv_qsa_select.py`: the kernels.
  - `tuned_gemm.json`: CMP-tuned tile configs.
* `deploy/Dockerfile` (`-inv` image) and `deploy_det/Dockerfile` (`-inv-det` image).
* Experiments:
  - `dt_window3.sh` + `dt3_*.py`: the mechanism and rev-0 validation.
  - `dt_window4.sh` + `tune_cmp.py` + `tune_splitk.py` + `build_tuned_table.py`: CMP tuning.
  - `dt_window5.sh` + `dt5_*.py`: rev-1 validation and the first KLD runs.
  - `dt_window6.sh`: rev-2 deploy.
  - `dt_window7.sh`: deterministic Inductor alone, not enough.
  - `dt_window8.sh`: instance invariance and the clean KLD runs.
  - `dt_window9.sh`: final deploy.
  - `kld_pairs.py`: KLD metrics for any pair.
* Offline tests (RTX 3080):
  - `test_hook3.py`, `test_inv_marlin_int4.py`, `test_sched_align.py`, `test_qsa_topk.py`, `test_qsa_select_fix.py`,
    `test_hook_qsa_select.py`, `test_fla_autotune.py`, `triton_mm_invariance.py`, `test_fused_pad.py`.
* Results:
  - `results/kld5/pairs.json`: the clean KLD numbers (REF_f1/REF_f2/Q4_f/FP8_f; REF_d1/REF_d2 for instance noise).
  - `results/kld5/compare.json`: window 5.
  - `results/dt5_analysis_*.json`, `results/decode_*.json`, `results/bench_*.json`, `results/prefill_*.json`,
    `results/prefix_*.json`, `results/tune_*.json`.
  - Logs in `logs/`.
