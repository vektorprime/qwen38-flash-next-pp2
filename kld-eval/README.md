# INT4 vs INT4+8 vs INT8: output-distribution eval (2026-09-29)

How far does each routed-expert quantization move the model's next-token
distribution? And does promoting 5 expert layers to INT8 (the production
"INT4+8" checkpoint) measurably help?

Everything except the routed experts is identical across the three variants:
BF16 dense/attention/router/shared expert, the FP8 PLE table (`../ple-fp8/`),
and the same config and tokenizer.

| Variant | Routed experts | Role |
|---|---|---|
| **INT8** | all 48 layers INT8 symmetric per-channel (W8A16), re-quantized from Qwen BF16 | reference (stands in for BF16) |
| **INT4+8** | layers 2, 4, 30, 46, 47 INT8 (same method); the other 43 are the original AWQ INT4 | production |
| **INT4** | `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` (INT4 g32 asymmetric) | original |

How the INT8 experts were built:

- gate/up = `INT8(W_bf16 * s)` in the AWQ-smoothed space, with `s` recovered
  from the checkpoint router against BF16;
- down = `INT8(W_bf16)`;
- MSE clip search over 21 ratios.

Expert-output relative error against BF16, averaged over 16 sampled experts
per layer: INT8 **1.51%** (max 1.72%), INT4 **13.94%**.

## Method

- **Corpus:** 77 windows of 2048 tokens (157,696 tokens). Every document was
  first published on or after 2026-09-01; the model was published 2026-08-24.
  | Domain | Content | Tokens |
  |---|---|---|
  | A | arXiv LaTeX | 45k |
  | B | source files newly added to vllm / llama.cpp / pytorch / transformers / tokio / ruff / vscode / kubernetes | 45k |
  | C | English Wikipedia current events and new articles | 35k |
  | D | new zh / de / ja / es Wikipedia articles | 16k |
  | E | new vllm / llama.cpp GitHub issues | 16k |

  The corpus is kept local and is not published here.
- **Collection:** `/v1/completions` with token ids as the prompt, `max_tokens 1`,
  `temperature 0`, and `prompt_logprobs` top-100 for INT8 and top-200 for the
  others. One request at a time.
- **Eval server:** PP=3, with layers split 19/19/10 over 2x64 GB + 40 GB CMP
  170HX. MTP, prefix caching and async scheduling are off; the YaRN rope
  override is kept. INT8 weights take 51.1 / 51.1 / 28.6 GiB per rank.
- **Scored positions:** 32-2047 of each window, **155,232** in total. Every
  position is teacher-forced on the real text; nothing is generated.
- **Metrics:**
  - **KLD:** a lower bound (the reference's top-100 tokens plus one "rest"
    bucket). The INT8 top-100 covers 98.2% of the probability on average.
  - **Top-1 same:** whether the argmax token matches.
  - **Sampling set:** the production sampler (top_k 20, then top_p 0.95 with
    vLLM's tail-drop rule), rebuilt exactly from the stored top-20.
  - **Same-token chance:** 1 - total variation between the two sampling
    distributions, i.e. the chance both models sample the same token if they
    share the random draw.
- **Reproducibility:** requires `../detmoe/` (patch 0015). With it, scoring
  INT8 twice gives bit-identical files (noise KLD 0, top-1 100%). Without it,
  the same pair differed noticeably (see the last table).

## Results

| | INT8 | INT4+8 | INT4 |
|---|---|---|---|
| Perplexity | 3.934 | 3.904 (-0.78%) | 3.899 (-0.90%) |
| KLD vs INT8: mean / median / p99 / p99.9 | 0 | 0.0383 / 0.0058 / 0.52 / 2.38 | 0.0413 / 0.0065 / 0.56 / 2.55 |
| Top-1 same as INT8 | 100% | 93.54% | 93.27% |
| Top-5 overlap with INT8 (identical top-5 set) | 100% | 88.2% (50.7%) | 87.8% (49.4%) |
| Top-10 overlap with INT8 | 100% | 88.3% | 87.9% |
| Sampling set identical to INT8's | 100% | 63.8% | 63.1% |
| Same-token chance vs INT8 | 100% | **94.24%** | **94.01%** |

INT4 vs INT4+8 directly: KLD 0.0176, top-1 same 95.49%, same-token chance
96.10%. The two share 43 of 48 expert layers.

**Is INT4+8 better than INT4?** Paired over the 77 windows (INT4+8 minus INT4,
95% bootstrap CI):

| Measure | Difference | Verdict |
|---|---|---|
| KLD vs INT8 | -0.0030 [-0.0040, -0.0021]; lower in 69/77 windows | better |
| Share of INT4's KLD gap closed | **7.3%** [5.2%, 9.3%] | significant |
| Top-1 same as INT8 | +0.27 pp [+0.15, +0.38] | better |
| Same-token chance vs INT8 | +0.23 pp [+0.19, +0.28] | better |
| Perplexity | +0.12%; NLL CI [-0.0023, +0.0056] includes 0; lower in 49/77 windows | no difference |

By domain (perplexity change vs INT8 / KLD vs INT8 / top-1 same as INT8):

| Domain | INT8 ppl | INT4+8 | INT4 |
|---|---|---|---|
| A arXiv | 4.125 | -1.20% / 0.027 / 94.1% | -0.92% / 0.029 / 93.9% |
| B code | 2.324 | -0.29% / 0.027 / 95.5% | -0.14% / 0.029 / 95.3% |
| C wiki-en | 5.326 | -0.36% / 0.039 / 92.7% | -0.37% / 0.042 / 92.4% |
| D multilingual | 4.964 | -0.10% / 0.025 / 93.6% | +0.05% / 0.029 / 92.8% |
| E GitHub issues | 6.118 | -2.55% / 0.112 / 88.5% | -4.91% / 0.120 / 88.3% |

## Reading it

- **Every fidelity measure agrees.** KLD, top-1, top-k and sampling agreement
  all put INT4+8 measurably closer to INT8 than INT4, but by a small amount:
  about 7% of INT4's KLD gap, or about 0.25 pp of agreement.
- **Perplexity is not a fidelity measure here.** Both quantized variants score
  slightly *lower* perplexity than INT8, mostly on GitHub issues (logs and stack
  traces). That means more confidence on repetitive text, not closeness to the
  unquantized model. Use KLD and agreement to judge fidelity.
- **GitHub issues is the most sensitive domain:** KLD about 4x the others, and
  top-1 agreement of 88%.
- **The model is very sensitive to any perturbation**, because tiny numeric
  changes flip MoE routing at sensitive positions:

  | What differs | KLD |
  |---|---|
  | Rounding-level MoE order (before patch 0015) | 0.0133 |
  | 5 layers INT4 vs INT8 | 0.0176 |
  | 43 layers | 0.0383 |
  | 48 layers | 0.0413 |

  KLD grows far less than linearly with how much of the model is perturbed.

## Caveats

- **INT8 stands in for BF16.** A BF16-expert model (about 240 GiB of weights)
  doesn't fit on 2x64 GB + 40 GB (about 156 GiB usable). INT8 is about 9x closer to BF16 than INT4
  per expert layer. Given the sensitivity above, INT8's own KLD against BF16 is
  probably at least about 0.013, and it is not measured.
- **INT4+8 gets a small edge from the reference.** In its 5 promoted layers it
  uses the same INT8 method as the reference, so those layers score near zero
  error. Against BF16 that edge shrinks by roughly 1% of those layers'
  contribution.
- **KLD is a lower bound** (top-100 + rest bucket).

## Before the determinism fix

The same eval run before patch 0015, where scoring INT8 twice did *not*
reproduce:

| Pair (pre-fix run) | Top-1 same | Top-5 overlap | Sampling set identical | Same-token chance | KLD |
|---|---|---|---|---|---|
| INT8 run B vs INT8 run A (noise only) | 96.37% | 92.91% | 72.95% | 96.77% | 0.0133 |
| INT4+8 vs INT8 run A | 93.58% | 88.19% | 63.84% | 94.27% | 0.0378 |
| INT4 vs INT8 run A | 93.24% | 87.71% | 63.13% | 94.00% | 0.0415 |

Nondeterminism alone changed the sampled token with probability 3.2%, against
5.7-6.0% for INT4 quantization. It widened the spread but hardly moved the
averages (compare with the results table).

## Re-run on the serving stack, 2026-10-02 (`../batch-invariance/`)

The same corpus, metrics and sampler, re-run on prod's own serving stack: PP=3 19/20/9, MTP k=3, prefix caching, async
scheduling and max-num-seqs 8. The stack carries the batch-invariance fixes plus the instance-determinism switches. Every
config ran in its own freshly compiled container, and **both sides use the original BF16 PLE table** (the table above used
FP8 PLE on both sides).

| Pair | KLD mean [95% CI] | Top-1 same | Sampling set identical | Same-token chance | ppl ref / test |
|---|---|---|---|---|---|
| INT8 vs INT8, two fresh containers (noise floor) | **0** (bit-identical) | 100% | 100% | 100% | 3.927 / 3.927 |
| INT4 vs INT8 (both BF16 PLE) | 0.0404 [0.0338, 0.0479] | 93.41% | 63.4% | 94.09% | 3.927 / 3.906 |
| INT8 + FP8 PLE vs INT8 + BF16 PLE | 0.0163 [0.0132, 0.0200] | 95.98% | 71.4% | 96.38% | 3.927 / 3.931 |

* INT4 vs INT8 agrees with the table above (0.0413 / 93.27% / 94.01%).
* FP8 vs BF16 PLE is a real shift, but it is the size of any 1-ulp-class seed on this model: two separately compiled
  containers *without* the determinism switches already differ by 0.0140. Perplexity is unchanged.

## Gotcha: custom checkpoint layouts with FP8 PLE offload

A per-layer checkpoint layout (hardlinked PLE shards plus new `base-*` /
`experts-*` files) failed to boot with
`ValueError: FP8 PLE offload checkpoint is missing its scale`, even though the
scale tensor was present and indexed. Here's why:

1. The loader reads files in natural sort order.
2. `AutoWeightsLoader` calls a submodule's `load_weights` once per
   **contiguous run** of that module's tensors.
3. The GPU-side `Qwen4ExpNGramEmbedding.load_weights` raises on any run that
   lacks `ngram_embedding.weight_scale`.

The fix is to keep every `*.ple.ple_embedding.*` tensor contiguous in load
order: the scale, `layer_multipliers`, `ngram_heads_*` and the shards. For
example, put the non-shard ones in a file whose name sorts right before the
PLE-only shards. This was verified by booting a truncated real-weight
checkpoint laid out that way.
