# Deterministic MoE token order (patch 0015)

Identical requests returned different logprobs on this stack. The cause is the
order in which `moe_align_block_size` sorts tokens into MoE expert blocks. This
patch fixes that order, and every run is now bit-reproducible: prompt logprobs,
greedy output and MTP acceptance. Validated on 2x/3x CMP 170HX, 2026-09-29.

## Symptom

Same token-id prompt, `temperature 0`, one request at a time, prefix caching
off, same server:

- The INT8 reference scored twice over 77 x 2048-token windows (see
  `../kld-eval/`):
  - The top-1 (argmax) token differed at **3.6%** of positions.
  - **0 of 77** windows were identical, and only **0.6%** of positions were
    bit-identical.
  - KLD between the two runs was **0.0133**, a third of the INT4 quantization
    signal being measured.
- Production (PP=2, MTP, async, prefix caching) behaved the same way.
  - Prompts of **16 tokens or fewer** were bit-identical.
  - From **17 tokens**, repeats differed from row 8 onward.
  - From **24 tokens**, every row differed, including position 1, which depends
    only on token 0.

## Root cause

`ops.moe_align_block_size` (csrc) writes each token slot into its expert's
segment of `sorted_token_ids` using GPU atomics. The order inside a segment,
and so which `block_size`-row block a token lands in, therefore changes from
call to call once a batch has more than about 16 tokens.

Marlin WNA16 MoE output depends on that placement at the rounding level. A
rounding-level difference then flips top-10 expert routing in later layers,
which amplifies it to 0.1-10 nats at sensitive positions.

How it was pinned down, with production left untouched (throwaway replicas on
spare GPUs):

| Test | Result |
|---|---|
| Raw kernel, 30 calls on identical `topk_ids` (512 experts, top-10) | distinct orders: `1/30` at <=16 tokens, `3-5/30` at 17-24, `30/30` at >=64 |
| Replica, same image, 8 layers, **dummy** weights, PP=1 and PP=2 | deterministic (dummy weights don't expose it) |
| Replica, 8 layers, **real** weights, 1 GPU, no PP / MTP / async | nondeterministic, differences up to 0.6 nats |
| Same, `--enforce-eager` | still nondeterministic, so not torch.compile or CUDA graphs |
| Forward-hook SHA-1 of every module's inputs and outputs | the first module whose output differs has **bit-identical inputs**: the routed experts (`MoERunner`). Router output is identical; `sorted_token_ids` differ on every run. Everything upstream is identical (GDN, QSA attention, FP8 PLE layer, shared expert, router). |
| Hook that sorts each expert segment | bit-identical at every length |

Ruled out:

- the FP8 PLE table (the PLE layer output is bit-identical);
- pipeline-parallel transport (PP=1 reproduces the problem);
- torch.compile and CUDA graphs;
- the attention and GDN kernels;
- Marlin `use_atomic_add` (hard-coded `False` in this build);
- fp16 storage in the scoring harness.

`VLLM_BATCH_INVARIANT=1` is **not** an option here. The server fails to boot
with `Only symmetric quantization is supported for MoE`, because the AWQ experts
are asymmetric INT4.

## Fix

The patch changes `vllm/model_executor/layers/fused_moe/moe_align_block_size.py`.

- **What it does:** right after the csrc kernel, one Triton launch (one program
  per expert) binary-searches that expert's segment start in `expert_ids`. It
  then streams the flattened `topk_ids` in order and rewrites the segment with
  the expert's slot ids in ascending order.
- **What stays the same:** segments, padding, `expert_ids` and
  `num_tokens_post_pad`.
- **Properties:** no atomics and no host sync, so it is safe under CUDA-graph
  capture. Expert-parallel layouts (local expert ids) use a layout-agnostic
  torch fallback.
- **Coverage:** it covers every caller (Marlin, Triton `fused_moe`, the MTP
  drafter).
- **Toggle:** `VLLM_MOE_DETERMINISTIC_ALIGN=0` restores the old behavior.

Cost, measured inside CUDA graphs on the slowest card (CMP 170HX capped at
1410 MHz):

| Batch | Extra per MoE layer | Per 48-layer forward |
|---|---|---|
| decode (<=64 tokens) | +4-6 us | ~0.3 ms |
| 2048-token prefill | +48 us | ~2.3 ms |

End-to-end single-stream decode on production (`bench.py`) was 96.7 / 97.2
tok/s. Earlier runs of the same configuration without the patch gave 91.7 /
101.0, so the difference is within run-to-run spread.

## Files and build

| File | What it is |
|---|---|
| `0015-deterministic-moe-align.diff` | Unified diff against `moe_align_block_size.py` in PR #53899 @ `a5530b9` (identical in all `pp2*` and `ple-fp8*` images) |
| `apply_patch.py` | `apply_patch.py <orig.py> <patched.py>`. Produces the patched file and asserts every anchor. |
| `Dockerfile` | `FROM ${BASE}` + COPY + `py_compile` + grep guards |
| `unit_test.py` | Checks raw vs patched determinism (1-8192 tokens), validity (same slots per segment, ascending, padding last), Triton == torch fallback, and CUDA-graph capture/replay. Prints `FINAL PASS`. |
| `graph_timing.py` | Overhead inside a CUDA graph (48 calls = one forward) |

```bash
IMG=qwen38-flash-next:ple-fp8-ring          # or any pp2*/ple-fp8* image
docker run --rm --entrypoint cat $IMG \
  /opt/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py > moe_align_block_size.py.orig
python3 apply_patch.py moe_align_block_size.py.orig moe_align_block_size.py
docker build --build-arg BASE=$IMG -t $IMG-detmoe .
docker run --rm --runtime nvidia --gpus '"device=0"' --entrypoint python3 \
  -v $PWD:/w $IMG-detmoe /w/unit_test.py        # -> FINAL PASS
```

No launch-flag changes are needed; the patch is on by default.

## Validation (2026-09-29)

| Where | Result |
|---|---|
| Unit test | patched: 1 distinct order in 30 calls at every size (1-8192 tokens); raw: 30/30 at >=64 |
| 8-layer real-weight replica, compiled + CUDA graphs | bit-identical at 8-2048 tokens, 2 windows x 5 repeats |
| Control: same image, `VLLM_MOE_DETERMINISTIC_ALIGN=0` | nondeterminism returns. Patched outputs sit inside the normal spread: median absolute logprob difference patched vs unpatched is 0.0097-0.0108 against 0.0076-0.0093 unpatched vs unpatched at 2048 tokens. So the patch fixes one valid ordering and adds no bias. |
| Full model, PP=3 eval server (INT8 reference) | two full scoring passes (77 x 2048) bit-identical in token ids, logprobs and actual-token logprobs |
| Production PP=2 + MTP k=3 + async + prefix caching + `align` | bit-identical at 16-2048 tokens x 5 repeats. Greedy `bench.py` gave 16/16 identical outputs across two runs with the same MTP acceptance (71.7%). |

Production was recreated from `docker inspect` of the previous container with
only the image changed (`ple-fp8-ring` -> `ple-fp8-ring-detmoe`). The old
container is kept stopped as the rollback.
