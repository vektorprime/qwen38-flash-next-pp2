# qwen38-flash-next-pp2

Serve **`cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`** on 2 GPUs with
**`--pipeline-parallel-size 2`** and **PLE CPU offload** — a combination
upstream vLLM (including PR #53899, which ships the model + PLE-offload
support) refuses at six separate places. This repo carries the build recipe,
the pipeline-parallel patch, and the exact launch flags, verified working on
2x NVIDIA CMP 170HX 64G.

Verified on 2026-08-29:

| | |
|---|---|
| vLLM | `0.1.dev1+ga5530b90c.d20260829` — PR #53899 @ `a5530b9` + 6-site PP patch |
| Context | `max-model-len` 262,144 (auto), KV cache 1,196,679 tokens, 4.56x concurrency |
| Weights per GPU | PP0 39.48 GiB, PP1 45.53 GiB (of 64 GiB cards) |
| PLE | 95 GiB ngram/PLE table offloaded to host RAM (`layers.1.ple`) |
| API | OpenAI-compatible on `:8001` — reasoning + tool-call parsers live |

**Part 1** gets you running in three commands. **Part 2** details every patch:
what upstream guards exist, why each is removable, and the exact wall you'll
hit if something moves under us.

---

# Part 1 — Quick Start

## Repo layout

| File | What it is |
|---|---|
| `Dockerfile` | Builds `qwen38-flash-next:pp2` from `vllm/vllm-openai:nightly` + PR #53899 + the PP patch |
| `patch_pp.py` | The 6-site pipeline-parallel patch (runs at build time; prints one line per site) |
| `scripts/launch.sh` | Verified `docker run` wrapper (token file as `$1`, or `HT` env var) |
| `scripts/verify_pp.py` | Post-build check that the PP patches landed in the image |

## Prerequisites

- Docker with the NVIDIA container toolkit (`--runtime nvidia` works)
- 2 GPUs with >= ~48 GiB each (weights split ~39.5 / ~45.5 GiB + KV headroom)
- >= ~110 GiB free host RAM for the PLE offload (we ran with 178 GiB)
- ~250 GiB free disk for the HF cache (~174 GB, 38 safetensors shards)
- A HuggingFace token with access to `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`

## 1. Build

```bash
docker build -t qwen38-flash-next-pp2:latest -f Dockerfile .
```

(Or keep the original tag `qwen38-flash-next:pp2` and pass `IMAGE=...` to the
launch script.) After the base `nightly` image is cached, the build takes ~2
minutes. **The output must end with these 11 lines:**

```
patched vllm/models/qwen4_exp/nvidia/model_state.py
patched vllm/models/qwen4_exp/amd/model_state.py
patched model_runner setup_ple_offload
patched mtp.py draft-branch vllm/models/qwen4_exp/nvidia/mtp.py
patched mtp.py draft-branch vllm/models/qwen4_exp/amd/mtp.py
patched gpu_worker validate
patched model.py HC skip vllm/models/qwen4_exp/nvidia/model.py
patched model.py HC skip vllm/models/qwen4_exp/amd/model.py
patched connector init
patched connector setup
patched connector launch
all patched
```

Any `WARNING ... pattern not found` line means the PR source moved and the
image is **not** patched correctly — do not run it; re-derive the pattern (see
Part 2).

## 2. Launch

```bash
bash scripts/launch.sh /path/to/hf_token.txt
```

The script takes the token from a file (or from `HT=...` in the env) and never
stores it. Overridable via env: `GPU_IDS` (default `0,1`), `PORT` (default
`8001`), `IMAGE` (default `qwen38-flash-next:pp2`). The exact flags it runs
equivalent to:

<details>
<summary>Full docker run command</summary>

```bash
docker run --runtime nvidia -d --gpus '"device=0,1"' \
  --ipc=host --shm-size=96g \
  --cap-add=SYS_PTRACE \
  --name qwen38-flash-next --restart always -p 8001:8000 \
  -v vllm-hf-cache:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --env "CUDA_DEVICE_ORDER=PCI_BUS_ID" \
  --env "HUGGING_FACE_HUB_TOKEN=***" \
  --env "VLLM_PLE_CPU_OFFLOAD=1" \
  --env "VLLM_GDN_DECODE_KERNEL=triton" \
  qwen38-flash-next:pp2 \
  serve cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4 \
  --served-model-name qwen38-flash-next-awq \
  --max-model-len auto \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8 \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --async-scheduling \
  --generation-config auto \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0,"presence_penalty":0.0}'
```
</details>

## 3. Verify

```bash
docker logs -f qwen38-flash-next
```

Cold start takes ~9 minutes (weights download once, then cached). Expected
milestones:

```
Resolved architecture: Qwen4ExpForConditionalGeneration
Using max model len 262144
Resolved architecture: Qwen4ExpMTP                          # spec decode
(Worker_PP0) Model loading took 39.48 GiB memory and 214.8 seconds
(Worker_PP0) PleOffload: registered 1 PleOffloadLayer(s) ... ['language_model.model.layers.1.ple.ple_embedding']
(Worker_PP0) Worker ready - 1 PleOffloadLayer(s) ...
(Worker_PP1) Model loading took 45.53 GiB memory and 242.7 seconds   # no PLE lines on PP1 — correct
(Worker_PP0) Initial profiling/warmup run took 86.92 s
(Worker_PP0) Available KV cache memory: 14.97 GiB
(EngineCore) GPU KV cache size: 1,196,679 tokens, Maximum concurrency for 262,144 tokens per request: 4.56x
(APIServer) Application startup complete.
```

Then smoke test:

```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen38-flash-next-awq","messages":[{"role":"user","content":"What is 7*8? Answer with just the number."}],"max_tokens":64}'
```

Expected: `"content":"\n\n56"` with a `reasoning` field and
`system_fingerprint` containing `pp2`.

### If it fails

- **Build prints a `WARNING`** — the PR source moved; fix patterns per Part 2.
- **`WorkerProc failed to start`** — find the error line in the wall map
  (Part 2); each of the six upstream guards is one recognizable message.
- **`No CUDA GPUs are available`** while host `nvidia-smi` is healthy —
  usually a crash-loop (`--restart always`) with a GPU still being released by
  the dying attempt: `docker rm -f`, confirm the cards read 0 MiB, relaunch.
- **OOM-ish "free 5 MB while mapping 20 MB"** — check you used
  `PYTORCH_CUDA_ALLOC_CONF` (not `PYTORCH_ALLOC_CONF`; the latter is silently
  ignored and fragmentation looks like OOM). See Part 2, Pitfalls.

---

# Part 2 — Patch Details (Reference)

## Why PP=2 + PLE is blocked upstream

The model is a hybrid: 512-expert AWQ-INT4 MoE (~85 GiB GPU-resident across
two stages) plus a **95 GiB bf16 PLE (per-layer embedding / ngram) table** and
an **MTP speculative-decoding module**. The PLE CPU-offload path shipped in
vLLM PR #53899 (which includes #53896) only supports
`pipeline_parallel_size=1`: the model is too big for PP=1 on any realistic
GPU pair, and PLE offload is the only way to fit the 95 GiB table. With TP=2
ruled out (the CMP cards sit on slow PCIe lanes and TP all-reduce dies
there), PP=2 + PLE-offload is the only viable shape — so upstream's "please
run with PP=1" is a non-starter.

The key insight: **PLE exists on exactly one layer** (`ple_layer_ids` ->
`language_model.model.layers.1.ple.ple_embedding`), which always lands on
**PP rank 0**. PP rank 1 never needs PLE, never allocates its buffers, and the
MTP drafter runs standalone on the last rank. The PP restriction is therefore
a series of conservative guards and rank-assumption bugs, not an
architectural impossibility. We removed/gated six of them (~90 lines across 5
files, in both `nvidia/` and `amd/` model variants).

## Wall map — errors in the order they were hit

All seven rows are the same bug class: "PLE/MTP/HC sub-modules assume PP=1"
(row 7 is not PP-related but looks like it is). Each fix advanced startup one
stage further.

| # | Error (file) | Meaning | Fix site |
|---|---|---|---|
| 1 | `RuntimeError: N-gram PLE embedding currently requires pipeline_parallel_size=1` (model_state.py) | Explicit upstream refusal | Site 1 |
| 2 | `ValueError: VLLM_PLE_CPU_OFFLOAD does not support ... Unsupported settings: PP=2` (gpu_worker.py) | Worker-side config whitelist | Site 2 |
| 3 | `RuntimeError: ... but the model has no PleOffloadLayer` (ple_offload/connector.py) | Connector spawned on PP1 which owns no PLE | Site 3 |
| 4 | `no module or parameter named 'hyper_connection_mixer' in Qwen4ExpModel` (model.py weight map) | Final HC mixer exists only on last rank but weights mapped to all | Site 5 |
| 5 | `RuntimeError: PLE offload requires a query_start_loc source` (model_runner.py) | Runner expects PLE tensors that only exist on PP0 | Site 4 |
| 6 | `torch._dynamo.exc.Unsupported: Data-dependent assertion ... assert intermediate_tensors is not None` (mtp.py:298) | MTP draft input branch keyed on target PP rank, not call payload | Site 6 |
| 7 | `expandable_segments: memory mapping failed with OOM while trying to map 20971520 bytes` | `PYTORCH_ALLOC_CONF` typo-env; allocator ran non-expandable and weight-load fragmentation killed a worker at ~41/63.5 GiB with 22 GiB "free" | Correct env var name (launch) |

## The six patch sites

Every replacement is an **exact-string match** against the PR source at
`a5530b9` with a `WARNING` printed if the pattern isn't found, so a silent
no-op can't slip past you. Where a patch applies to `nvidia/` and `amd/`
model variants, both files are patched.

### Site 1 — `vllm/models/qwen4_exp/{nvidia,amd}/model_state.py`: gate PLE to first PP rank

Removes the explicit `pipeline_parallel_size=1` `RuntimeError` and instead
sets `uses_ngram_embedding` false on non-first ranks (they early-return and
skip the PLE buffer allocation entirely).

```python
# BEFORE
self.uses_ngram_embedding = bool(config.ple_layer_ids)
if not self.uses_ngram_embedding:
    self.ngram_context_len = 0
    self.ngram_eos_token_id = 0
    return

if vllm_config.parallel_config.pipeline_parallel_size > 1:
    raise RuntimeError(
        "N-gram PLE embedding currently requires "
        "pipeline_parallel_size=1 because non-first pipeline ranks do "
        "not receive the raw input_ids required by PLE. Please run "
        "with PP=1."
    )

# AFTER
try:
    is_first_pp = get_pp_group().is_first_rank
except Exception:
    is_first_pp = True
self.uses_ngram_embedding = bool(config.ple_layer_ids) and is_first_pp
if not self.uses_ngram_embedding:
    self.ngram_context_len = 0
    self.ngram_eos_token_id = 0
    return
```

Also adds `from vllm.distributed.parallel_state import get_pp_group` if missing.

### Site 2 — `vllm/v1/worker/gpu_worker.py`: drop the PP rejection from the offload-config validator

`_validate_ple_offload_config()` whitelists parallel settings and appends
`PP=<n>` to `unsupported` (raising `ValueError: VLLM_PLE_CPU_OFFLOAD does not
support the requested configuration`) — even though Site 1 now makes PP safe.

```python
# BEFORE
if parallel_config.pipeline_parallel_size != 1:
    unsupported.append(f"PP={parallel_config.pipeline_parallel_size}")

# AFTER
pass
```

### Site 3 — `vllm/v1/ple_offload/connector.py`: make the connector PP-aware

`PleOffloadConnector` assumes every rank has `PleOffloadLayer`s; on non-first
ranks `_setup_layers` returns nothing and it raised
`"VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"`.
Three changes: store `is_first_pp` in `__init__` and return early when a
non-first rank has no layers; make `_setup_layers` tolerate missing layers on
non-first ranks; make the CUDA-input launch path (`_launch`) a no-op on
non-first ranks so no ZMQ/CUDA-IPC/shared buffers spawn there.

```python
# in __init__
try:
    self.is_first_pp = get_pp_group().is_first_rank
except Exception:
    self.is_first_pp = True
self._layers = self._setup_layers(vllm_config, model)
if not self._layers and not self.is_first_pp:
    return

# in _setup_layers
if not layers:
    try:
        is_first = get_pp_group().is_first_rank
    except Exception:
        is_first = True
    if not is_first:
        return {}
    raise RuntimeError(
        "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
    )

# in _launch
if not getattr(self, 'is_first_pp', True):
    return
if self.tp_rank != 0:
    return
```

### Site 4 — `vllm/v1/worker/gpu/model_runner.py`: `_setup_ple_offload` is a no-op on non-first ranks

Because Site 1 skips the PLE buffers on non-first ranks,
`model_state.ple_query_start_loc` doesn't exist there, and the hard
`raise RuntimeError("PLE offload requires a query_start_loc source")` killed
PP1 after it loaded 45.5 GiB of weights.

```python
# BEFORE
if not isinstance(query_start_loc_source, torch.Tensor):
    raise RuntimeError("PLE offload requires a query_start_loc source")

# AFTER
from vllm.distributed.parallel_state import get_pp_group as _get_pp_group
try:
    _is_first_pp = _get_pp_group().is_first_rank
except Exception:
    _is_first_pp = True
if not isinstance(query_start_loc_source, torch.Tensor):
    if not _is_first_pp:
        return
    raise RuntimeError("PLE offload requires a query_start_loc source")
```

### Site 5 — `vllm/models/qwen4_exp/{nvidia,amd}/model.py`: skip `hyper_connection_mixer` weights on non-last ranks

`Qwen4ExpModel.__init__` creates the final HC mixer **only on
`is_last_rank`** (otherwise `self.hyper_connection_mixer = None`), but
`load_weights` only skipped the non-persistent
`hyper_connection_mixer.block_inject_weight`. The three persistent checkpoint
keys `model.language_model.hyper_connection_mixer.{hc_norm,
input_mix_weight_down, input_mix_weight_up}.weight` then hit an empty module
on rank 0: `no module or parameter named 'hyper_connection_mixer' in
Qwen4ExpModel`.

```python
# BEFORE
skip_substrs = [
    "hashstats_",
    "token_lookup",
    "hyper_connection_mixer.block_inject_weight",
]

# AFTER
skip_substrs = [
    "hashstats_",
    "token_lookup",
    "hyper_connection_mixer.block_inject_weight",
]
if not get_pp_group().is_last_rank:
    skip_substrs.append("hyper_connection_mixer.")
```

### Site 6 — `vllm/models/qwen4_exp/{nvidia,amd}/mtp.py`: MTP drafter input branch

The MTP draft model is **not PP-partitioned** — the speculator runs it
standalone on the last rank and passes `hidden_states` directly. But
`Qwen4ExpMTPModel.forward` branched on `get_pp_group().is_first_rank`, so on
PP1 it took the `intermediate_tensors` path and hit
`assert intermediate_tensors is not None` (mtp.py:298) during Dynamo
fullgraph capture — a *data-dependent* assert, which torch.compile treats as
a fundamental graph-break error, crashing `profile_run`. Branch on what the
caller actually passed instead (this is also correct if the draft itself were
ever PP-split):

```python
# BEFORE
if get_pp_group().is_first_rank:
    assert hidden_states is not None

# AFTER
if get_pp_group().is_first_rank or hidden_states is not None:
    assert hidden_states is not None
```

## Full patch script

Saved as `patch_pp.py` in this repo and applied by the Dockerfile
(`COPY patch_pp.py` + `RUN python3 /patch_pp.py`) before the editable install.

```python
import pathlib, re
for p in [pathlib.Path("vllm/models/qwen4_exp/nvidia/model_state.py"),
          pathlib.Path("vllm/models/qwen4_exp/amd/model_state.py")]:
    t = p.read_text()
    if "get_pp_group" not in t:
        t = t.replace("from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState",
                      "from vllm.distributed.parallel_state import get_pp_group\nfrom vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState")
    old = """        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )"""
    new = """        try:
            is_first_pp = get_pp_group().is_first_rank
        except Exception:
            is_first_pp = True
        self.uses_ngram_embedding = bool(config.ple_layer_ids) and is_first_pp
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return"""
    if old in t:
        t = t.replace(old, new)
        print("patched", p)
    else:
        print("WARNING model_state pattern not found:", p)
        t = re.sub(r"if vllm_config\.parallel_config\.pipeline_parallel_size > 1:\s+raise RuntimeError\(.*?with PP=1\.\n\s*\)",
                   "if False:", t, flags=re.S)
        t = t.replace("        self.uses_ngram_embedding = bool(config.ple_layer_ids)",
                      "        self.uses_ngram_embedding = bool(config.ple_layer_ids) and get_pp_group().is_first_rank")
    p.write_text(t)

# 5) model_runner.py — _setup_ple_offload must be a no-op on non-first PP
#    ranks: model_state has no ple_query_start_loc/ngram_context there
#    (PLE lives only in stage 0), so the "requires a source" raise is wrong
#    under PP>1. Gate on get_pp_group().is_first_rank before raising.
p = pathlib.Path("vllm/v1/worker/gpu/model_runner.py")
t = p.read_text()
old_setup = """        query_start_loc_source = getattr(self.model_state, "ple_query_start_loc", None)
        ngram_context_source = getattr(self.model_state, "ngram_context", None)
        if not isinstance(query_start_loc_source, torch.Tensor):
            raise RuntimeError("PLE offload requires a query_start_loc source")"""
new_setup = """        query_start_loc_source = getattr(self.model_state, "ple_query_start_loc", None)
        ngram_context_source = getattr(self.model_state, "ngram_context", None)
        # PP patched (qwen38-flash-next:pp2): non-first pipeline ranks own no
        # PLE layers and their model_state skips the PLE buffers entirely.
        from vllm.distributed.parallel_state import get_pp_group as _get_pp_group
        try:
            _is_first_pp = _get_pp_group().is_first_rank
        except Exception:
            _is_first_pp = True
        if not isinstance(query_start_loc_source, torch.Tensor):
            if not _is_first_pp:
                return
            raise RuntimeError("PLE offload requires a query_start_loc source")"""
if old_setup in t:
    t = t.replace(old_setup, new_setup)
    print("patched model_runner setup_ple_offload")
else:
    print("WARNING model_runner setup_ple_offload pattern not found")
p.write_text(t)

# 6) mtp.py (nvidia + amd) — PP-aware MTP draft input.
#    Under PP>1 the MTP drafter runs on the LAST rank, where the target's
#    hidden_states arrive as intermediate_tensors["hidden_states"]
#    (make_layers puts PPMissingLayer for layers 0..23, so the
#    is_first_rank branch never fires). The draft path in
#    spec_decode/autoregressive/speculator.py calls the MTP model with
#    hidden_states= and intermediate_tensors=None ->
#    "assert intermediate_tensors is not None" (fails Dynamo fullgraph
#    compile as a data-dependent assert). Accept hidden_states directly.
for p in [pathlib.Path("vllm/models/qwen4_exp/nvidia/mtp.py"),
          pathlib.Path("vllm/models/qwen4_exp/amd/mtp.py")]:
    t = p.read_text()
    old = """        if get_pp_group().is_first_rank:
            assert hidden_states is not None"""
    new = """        if get_pp_group().is_first_rank or hidden_states is not None:
            assert hidden_states is not None"""
    if old in t:
        t = t.replace(old, new)
        print("patched mtp.py draft-branch", p)
    else:
        print("WARNING mtp.py pattern not found:", p)
    p.write_text(t)

# 3) gpu_worker.py — _validate_ple_offload_config no longer rejects PP>1
#    (the PLE offload path is now PP-aware: only stage 0 holds PLE layers)
p = pathlib.Path("vllm/v1/worker/gpu_worker.py")
t = p.read_text()
old_pp = """        if parallel_config.pipeline_parallel_size != 1:
            unsupported.append(f"PP={parallel_config.pipeline_parallel_size}")"""
new_pp = """        # PP patched (qwen38-flash-next:pp2): PLE offload is PP-aware — only
        # the first pipeline stage holds PLE layers, so PP > 1 is supported.
        pass"""
if old_pp in t:
    t = t.replace(old_pp, new_pp)
    print("patched gpu_worker validate")
else:
    print("WARNING gpu_worker PP check pattern not found")
p.write_text(t)

# 4) model.py (nvidia + amd) — PP-aware skip of the 3 persistent
#    hyper_connection_mixer weights on non-last pipeline ranks.
#    Root cause (confirmed from source): Qwen4ExpModel.__init__ creates
#    self.hyper_connection_mixer only on is_last_rank (else None), but
#    load_weights only skipped 'hyper_connection_mixer.block_inject_weight'
#    (the non-persistent one). The 3 persistent keys
#    model.language_model.hyper_connection_mixer.{hc_norm,
#    input_mix_weight_down, input_mix_weight_up}.weight then hit an empty
#    module on rank 0 (PP>1) -> 'no module or parameter named
#    hyper_connection_mixer in Qwen4ExpModel'. Skip them on non-last ranks.
for p in [pathlib.Path("vllm/models/qwen4_exp/nvidia/model.py"),
          pathlib.Path("vllm/models/qwen4_exp/amd/model.py")]:
    t = p.read_text()
    old = """        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]"""
    new = """        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]
        # PP patched (qwen38-flash-next:pp2): hyper_connection_mixer exists
        # only on the final pipeline rank, so skip its persistent weights on
        # non-last ranks (they would otherwise hit an empty module).
        if not get_pp_group().is_last_rank:
            skip_substrs.append("hyper_connection_mixer.")"""
    if old in t:
        t = t.replace(old, new)
        print("patched model.py HC skip", p)
    else:
        print("WARNING model.py HC skip pattern not found:", p)
        # fallback: append to the literal block_inject_weight line
        t = t.replace('"hyper_connection_mixer.block_inject_weight",',
                      '"hyper_connection_mixer.block_inject_weight",\n        ]\n        if not get_pp_group().is_last_rank:\n            skip_substrs.append("hyper_connection_mixer.")\n        [')
    p.write_text(t)

p = pathlib.Path("vllm/v1/ple_offload/connector.py")
t = p.read_text()
if "get_pp_group" not in t:
    t = t.replace("from vllm.distributed.parallel_state import get_dp_group, get_tp_group",
                  "from vllm.distributed.parallel_state import get_dp_group, get_pp_group, get_tp_group")
old_init = "        self.dp_rank = get_dp_group().rank_in_group\n        self.tp_rank = get_tp_group().rank_in_group\n        self._layers = self._setup_layers(vllm_config, model)"
new_init = ("        self.dp_rank = get_dp_group().rank_in_group\n"
            "        self.tp_rank = get_tp_group().rank_in_group\n"
            "        try:\n"
            "            self.is_first_pp = get_pp_group().is_first_rank\n"
            "        except Exception:\n"
            "            self.is_first_pp = True\n"
            "        self._layers = self._setup_layers(vllm_config, model)\n"
            "        if not self._layers and not self.is_first_pp:\n"
            "            return")
if old_init in t:
    t = t.replace(old_init, new_init)
    print("patched connector init")
else:
    print("WARNING connector init pattern not found")
old_setup = """        if not layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )"""
new_setup = """        if not layers:
            try:
                is_first = get_pp_group().is_first_rank
            except Exception:
                is_first = True
            if not is_first:
                return {}
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )"""
if old_setup in t:
    t = t.replace(old_setup, new_setup)
    print("patched connector setup")
old_launch = "        if self.tp_rank != 0:\n            return\n\n        if self._uses_cuda_inputs:\n            assert self._input_ready_event is not None"
new_launch = "        if not getattr(self, 'is_first_pp', True):\n            return\n        if self.tp_rank != 0:\n            return\n\n        if self._uses_cuda_inputs:\n            assert self._input_ready_event is not None"
if old_launch in t:
    t = t.replace(old_launch, new_launch)
    print("patched connector launch")
else:
    print("WARNING connector launch pattern not found")
p.write_text(t)
print("all patched")
```

## Dockerfile

The base image `vllm/vllm-openai:nightly` has no `git`, so install it first.
We fetch the PR ref directly and use `VLLM_USE_PRECOMPILED=1` so only Python
changes are overlaid on the precompiled binaries.

```dockerfile
FROM vllm/vllm-openai:nightly
WORKDIR /opt/vllm
ARG PR=53899
RUN apt-get update -qq && apt-get install -y -qq git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "setuptools-scm>=8" "setuptools-rust>=1.9" cmake ninja 2>&1 | tail -5
RUN git init -q && git remote add origin https://github.com/vllm-project/vllm.git && \
    git fetch --depth 1 origin pull/${PR}/head:qwen38 && git checkout qwen38 && \
    git log --oneline -1 && ls vllm/models/qwen4_exp/nvidia/model_state.py
COPY patch_pp.py /patch_pp.py
RUN python3 /patch_pp.py
RUN VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation --no-cache-dir 2>&1 | tail -30 && python3 -c "import vllm; print(vllm.__version__)"
WORKDIR /workspace
ENTRYPOINT ["vllm"]
```

## Launch flag rationale

- `--gpus '"device=0,1"'` — the nested quoting is **required** (device list,
  not count). In a script, single-quote the whole value as in `launch.sh`.
- `VLLM_PLE_CPU_OFFLOAD=1` — the whole point: PLE table in host RAM.
- `VLLM_GDN_DECODE_KERNEL=triton` + `--mamba-cache-mode align` — required by
  this hybrid architecture's GDN/mamba state layout.
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` — the
  fork model ships MTP heads; without it vLLM wastes the drafter weights. Note
  vLLM's warning that `num_speculative_tokens>1` re-runs one MTP layer 3x and
  may lower acceptance.
- `PYTORCH_CUDA_ALLOC_CONF` — the correct name. `PYTORCH_ALLOC_CONF`
  (non-CUDA) exists only in newer torch CPU paths; **this env var being
  silently ignored is what turned mild fragmentation into an OOM crash-loop**
  on the GPU allocator.
- `--ipc=host` + `--shm-size=96g` — the PleOffload worker moves PLE lookups
  over host-IPC (`ipc:///tmp/...` ZMQ + CUDA IPC); shared memory pressure is
  real.
- `--max-model-len auto` resolves to the full **262144** context; KV profiled
  fine at util 0.9 (1.19M-token KV cache), so the earlier fear that 262k
  wouldn't fit was wrong for this model — the MoE experts shard cleanly across
  stages.

## Reproduction gotchas (operational)

- **`git` missing** from `vllm/vllm-openai:nightly` — the Dockerfile installs
  it; any rebuild without that line fails at `git init`.
- **`python` vs `python3`** in any verify line (image has only `python3`).
- **`ENTRYPOINT ["vllm"]`** swallows ad-hoc commands — use
  `docker run --rm --entrypoint python3 qwen38-flash-next:pp2 -c "..."`.
- **Don't pipe through `tail` when you need real exit codes**, and
  heredocs-over-SSH with `$(...)` command substitution get mangled — write
  scripts to a file, scp, run.
- `--restart always` + a crash-loop can leave a second container's workers
  unable to grab the GPU mid-release. If you see `No CUDA GPUs are available`
  with healthy host `nvidia-smi`, `docker rm -f` the loop, confirm the cards
  are free, then relaunch (once: a full host reboot also clears it).
- `--privileged` is **not required** — the tested configuration uses only
  `--runtime nvidia --gpus '"device=0,1"' --ipc=host`.

## What this does NOT do

- Not TP=2 — tensor parallel is deliberately avoided (slow PCIe lanes on the
  CMP cards).
- The PP=2 split is even layer-wise; the 95 GiB PLE stays resident in host RAM
  for the process lifetime and its first fill is slow (the offload worker
  streams shards on load).
- PLE correctness on non-first ranks is assumed-by-construction (layer 1 of
  48 lives on rank 0 only). If upstream ever ships `ple_layer_ids` on a later
  layer that crosses the stage boundary, revisit Site 1/4 gating.
- Not benchmarked: tokens/sec, acceptance rate of the 3-token MTP under PP,
  and PLE-hit latency on host RAM were not measured in this session.
- No claim this is upstreamable as-is; it's a local fork patch validated
  against commit `a5530b9`. If you rebuild on a newer `pull/53899` head, the
  `WARNING` lines in `patch_pp.py` are your tripwire.

## Source references

- Model: https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4
- PLE-offload PR: https://github.com/vllm-project/vllm/pull/53899 (includes
  https://github.com/vllm-project/vllm/pull/53896)
- vLLM version in image: `0.1.dev1+ga5530b90c.d20260829.precompiled`
  (editable, PR ref `qwen38`, HEAD
  `a5530b90cab09b187463396a99612a486ba91d6f`, commit subject "fix dummy load
  for PLE-Offload")
- Graph-break explanation cited during debugging:
  https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0034.html

---
Written 2026-08-29 from the live session; all quoted log lines, byte counts,
SHAs, and the smoke-test response were captured directly from the deployment
host.
