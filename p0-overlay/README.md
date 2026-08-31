# p0-overlay: files applied to the image by `docker cp` + `docker commit`

These files are NOT applied by patch_p0.py (whole-file replacements / large
overlays). Apply them after building qwen38-flash-next:pp2 to reproduce
qwen38-flash-next:pp2-p0:

```bash
docker create --name tmp-overlay qwen38-flash-next:pp2 bash
docker cp p0-overlay/pp_utils.py      tmp-overlay:/opt/vllm/vllm/v1/worker/gpu/pp_utils.py
docker cp p0-overlay/model_runner.py  tmp-overlay:/opt/vllm/vllm/v1/worker/gpu/model_runner.py
docker cp p0-overlay/qwen4_exp_nvidia_mtp.py tmp-overlay:/opt/vllm/vllm/models/qwen4_exp/nvidia/mtp.py
docker cp p0-overlay/qwen4_exp_nvidia_mtp.py tmp-overlay:/opt/vllm/vllm/models/qwen4_exp/amd/mtp.py
docker commit tmp-overlay qwen38-flash-next:pp2-p0 && docker rm tmp-overlay
```

## What each file is

- **pp_utils.py** — from zebgop-ops/qwen38-flashnext-pp `ported-files/` (vllm#46994 MTP relay). Adds `broadcast_draft` so the 3rd PP collective for draft tokens exists; without it `mtp` + `PP=2` deadlocks in warmup (`NCCL BROADCAST` 600s timeout at pp_utils.py:146).
- **model_runner.py** — from zebgop-ops `ported-files/v2_model_runner.py`. Adds `relay_draft_tokens` + `get_prev_sampled_outputs` scatter so PP0 sees verified tokens.
- **qwen4_exp/{nvidia,amd}/mtp.py** — MTP entry gate widened to `is_first_rank or is_last_rank or hidden_states is not None` (3 occurrences); PP-last rank enters the MTP block directly instead of the `intermediate_tensors` path.

## Verified behavior (2x CMP 170HX, PP=2, AWQ-INT4)

- MTP k=1: 75.4-75.8 tok/s steady single-stream (vs 58.1-58.6 no-MTP, +29%).
- MTP k=3: 64-65 tok/s (+11%) — positions 2/3 acceptance decay makes extra drafts net-negative. **Use k=1.**
- mtp.py `is_last_rank` change did NOT alter acceptance rates (still 0.61/0.40/0.22) — kept as harmless parity with zebgop pp3 image.
