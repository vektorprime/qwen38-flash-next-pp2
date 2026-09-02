# FP8 PLE table (halve host RAM: 97 -> 48 GiB)

Quantizes the 95.43 GiB BF16 PLE n-gram table to FP8 e4m3 with a single
per-tensor scale, in the exact shard layout Qwen's official FP8 checkpoint
uses (`ngram_embedding.shard_<0..511>.weight` fp8 + one `weight_scale`).
No GPU FP8 hardware needed — the table lives in host RAM; dequant happens
on the CPU offload worker before rows cross PCIe (they arrive bf16).

## Files
- `repack_ple_fp8.py` — run inside a container with the HF cache mounted at
  /hf. Copies the checkpoint to `/hf/ple-fp8-checkpoint` (hardlinks unchanged
  files, rewrites the 22 PLE-bearing files, replaces 16 HF snapshot symlinks
  with real copies). Emits one global bf16 `weight_scale` (amax/448).
- `apply_gate.py` — patches `_get_ple_embedding_quant_method` in
  `vllm/models/qwen4_exp/nvidia/ple_layer.py` (nvidia only; the amd variant
  has no FP8 PLE support) to return the FP8 method when
  `VLLM_PLE_FP8_TABLE=1`, regardless of the outer quant config (AWQ).

## Apply to an existing pp2-p0 image
```bash
docker create --name pleq2 --entrypoint sleep qwen38-flash-next:pp2-p0 14400
docker start pleq2
docker cp repack_ple_fp8.py pleq2:/tmp/
docker exec -d pleq2 sh -c "python3 /tmp/repack_ple_fp8.py > /tmp/repack.log 2>&1"
# wait for "repack complete" in /tmp/repack.log (~15 min)
docker cp apply_gate.py pleq2:/tmp/
docker exec pleq2 python3 /tmp/apply_gate.py
docker commit --change 'ENTRYPOINT ["vllm"]' --change 'CMD []' pleq2 qwen38-flash-next:ple-fp8
docker rm -f pleq2
```

## Launch
Same as pp2-p0 but: model path = `/root/.cache/huggingface/ple-fp8-checkpoint`,
plus `--env VLLM_PLE_FP8_TABLE=1`.

## Results (2x CMP 170HX, PP=2, 2026-09-02)
- Host RAM: container RSS 59.15 GiB (vs ~107 GiB bf16-PLE run); PLE worker
  table 48.4 GiB vs 96.8.
- Boot: clean; `PLE offload matched 132 checkpoint tensor(s) ... 2/2
  materialized parameter(s)`.
- Quality: `7*8=56` correct; shared-doc codeword recalls intact (content
  matches the document, not another conversation).
- Note: cross-request state bleed (see README Part 5) is UNCHANGED — that is
  a separate concurrency bug in the V2/PP path, not a PLE-dtype issue.

## Gotchas hit during bring-up
- `shutil` import missing on first repack run (partially-written dir — the
  script is resumable: delete /hf/ple-fp8-checkpoint and rerun).
- HF snapshot symlinks (`../../blobs/...`) don't resolve relative to the
  copied dir — the repack script now real-copies the 16 symlinked non-PLE
  shards (in an earlier run these arrived as 0-byte files: "header too
  small" / "incomplete metadata" errors at load).
- The docker commit of a container created with a custom --entrypoint must
  reset `ENTRYPOINT ["vllm"]` (same trap as documented in findings.md).
- Disk: needs ~52 GiB free in the HF volume for the repacked dir.
