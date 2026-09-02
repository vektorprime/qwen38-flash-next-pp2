"""Repack AWQ checkpoint PLE shards bf16 -> fp8 e4m3 with per-tensor scale.

Copies the checkpoint dir (hardlink unchanged files; rewrite the files that
contain PLE shards), converting each
`*.ple.ple_embedding.ngram_embedding.shard_<i>.weight` BF16 -> F8_E4M3, and
emitting ONE global `...ngram_embedding.weight_scale` (bf16, shape [1]) =
amax/448 across all shards — matching the Qwen FP8 PLE layout the loader
expects (per-tensor scale retained as `_offload_weight_scale`).
"""
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SNAP = glob.glob("/hf/hub/models--cyankiwi--Qwen3.8-Flash-Next-AWQ-INT4/snapshots/*")[0]
OUT = "/hf/ple-fp8-checkpoint"
os.makedirs(OUT, exist_ok=True)
FP8_MAX = 448.0  # e4m3 max

idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))
wm = idx["weight_map"]
is_shard = lambda k: ".shard_" in k and "ngram_embedding" in k and k.endswith(".weight")
ple_keys = sorted(k for k in wm if is_shard(k))
ple_files = sorted(set(wm[k] for k in wm if is_shard(k)))
print(f"PLE shard tensors: {len(ple_keys)} in {len(ple_files)} files")

# Pass 1: global amax over all shards (streamed per file).
amax = 0.0
for fname in ple_files:
    tensors = load_file(os.path.join(SNAP, fname))
    for k, t in tensors.items():
        if is_shard(k):
            amax = max(amax, t.abs().max().item())
    del tensors
print(f"global amax = {amax:.3f}")
scale = torch.tensor([amax / FP8_MAX], dtype=torch.bfloat16)

# Pass 2: rewrite PLE files with fp8 shards + add scale key to the FIRST file.
new_index = {"metadata": idx.get("metadata", {}), "weight_map": dict(wm)}
scale_key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
first_ple_file = ple_files[0]

for fname in os.listdir(SNAP):
    src = os.path.join(SNAP, fname)
    dst = os.path.join(OUT, fname)
    if fname in ple_files:
        tensors = load_file(src)
        out = {}
        for k, t in tensors.items():
            if is_shard(k):
                out[k] = (t.float() / scale.item()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            else:
                out[k] = t
        if fname == first_ple_file:
            out[scale_key] = scale
            new_index["weight_map"][scale_key] = fname
        save_file(out, dst, metadata={"format": "pt"})
        del tensors, out
    else:
        # hardlink non-PLE files (index.json, config, etc. copied later)
        if fname.endswith(".safetensors"):
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"done {fname}", flush=True)

with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump(new_index, f)
print("repack complete ->", OUT)
print(f"scale = {scale.item():.6f}")
