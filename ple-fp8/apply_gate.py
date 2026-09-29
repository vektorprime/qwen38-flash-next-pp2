"""Apply the FP8-PLE env gate to the ple-fp8 image.

Patch: _get_ple_embedding_quant_method in
/opt/vllm/vllm/models/qwen4_exp/{nvidia,amd}/ple_layer.py

The FP8 method requires quant_config to be Fp8Config. Our checkpoint is AWQ
(compressed-tensors). Add an env override VLLM_PLE_FP8_TABLE=1 that returns
the FP8 method for ANY quant config, since our repacked checkpoint now ships
fp8 PLE shards + weight_scale in the exact Qwen-FP8 layout.
"""
import pathlib

for variant in ("nvidia", "amd"):
    p = pathlib.Path(f"/opt/vllm/vllm/models/qwen4_exp/{variant}/ple_layer.py")
    src = p.read_text()
    if "VLLM_PLE_FP8_TABLE" in src:
        print(f"{variant}: already patched")
        continue
    old = """    if isinstance(quant_config, Fp8Config):
        if not quant_config.is_checkpoint_fp8_serialized:
            return None"""
    new = """    # PATCH (ple-fp8): allow an FP8-serialized PLE table inside a non-FP8
    # (e.g. AWQ) checkpoint. The repacked checkpoint ships fp8 shards + a
    # single ngram_embedding.weight_scale in the exact Qwen-FP8 layout, so
    # force the FP8 embedding method regardless of the outer quant config.
    import os as _os
    if _os.getenv("VLLM_PLE_FP8_TABLE") == "1":
        logger.info_once(
            "VLLM_PLE_FP8_TABLE=1: using FP8 PLE embedding method for %s", prefix
        )
        return Qwen4ExpPLEFp8EmbeddingMethod()

    if isinstance(quant_config, Fp8Config):
        if not quant_config.is_checkpoint_fp8_serialized:
            return None"""
    assert src.count(old) == 1, f"{variant}: anchor not found"
    p.write_text(src.replace(old, new, 1))
    import ast
    ast.parse(p.read_text())
    print(f"{variant}: patched, syntax OK")
