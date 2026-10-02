# Prod image with the batch-invariance fixes (batchinv/hook3) baked in. Same base image as prod; the hook is a sitecustomize
# module on PYTHONPATH that patches vLLM modules at import time (see hook/sitecustomize.py). Build context: this directory.
# TRITON_CACHE_DIR on the compile-cache volume so JIT-compiled Triton kernels survive restarts (first request after a boot
# otherwise spends ~10 s compiling).
FROM qwen38-flash-next:ple-fp8-pp3-detmoe
COPY hook/sitecustomize.py hook/inv_gemm.py hook/inv_marlin.py hook/inv_qsa_select.py hook/tuned_gemm.json /opt/plefp8_hook/
ENV PYTHONPATH=/opt/plefp8_hook \
    TRITON_CACHE_DIR=/root/.cache/vllm/triton_cache \
    PLEFP8_INV_MOE=1 \
    PLEFP8_INV_GEMM=1 \
    PLEFP8_INV_QSA=1 \
    PLEFP8_ALIGN_CHUNKS=64 \
    PLEFP8_MARLIN_CFG=64,128,2,128,2
