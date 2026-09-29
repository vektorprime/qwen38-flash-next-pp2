ARG BASE
FROM ${BASE}
COPY moe_align_block_size.py /opt/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py
RUN python3 -m py_compile /opt/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py \
 && grep -q 'VLLM_MOE_DETERMINISTIC_ALIGN", "1"' /opt/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py \
 && grep -q 'sorted_ids = _canonicalize_sorted_ids(' /opt/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py
