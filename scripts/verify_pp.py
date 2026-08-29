import re
p1 = open('/opt/vllm/vllm/models/qwen4_exp/nvidia/model_state.py').read()
p2 = open('/opt/vllm/vllm/v1/ple_offload/connector.py').read()
print('model_state has is_first_pp:', 'is_first_pp' in p1)
print('model_state still has PP raise:', 'pipeline_parallel_size=1 because non-first' in p1)
print('connector has is_first_pp:', 'is_first_pp' in p2)
print('connector has get_pp_group:', 'get_pp_group' in p2)
import vllm.models.qwen4_exp.nvidia.model_state as m
import inspect
src = inspect.getsource(m.Qwen4ExpModelState.__init__)
print('---- init source ----')
print(src[:1500])
