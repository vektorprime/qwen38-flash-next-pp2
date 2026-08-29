import pathlib
checks=[]
def check(desc, ok):
    checks.append((desc, ok))
    print(f"{'OK' if ok else 'FAIL'} {desc}")

p1 = open('/opt/vllm/vllm/models/qwen4_exp/nvidia/model_state.py').read()
check('model_state is_first_pp', 'is_first_pp' in p1)
check('model_state no PP raise', 'pipeline_parallel_size=1 because non-first' not in p1)
p2 = open('/opt/vllm/vllm/v1/ple_offload/connector.py').read()
check('connector is_first_pp', 'is_first_pp' in p2)
check('connector _staged handshake', '_staged' in p2)
for f, needle in [
    ('/opt/vllm/vllm/v1/worker/mamba_utils.py', 'bt_row_idx = req_idx'),
    ('/opt/vllm/vllm/v1/worker/gpu/block_table.py', 'VLLM_BT_POOL'),
    ('/opt/vllm/vllm/v1/core/single_type_kv_cache_manager.py', 'if drop_eagle_block:'),
    ('/opt/vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py', 'mamba_bs ='),
    ('/opt/vllm/vllm/models/qwen4_exp/nvidia/ple_layer.py', 'HARDENING'),
]:
    try:
        txt=open(f).read()
        check(f"{pathlib.Path(f).name} {needle}", needle in txt)
    except Exception as e:
        check(f"{f} exists", False)
import os
check('env VLLM_BT_POOL', os.getenv('VLLM_BT_POOL') is not None)
try:
    import vllm
    from vllm.config import CacheConfig
    print(f"vllm {vllm.__version__}")
    import inspect
    src=inspect.getsource(CacheConfig)
    check('CacheConfig has mamba_ssm_cache_dtype', 'mamba_ssm_cache_dtype' in src)
except Exception as e:
    print(f"import check failed: {e}")
print("\n--- summary ---")
for desc,ok in checks:
    print(f"{'OK' if ok else 'FAIL'} {desc}")
if all(ok for _,ok in checks):
    print("ALL CHECKS PASS")
else:
    print("SOME CHECKS FAIL - rebuild needed")
