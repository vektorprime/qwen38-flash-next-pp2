# GPU time of moe_align_block_size inside a CUDA graph, with and without the patch.
import torch

import vllm.model_executor.layers.fused_moe.moe_align_block_size as m

torch.manual_seed(0)
dev = "cuda"
E, TOPK, BLOCK, LAYERS = 512, 10, 8, 48

for M in (4, 32, 64, 2048):
    bias = torch.linspace(4, 0, E, device=dev)[torch.randperm(E, device=dev)]
    ids = torch.topk(torch.randn(M, E, device=dev) + bias, TOPK, dim=-1).indices.to(torch.int32)
    res = {}
    for flag in (False, True):
        m._DETERMINISTIC_ALIGN = flag
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                m.moe_align_block_size(ids, BLOCK, E, ignore_invalid_experts=True)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(LAYERS):  # one call per MoE layer, like one forward
                m.moe_align_block_size(ids, BLOCK, E, ignore_invalid_experts=True)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(50):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        res[flag] = st.elapsed_time(en) / 50
    print(f"M={M:5d}: {LAYERS} align calls per forward in a graph: "
          f"unpatched {res[False]*1000:.0f} us, patched {res[True]*1000:.0f} us, "
          f"overhead {(res[True]-res[False])*1000:.0f} us per forward")
