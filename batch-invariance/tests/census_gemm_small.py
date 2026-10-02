"""Dense bf16 GEMM batch invariance at decode batch sizes: rows 0..3 computed at M in {4, 8, 16, 24, 32, 40, 48, 56, 64}
(prod's CUDA-graph sizes x 4 tokens per request with MTP k=3) and at M=1 for row 0. Bitwise comparison vs M=4."""
import json, sys, torch
import torch.nn.functional as F
torch.manual_seed(0)
dev = "cuda"
SHAPES = {"gdn in_proj_qkvz": (2560, 16384), "gdn in_proj_ba": (2560, 96), "gdn out_proj": (6144, 2560), "moe router gate": (2560, 512),
          "shared expert gate_up": (2560, 1280), "shared expert down": (640, 2560), "shared_expert_gate": (2560, 1), "hc down+inject": (10240, 336),
          "hc up": (320, 10240), "ple key_proj": (2560, 10240), "attn q_proj": (2560, 12288), "attn k/v_proj": (2560, 512), "qsa index_qk_proj": (2560, 640),
          "attn o_proj": (6144, 2560), "lm_head": (2560, 248320), "mtp fc": (2560, 2560)}
MS = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, 256]
res = {"device": torch.cuda.get_device_name()}
for name, (K, N) in SHAPES.items():
    W = (torch.randn(N, K, device=dev) / K ** 0.5).to(torch.bfloat16)
    X = torch.randn(256, K, device=dev).to(torch.bfloat16)
    ref = F.linear(X[:1], W)
    groups = {}
    for M in MS:
        y = F.linear(X[:M], W)[:1]
        key = next((g for g in groups if torch.equal(groups[g][0], y)), None)
        if key is None: groups[M] = (y, [M])
        else: groups[key][1].append(M)
    res[name] = [g[1] for g in groups.values()]
    print(f"{name:22s} row-0 result groups by M: {res[name]}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
