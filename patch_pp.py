import pathlib, re
for p in [pathlib.Path("vllm/models/qwen4_exp/nvidia/model_state.py"),
          pathlib.Path("vllm/models/qwen4_exp/amd/model_state.py")]:
    t = p.read_text()
    if "get_pp_group" not in t:
        t = t.replace("from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState",
                      "from vllm.distributed.parallel_state import get_pp_group\nfrom vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState")
    old = """        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )"""
    new = """        try:
            is_first_pp = get_pp_group().is_first_rank
        except Exception:
            is_first_pp = True
        self.uses_ngram_embedding = bool(config.ple_layer_ids) and is_first_pp
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return"""
    if old in t:
        t = t.replace(old, new)
        print("patched", p)
    else:
        print("WARNING model_state pattern not found:", p)
        t = re.sub(r"if vllm_config\.parallel_config\.pipeline_parallel_size > 1:\s+raise RuntimeError\(.*?with PP=1\.\n\s*\)",
                   "if False:", t, flags=re.S)
        t = t.replace("        self.uses_ngram_embedding = bool(config.ple_layer_ids)",
                      "        self.uses_ngram_embedding = bool(config.ple_layer_ids) and get_pp_group().is_first_rank")
    p.write_text(t)

# 5) model_runner.py — _setup_ple_offload must be a no-op on non-first PP
#    ranks: model_state has no ple_query_start_loc/ngram_context there
#    (PLE lives only in stage 0), so the "requires a source" raise is wrong
#    under PP>1. Gate on get_pp_group().is_first_rank before raising.
p = pathlib.Path("vllm/v1/worker/gpu/model_runner.py")
t = p.read_text()
old_setup = """        query_start_loc_source = getattr(self.model_state, "ple_query_start_loc", None)
        ngram_context_source = getattr(self.model_state, "ngram_context", None)
        if not isinstance(query_start_loc_source, torch.Tensor):
            raise RuntimeError("PLE offload requires a query_start_loc source")"""
new_setup = """        query_start_loc_source = getattr(self.model_state, "ple_query_start_loc", None)
        ngram_context_source = getattr(self.model_state, "ngram_context", None)
        # PP patched (qwen38-flash-next:pp2): non-first pipeline ranks own no
        # PLE layers and their model_state skips the PLE buffers entirely.
        from vllm.distributed.parallel_state import get_pp_group as _get_pp_group
        try:
            _is_first_pp = _get_pp_group().is_first_rank
        except Exception:
            _is_first_pp = True
        if not isinstance(query_start_loc_source, torch.Tensor):
            if not _is_first_pp:
                return
            raise RuntimeError("PLE offload requires a query_start_loc source")"""
if old_setup in t:
    t = t.replace(old_setup, new_setup)
    print("patched model_runner setup_ple_offload")
else:
    print("WARNING model_runner setup_ple_offload pattern not found")
p.write_text(t)

# 6) mtp.py (nvidia + amd) — PP-aware MTP draft input.
#    Under PP>1 the MTP drafter runs on the LAST rank, where the target's
#    hidden_states arrive as intermediate_tensors["hidden_states"]
#    (make_layers puts PPMissingLayer for layers 0..23, so the
#    is_first_rank branch never fires). The draft path in
#    spec_decode/autoregressive/speculator.py calls the MTP model with
#    hidden_states= and intermediate_tensors=None ->
#    "assert intermediate_tensors is not None" (fails Dynamo fullgraph
#    compile as a data-dependent assert). Accept hidden_states directly.
for p in [pathlib.Path("vllm/models/qwen4_exp/nvidia/mtp.py"),
          pathlib.Path("vllm/models/qwen4_exp/amd/mtp.py")]:
    t = p.read_text()
    old = """        if get_pp_group().is_first_rank:
            assert hidden_states is not None"""
    new = """        if get_pp_group().is_first_rank or hidden_states is not None:
            assert hidden_states is not None"""
    if old in t:
        t = t.replace(old, new)
        print("patched mtp.py draft-branch", p)
    else:
        print("WARNING mtp.py pattern not found:", p)
    p.write_text(t)

# 3) gpu_worker.py — _validate_ple_offload_config no longer rejects PP>1
#    (the PLE offload path is now PP-aware: only stage 0 holds PLE layers)
p = pathlib.Path("vllm/v1/worker/gpu_worker.py")
t = p.read_text()
old_pp = """        if parallel_config.pipeline_parallel_size != 1:
            unsupported.append(f"PP={parallel_config.pipeline_parallel_size}")"""
new_pp = """        # PP patched (qwen38-flash-next:pp2): PLE offload is PP-aware — only
        # the first pipeline stage holds PLE layers, so PP > 1 is supported.
        pass"""
if old_pp in t:
    t = t.replace(old_pp, new_pp)
    print("patched gpu_worker validate")
else:
    print("WARNING gpu_worker PP check pattern not found")
p.write_text(t)

# 4) model.py (nvidia + amd) — PP-aware skip of the 3 persistent
#    hyper_connection_mixer weights on non-last pipeline ranks.
#    Root cause (confirmed from source): Qwen4ExpModel.__init__ creates
#    self.hyper_connection_mixer only on is_last_rank (else None), but
#    load_weights only skipped 'hyper_connection_mixer.block_inject_weight'
#    (the non-persistent one). The 3 persistent keys
#    model.language_model.hyper_connection_mixer.{hc_norm,
#    input_mix_weight_down, input_mix_weight_up}.weight then hit an empty
#    module on rank 0 (PP>1) -> 'no module or parameter named
#    hyper_connection_mixer in Qwen4ExpModel'. Skip them on non-last ranks.
for p in [pathlib.Path("vllm/models/qwen4_exp/nvidia/model.py"),
          pathlib.Path("vllm/models/qwen4_exp/amd/model.py")]:
    t = p.read_text()
    old = """        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]"""
    new = """        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]
        # PP patched (qwen38-flash-next:pp2): hyper_connection_mixer exists
        # only on the final pipeline rank, so skip its persistent weights on
        # non-last ranks (they would otherwise hit an empty module).
        if not get_pp_group().is_last_rank:
            skip_substrs.append("hyper_connection_mixer.")"""
    if old in t:
        t = t.replace(old, new)
        print("patched model.py HC skip", p)
    else:
        print("WARNING model.py HC skip pattern not found:", p)
        # fallback: append to the literal block_inject_weight line
        t = t.replace('"hyper_connection_mixer.block_inject_weight",',
                      '"hyper_connection_mixer.block_inject_weight",\n        ]\n        if not get_pp_group().is_last_rank:\n            skip_substrs.append("hyper_connection_mixer.")\n        [')
    p.write_text(t)

p = pathlib.Path("vllm/v1/ple_offload/connector.py")
t = p.read_text()
if "get_pp_group" not in t:
    t = t.replace("from vllm.distributed.parallel_state import get_dp_group, get_tp_group",
                  "from vllm.distributed.parallel_state import get_dp_group, get_pp_group, get_tp_group")
old_init = "        self.dp_rank = get_dp_group().rank_in_group\n        self.tp_rank = get_tp_group().rank_in_group\n        self._layers = self._setup_layers(vllm_config, model)"
new_init = ("        self.dp_rank = get_dp_group().rank_in_group\n"
            "        self.tp_rank = get_tp_group().rank_in_group\n"
            "        try:\n"
            "            self.is_first_pp = get_pp_group().is_first_rank\n"
            "        except Exception:\n"
            "            self.is_first_pp = True\n"
            "        self._layers = self._setup_layers(vllm_config, model)\n"
            "        if not self._layers and not self.is_first_pp:\n"
            "            return")
if old_init in t:
    t = t.replace(old_init, new_init)
    print("patched connector init")
else:
    print("WARNING connector init pattern not found")
old_setup = """        if not layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )"""
new_setup = """        if not layers:
            try:
                is_first = get_pp_group().is_first_rank
            except Exception:
                is_first = True
            if not is_first:
                return {}
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )"""
if old_setup in t:
    t = t.replace(old_setup, new_setup)
    print("patched connector setup")
old_launch = "        if self.tp_rank != 0:\n            return\n\n        if self._uses_cuda_inputs:\n            assert self._input_ready_event is not None"
new_launch = "        if not getattr(self, 'is_first_pp', True):\n            return\n        if self.tp_rank != 0:\n            return\n\n        if self._uses_cuda_inputs:\n            assert self._input_ready_event is not None"
if old_launch in t:
    t = t.replace(old_launch, new_launch)
    print("patched connector launch")
else:
    print("WARNING connector launch pattern not found")
p.write_text(t)
print("all patched")
