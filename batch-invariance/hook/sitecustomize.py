# Batch-invariance + mechanism hook for eval containers (mounted via PYTHONPATH). Every feature is env-gated:
#
# PLEFP8_INV_MOE=1   Batch-invariant Marlin MoE. Marlin's scheduler (marlin_template.h, "DP + two-tile stream-K") splits
#     the K-reduction of the last 1/3..4/3 grid of output tiles across threadblocks; which tiles and where depends on the
#     total tile count, i.e. on the batch. Fix: one pinned launch config per block-size class with thread_k=64 for every
#     block size (thread_k changes per-row results; thread_n and blocks_per_sm do not), and append on the device the
#     smallest number P of empty expert blocks such that, for both GEMMs, Marlin's stream-K region (the last part2 output
#     tiles) lies entirely in padding: every real tile is then computed whole by one threadblock, in a fixed K order.
#     Padding blocks point at expert 0 and contain only the sentinel token, so they write nothing. Graph-safe (no host
#     sync). Launch config PLEFP8_MARLIN_CFG="tk,tn_small,bps_small,tn_large,bps_large"; default 64,128,2,128,2 (fastest
#     invariant config on the CMP 170HX for INT8; INT4 group-32 needs bps_large=1 where shared memory is < 100 KB/block).
# PLEFP8_INV_GEMM=1  Dense bf16 GEMMs (vLLM default_unquantized_gemm: all linear layers, router gate, lm_head) via a plain
#     Triton GEMM whose per-row result is independent of tile config and M (inv_gemm.py), as an opaque custom op.
#     Needs a fresh torch.compile cache (set VLLM_CACHE_ROOT to an empty dir), or the old cuBLAS graph is reused.
# PLEFP8_INV_QSA=1   QSA sparse attention with one fixed split profile (block_n=64, PLEFP8_QSA_SPLITS splits, default 8)
#     instead of a profile chosen from the number of tokens in the batch; queries are processed in chunks of 256 rows so
#     the split-K partial buffers stay small (rows are independent, so chunking does not change any result). Also the QSA
#     indexer's token selection: persistent_topk returns the selected blocks in a run-dependent order (and with exact score
#     ties a run-dependent set) once a query sees > 512 compressed blocks (context > 2048 tokens); re-selected
#     deterministically (inv_qsa_select.py: score > k-th score, ties by lowest index, ascending order).
# PLEFP8_ALIGN_CHUNKS=N  Scheduler: every non-final prefill chunk ends on a multiple of N (64 = GDN chunk size), so where a
#     prompt is split never depends on its length or on the load (chunk-boundary invariance; a GDN continuation from a
#     cached state at a multiple of 64 is bit-identical to computing through it). Two places choose chunk ends:
#     _mamba_block_aligned_split (mamba "align" cache stops, incl. the last cacheable position floor8(L)-8 near every
#     prompt end) and _reserve_prefill_lookahead (MTP lookahead). A non-aligned stop is rounded down to a multiple of N;
#     if that leaves nothing to compute, the chunk runs on to the prompt end (or the last multiple of N within the token
#     budget) instead. Skipping a cache stop only means that state is not cached (fewer prefix-cache hit positions).
# PLEFP8_CTRL=<dir>  Mechanism experiments, all prefill-only (calls with >= PLEFP8_MIN_M rows, default 512):
#     <dir>/inject.json  {"layers": [ids] or "all", "frac": f, "cols": c, "seed": s}: after the MoE of those layers, add 1 bf16
#                        ulp to c fixed random columns of a fixed random fraction f of rows (row = position in the step).
#     <dir>/route/ON + <dir>/route/TAG   capture logical top-k expert ids per layer -> <dir>/route/<TAG>/r{rank}_L{l}_{k}_M{M}.npy
#     <dir>/replay/ON + <dir>/replay/SRC [+ <dir>/replay/LMAX]  routing replay (R3-style): replace the top-k expert ids of
#                        rows < LMAX by those captured under tag SRC (same window order), weights recomputed from this run's
#                        router logits with the router's own normalisation.
import os

_INV_MOE = os.environ.get("PLEFP8_INV_MOE") == "1"
_INV_GEMM = os.environ.get("PLEFP8_INV_GEMM") == "1"
_INV_QSA = os.environ.get("PLEFP8_INV_QSA") == "1"
_CTRL = os.environ.get("PLEFP8_CTRL")
_MIN_M = int(os.environ.get("PLEFP8_MIN_M", "512"))

if _INV_MOE or _INV_GEMM or _INV_QSA or _CTRL or os.environ.get("PLEFP8_ALIGN_CHUNKS"):
    import sys, json, math, importlib.abc, importlib.util

    def _log(msg):
        print(f"[hook3] {msg}", flush=True)

    def _pp_rank():
        try:
            from vllm.distributed.parallel_state import get_pp_group
            return get_pp_group().rank_in_group
        except Exception:  # noqa
            return 0

    # ------------------------------------------------------------------ invariant Marlin MoE
    def _patch_marlin(mod):
        import torch
        from vllm import _custom_ops as ops
        orig_align, orig_gemm = mod.moe_align_block_size, ops.moe_wna16_marlin_gemm
        tk, tns, bpss, tnl, bpsl = (int(v) for v in os.environ.get("PLEFP8_MARLIN_CFG", "64,128,2,128,2").split(","))
        CFG_SMALL = {"thread_k": tk, "thread_n": tns, "blocks_per_sm": bpss}   # thread_m_blocks == 1 (block size <= 16)
        CFG_LARGE = {"thread_k": tk, "thread_n": tnl, "blocks_per_sm": bpsl}
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import inv_marlin
        align, gemm = inv_marlin.make(orig_align, orig_gemm, CFG_SMALL, CFG_LARGE,
                                      on_first=lambda sms: _log(f"invariant Marlin MoE active: sms={sms} small={CFG_SMALL} large={CFG_LARGE} (pp rank {_pp_rank()})"))

        mod.moe_align_block_size = align
        ops.moe_wna16_marlin_gemm = gemm
        _log("marlin_moe patched (smart padding + pinned exec config)")

    # ------------------------------------------------------------------ invariant dense GEMM
    def _patch_gemm(mod):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import inv_gemm
        orig = mod.default_unquantized_gemm
        st = {"banner": False}

        def default_unquantized_gemm(layer, x, weight, bias=None):
            # traced by torch.compile (fullgraph): no side effects here; the banner is printed by the op at runtime
            if not inv_gemm.applicable(x, weight):
                return orig(layer, x, weight, bias)
            return inv_gemm.linear(x, weight, bias)

        mod.default_unquantized_gemm = default_unquantized_gemm
        _log("default_unquantized_gemm patched (invariant Triton GEMM)")

    # ------------------------------------------------------------------ invariant QSA split profile
    def _patch_qsa(mod):
        import torch, triton
        orig = mod.qsa_sparse_paged_attention
        BLOCK_N, TARGET_SPLITS, PARTIAL_WARPS = 64, int(os.environ.get("PLEFP8_QSA_SPLITS", "8")), 2
        ROWS = 256
        st = {"banner": False}

        def _launch(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out):
            group_size = q.shape[1] // k_cache.shape[2]
            block_m = triton.next_power_of_2(group_size)
            num_tiles = triton.cdiv(logical_indices.shape[1], BLOCK_N)
            num_splits = min(1 << (num_tiles.bit_length() - 1), TARGET_SPLITS)
            if num_splits == 1:
                partial_output = partial_lse = out
            else:
                partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
                partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
            mod._qsa_sparse_paged_gqa_splitk_kernel[(q.shape[0], k_cache.shape[2], num_splits)](
                q, k_cache, v_cache, logical_indices, block_table, token_to_req, partial_output, partial_lse, out,
                q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), logical_indices.stride(0), block_table.stride(0),
                out.stride(0), out.stride(1), q.shape[0], k_cache.shape[0], block_table.shape[0],
                TOPK=logical_indices.shape[1], PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
                GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
                NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=BLOCK_N, num_warps=PARTIAL_WARPS, num_stages=2)
            if num_splits > 1:
                mod._qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
                    partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0], HEAD_DIM=q.shape[2],
                    NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits, BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                    num_warps=2, num_stages=1)

        def qsa_sparse_paged_attention(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out=None):
            if out is None:
                out = torch.empty_like(q)
            T = q.shape[0]
            if not T:
                return out
            if not st["banner"]:
                st["banner"] = True
                _log(f"QSA pinned split profile in use: block_n={BLOCK_N} target_splits={TARGET_SPLITS} rows/launch={ROWS} (pp rank {_pp_rank()})")
            for i in range(0, T, ROWS):
                j = min(T, i + ROWS)
                _launch(q[i:j], k_cache, v_cache, logical_indices[i:j], block_table, token_to_req[i:j], out[i:j])
            return out

        qsa_sparse_paged_attention._plefp8_orig = orig
        mod.qsa_sparse_paged_attention = qsa_sparse_paged_attention
        _log("qsa_sparse_paged_attention patched (pinned split profile)")

        # deterministic token selection (inv_qsa_select.py): vLLM's qsa_select_paged_tokens with one added step after the top-k
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import inv_qsa_select
        from vllm.platforms import current_platform
        orig_sel = mod.qsa_select_paged_tokens
        st_sel = {"banner": False}

        def qsa_select_paged_tokens(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, token_topk,
                                    compress_ratio, out=None):
            rows = q.shape[0]
            output_width = token_topk + compress_ratio - 1
            if out is None:
                out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
            if out.shape != (rows, output_width):
                raise ValueError("QSA selection output has an invalid shape")
            if not rows:
                return out
            if not st_sel["banner"]:
                st_sel["banner"] = True
                _log(f"QSA deterministic token selection in use (pp rank {_pp_rank()})")
            columns = page_table.shape[1] * k_cache.shape[1]
            block_topk = token_topk // compress_ratio
            rows_per_chunk = max(1, mod._LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
            chunk_rows = min(rows, rows_per_chunk)
            blocks_buffer = torch.empty((chunk_rows, block_topk), dtype=torch.int32, device=q.device)
            topk_workspace = torch.empty((mod._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device)
            for row_start in range(0, rows, rows_per_chunk):
                row_end = min(row_start + rows_per_chunk, rows)
                row_slice = slice(row_start, row_end)
                logits, visible_blocks = mod.qsa_mqa_paged(q[row_slice], k_cache, page_table, token_to_req[row_slice],
                                                           query_positions[row_slice], sequence_lengths, compress_ratio)
                blocks = blocks_buffer[: row_end - row_start]
                use_cooperative_topk = (blocks.shape[0] <= 32 and logits.stride(0) % 4 == 0
                                        and current_platform.has_device_capability(90)
                                        and not current_platform.is_device_capability_family(120))
                topk_op = torch.ops._C.cooperative_topk if use_cooperative_topk else torch.ops._C.persistent_topk
                topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)
                inv_qsa_select.fix_selection(logits, visible_blocks, blocks, block_topk, columns)
                mod.expand_qsa_block_indices_cuda(blocks, query_positions[row_slice], sequence_lengths, token_to_req[row_slice],
                                                  compress_ratio, token_topk, out[row_slice])
            return out

        qsa_select_paged_tokens._plefp8_orig = orig_sel
        mod.qsa_select_paged_tokens = qsa_select_paged_tokens
        _log("qsa_select_paged_tokens patched (deterministic top-k order and ties)")

    # ------------------------------------------------------------------ mechanism experiments (prefill only)
    _ctl = {"inject": None, "inject_mtime": None, "rowsel": None, "colsel": None, "k": 0,
            "replay_src": None, "replay": {}, "replay_ctr": {}, "replay_norm": None, "replay_lmax": 1 << 30}

    def _inject_cfg():
        p = f"{_CTRL}/inject.json"
        try:
            mt = os.path.getmtime(p)
        except OSError:
            _ctl["inject"] = None
            return None
        if mt != _ctl["inject_mtime"]:
            _ctl["inject_mtime"] = mt
            try:
                cfg = json.load(open(p))
            except Exception:  # noqa  (file being written)
                return _ctl["inject"]
            import torch
            g = torch.Generator().manual_seed(int(cfg.get("seed", 1)))
            R, H = 8192, 2560
            _ctl["rowsel"] = (torch.rand(R, generator=g) < float(cfg["frac"]))
            _ctl["colsel"] = torch.randint(0, H, (R, int(cfg.get("cols", 16))), generator=g)
            _ctl["inject"] = cfg
            _ctl["dev_cache"] = {}
            _log(f"inject config loaded: {cfg} -> {int(_ctl['rowsel'].sum())} of {R} row slots")
        return _ctl["inject"]

    def _apply_inject(layer_id, out):
        cfg = _inject_cfg()
        if cfg is None:
            return
        layers = cfg.get("layers", [0])
        if layers != "all" and layer_id not in layers:
            return
        import torch
        M = out.shape[0]
        key = (str(out.device), M)
        dc = _ctl["dev_cache"]
        if key not in dc:
            rows = torch.nonzero(_ctl["rowsel"][:M]).flatten()
            cols = _ctl["colsel"][rows]
            dc[key] = (rows.repeat_interleave(cols.shape[1]).to(out.device), cols.flatten().to(out.device))
        r, c = dc[key]
        if r.numel():
            v = out.view(torch.int16)
            v[r, c] += 1   # +1 ulp in magnitude (bf16 sign-magnitude)

    def _capture_routes(layer_id, topk_ids):
        try:
            if not os.path.exists(f"{_CTRL}/route/ON"):
                return
            import numpy as np, torch
            tag = open(f"{_CTRL}/route/TAG").read().strip()
            d = f"{_CTRL}/route/{tag}"
            os.makedirs(d, exist_ok=True)
            k = _ctl["k"]; _ctl["k"] += 1
            np.save(f"{d}/r{_pp_rank()}_L{int(layer_id):02d}_{k:05d}_M{int(topk_ids.shape[0])}.npy", topk_ids.detach().to(torch.int16).cpu().numpy())
        except Exception as e:  # noqa
            _log(f"route capture failed: {e!r}")

    def _replay_ids(layer_id, M):
        """captured ids for this (layer, call) or None"""
        on = os.path.exists(f"{_CTRL}/replay/ON")
        if not on:
            _ctl["replay_src"] = None
            return None
        src = open(f"{_CTRL}/replay/SRC").read().strip()
        if src != _ctl["replay_src"]:
            import glob, re, numpy as np
            per = {}
            files = sorted(glob.glob(f"{_CTRL}/route/{src}/r*_L*_*_M*.npy"), key=lambda p: int(re.search(r"_(\d{5})_M", p).group(1)))
            for f in files:
                lid = int(re.search(r"_L(\d+)_", os.path.basename(f)).group(1))
                per.setdefault(lid, []).append(f)
            _ctl.update(replay_src=src, replay=per, replay_ctr={})
            try:
                _ctl["replay_lmax"] = int(open(f"{_CTRL}/replay/LMAX").read().strip())
            except Exception:  # noqa
                _ctl["replay_lmax"] = 1 << 30
            _log(f"replay source {src}: {sum(len(v) for v in per.values())} files, {len(per)} layers, lmax {_ctl['replay_lmax']}")
        lst = _ctl["replay"].get(layer_id)
        if not lst:
            return None
        i = _ctl["replay_ctr"].get(layer_id, 0)
        _ctl["replay_ctr"][layer_id] = i + 1
        if i >= len(lst):
            return None
        import numpy as np
        return np.load(lst[i])

    def _patch_runner(mod):
        orig = mod.MoERunner._forward_impl

        def _forward_impl(self, hidden_states, router_logits, *a, **kw):
            if not getattr(self, "_plefp8_bound", False):
                self._plefp8_bound = True
                try:
                    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
                    lid = int(self.layer_id)
                    if isinstance(self.router, BaseRouter):
                        self.router._plefp8_layer = lid
                        self.router.set_capture_fn(lambda ids, _l=lid: _capture_routes(_l, ids) if ids.shape[0] >= _MIN_M else None)
                    else:
                        _log(f"layer {lid}: router {type(self.router).__name__} not hookable")
                except Exception as e:  # noqa
                    _log(f"bind failed: {e!r}")
            res = orig(self, hidden_states, router_logits, *a, **kw)
            try:
                if hidden_states.shape[0] >= _MIN_M:
                    out = res[1] if isinstance(res, tuple) else res
                    _apply_inject(int(self.layer_id), out)
            except Exception as e:  # noqa
                _log(f"inject failed: {e!r}")
            return res

        mod.MoERunner._forward_impl = _forward_impl
        _log("MoERunner._forward_impl wrapped (inject / route capture)")

    def _patch_router(mod):
        import torch
        orig = mod.BaseRouter._select_experts

        def _select_experts(self, hidden_states, router_logits, *a, **kw):
            topk_weights, topk_ids = orig(self, hidden_states, router_logits, *a, **kw)
            lid = getattr(self, "_plefp8_layer", None)
            if lid is None or router_logits.shape[0] < _MIN_M:
                return topk_weights, topk_ids
            src = _replay_ids(lid, router_logits.shape[0])
            if src is None:
                return topk_weights, topk_ids
            p = torch.softmax(router_logits.float(), dim=-1)
            if _ctl["replay_norm"] is None:   # calibrate against the router's own weights
                w = p.gather(1, topk_ids.long())
                if torch.allclose(w, topk_weights.float(), atol=1e-4, rtol=1e-3):
                    _ctl["replay_norm"] = False
                elif torch.allclose(w / w.sum(-1, keepdim=True), topk_weights.float(), atol=1e-4, rtol=1e-3):
                    _ctl["replay_norm"] = True
                else:
                    _log("replay: cannot reproduce router weights (not softmax top-k); replay disabled")
                    _ctl["replay_norm"] = "off"
                _log(f"replay: renormalize={_ctl['replay_norm']}")
            if _ctl["replay_norm"] == "off":
                return topk_weights, topk_ids
            n = min(topk_ids.shape[0], src.shape[0], _ctl["replay_lmax"])
            ids = topk_ids.clone()
            ids[:n] = torch.from_numpy(src[:n]).to(device=ids.device, dtype=ids.dtype)
            w = p.gather(1, ids.long())
            if _ctl["replay_norm"]:
                w = w / w.sum(-1, keepdim=True)
            return w.to(topk_weights.dtype), ids

        mod.BaseRouter._select_experts = _select_experts
        _log("BaseRouter._select_experts wrapped (routing replay)")

    def _patch_sched(mod):
        A = int(os.environ["PLEFP8_ALIGN_CHUNKS"])
        orig = mod.Scheduler._reserve_prefill_lookahead
        orig_split = getattr(mod.Scheduler, "_mamba_block_aligned_split", None)
        st = {"n": 0, "ns": 0}

        if orig_split is not None:
            def _mamba_block_aligned_split(self, request, num_new_tokens, num_new_local_computed_tokens=0, num_external_computed_tokens=0):
                n = orig_split(self, request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens)
                start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
                prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
                end = start + n
                if n <= 0 or start >= prefill_end or end >= prefill_end or end % A == 0:
                    return n   # decode, final prefill chunk, or already aligned
                aligned = end // A * A
                if aligned > start:
                    new = aligned - start
                else:   # vLLM stopped inside this A-block: run on to the prompt end, or the last A boundary within the budget
                    req_end = start + num_new_tokens
                    if req_end >= prefill_end:
                        new = num_new_tokens   # final chunk, exactly as vLLM would schedule it without the stop
                    elif req_end // A * A > start:
                        new = req_end // A * A - start
                    else:
                        return n   # budget smaller than one A-block: keep vLLM's choice
                if st["ns"] < 3:
                    st["ns"] += 1
                    _log(f"mamba split aligned: start {start} end {end} -> {start + new} (prompt {request.num_prompt_tokens})")
                return new

            mod.Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split

        def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
            n = orig(self, request, num_computed_tokens, num_new_tokens)
            prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
            end = num_computed_tokens + n
            if n > 0 and num_computed_tokens < prefill_end and end < prefill_end:   # non-final prefill chunk
                aligned = end // A * A
                if aligned > num_computed_tokens and aligned != end:
                    if st["n"] < 3:
                        st["n"] += 1
                        _log(f"prefill chunk aligned: start {num_computed_tokens} end {end} -> {aligned} (prompt {request.num_prompt_tokens})")
                    n = aligned - num_computed_tokens
            return n

        mod.Scheduler._reserve_prefill_lookahead = _reserve_prefill_lookahead
        _log(f"Scheduler: non-final prefill chunks aligned to multiples of {A} (lookahead{' + mamba split' if orig_split else ''})")

    _TARGETS = {}
    if _INV_MOE:
        _TARGETS["vllm.model_executor.layers.fused_moe.experts.marlin_moe"] = _patch_marlin
    if _INV_GEMM:
        _TARGETS["vllm.model_executor.layers.utils"] = _patch_gemm
    if _INV_QSA:
        _TARGETS["vllm.models.qwen4_exp.nvidia.ops.qsa"] = _patch_qsa
    if os.environ.get("PLEFP8_ALIGN_CHUNKS"):
        _TARGETS["vllm.v1.core.sched.scheduler"] = _patch_sched
    if _CTRL:
        _TARGETS["vllm.model_executor.layers.fused_moe.runner.moe_runner"] = _patch_runner
        _TARGETS["vllm.model_executor.layers.fused_moe.router.base_router"] = _patch_router

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _TARGETS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _o=orig_exec, _p=_TARGETS[name]):
                _o(module)
                try:
                    _p(module)
                except Exception as e:  # noqa  never break the server
                    _log(f"patch of {name} FAILED: {e!r}")
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
