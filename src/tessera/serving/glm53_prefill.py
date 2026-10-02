"""GLM-5.3 (Glm5Next) prefill: the KDA output norm's CUDA path and sequence-parallel mHC.

Two serve-time changes to the pinned vLLM's stock GLM-5.3 model, installed from
``TesseraConfig.get_quant_method`` (the hook ``mtp_draft_lifetime`` uses), and
both declining to stock when the serve is not the one they were measured on.

**The KDA output norm.**  vLLM serves Glm5Next with breakable CUDA graphs by
default, which set compilation mode ``NONE`` and so ``custom_ops`` ``all``.  A
serve that opts out (``VLLM_USE_BREAKABLE_CUDAGRAPH=0``) with graphs on keeps
mode ``VLLM_COMPILE`` and the inductor backend, where ``custom_ops`` resolves
to ``none``; the model has no ``@support_torch_compile``, so inductor never
compiles it, and every custom op whose name is not enabled runs
``forward_native``.  The serve flag ``-cc.mode=none`` is the fix for every op;
this hook is the backstop for the one that costs most.  For ``FusedRMSNormGated`` that is nine eager kernels per
KDA layer (the decomposition meant for inductor fusion); ``forward_cuda`` is
one Triton kernel.  With ``TESSERA_GLM53_ONORM_CUDA=1``,
:func:`enable_onorm_cuda` appends ``+fused_rms_norm_gated`` to the current
config's ``custom_ops`` when the model is Glm5Next and the serve's own
``custom_ops`` names that op neither way, so a stock serve command runs vLLM's
own ``forward_cuda``.  Nothing is rebound.  It is off by default: under mode
``NONE`` (the release serve) ``custom_ops`` is already ``all`` and the hook
changes nothing, and in any other mode it would change a stock default that
no served A/B covers.

**Sequence-parallel mHC.**  At TP 2 every rank computes the mHC residual stream
(4 x 4096 per token) for every token, while attention and the MoE are sharded.
:func:`install_sp_mhc` rebinds ``Glm5NextDecoderLayer.forward`` to a copy of
the stock forward that, for a batch of at least ``T*`` tokens, keeps the mHC
state on this rank's half of the tokens: it all-gathers the layer input before
attention and before the MLP, and reduce-scatters their outputs where stock
all-reduces them.  On two ranks a reduce-scatter plus an all-gather moves the
bytes one all-reduce moves; the mHC work per rank halves.

**Exact SP.**  The pinned mHC kernels are not invariant to the token count of
a call: two 1024-token calls differ from one 2048-token call over the same
tokens (post and comb mixes at about 1e-5, the layer input by up to 2 bf16
ulps of its row max).  The whole cause is the pre-norm GEMM's split-k, which
``compute_num_split`` derives from the call's token count: with the split
forced to one value for a call and its chunks, every output is bitwise equal
at 256 to 8192 tokens in 2 and 4 chunks, on both mHC sites
(``experiments/mhc/mhc_probe.py`` ``mhcsplit``, PrismaBuild row 22f7d4f2).
So on an SP pass every mHC call on this rank's shard runs at the split the
full batch's call takes.  :class:`SplitForcer` replaces
``tilelang_kernels.compute_num_split``, which ``_hc_prenorm_gemm_outputs``
reads at call time, with a wrapper that returns the stock value except inside
:meth:`SplitForcer.full_batch`, where it answers for the full batch's grid.
Only the SP branch's mHC calls run inside it.  A pass takes SP only when the
full batch and its shard both reach the split-k path: vLLM's own
``mhc_fused_post_pre_split_config`` returns None for both token counts.  At
or below its small-batch limit, the fused kernel picks its own split, so SP
there would not be exact, and the pass runs stock.  The one inexact run
(window u4-R1-20261001T0058Z, before this) moved the served TR3 panel from
0.027886 to 0.028144.  The two-operand sums of the collectives are expected
to match the all-reduce's, but that is an inference, and the served A/B
against stock is the test: identical KL and logprobs, not "within noise".
The attention ``o_proj`` and the MLP's final reduction are switched off on
each layer's first forward, and below ``T*`` the rebound forward performs those
two all-reduces itself with the op the modules call.  A forward under graph
capture never takes the SP branch, so a decode graph captures the stock op
sequence whatever ``T*`` is.  Module construction and weight sharding do
not change.

``T*`` is measured, per serve, at the first forward that is not a graph
capture (vLLM's profile run, which runs even with ``kv_cache_memory_bytes``),
on tensors of its own up to ``max_num_batched_tokens``: per token count on a
power-of-two grid where SP is exact, the mHC saving (two
``hc_fused_post_pre`` calls on ``T`` tokens against two on ``T/2`` at the
``T``-token split) against the extra collective time (an all-gather plus a
reduce-scatter against an all-reduce, twice per layer), each the median of
CUDA-event timings on this serve's TP group.  ``T*`` is the smallest grid count from which the saving
exceeds the cost at every larger grid count, agreed across ranks by a MAX
reduction, and logged with the table.  No threshold is a constant here.
The measurement is known to be wrong at small ``T``: isolated medians read the
all-gather plus reduce-scatter as cheaper than the all-reduce from 32 to 1024
tokens and chose ``T*`` 32, while the same serve's profile at 512 tokens
showed SP 10 ms per step slower than stock (window u4-R1-20261001T0058Z).

**The KDA prefill conv, per q/k/v slice.**  The pinned KDA layer runs one
short causal conv over the merged q|k|v channels and splits its token-major
output, so q, k and v reach FlashKDA as row-strided views, and FlashKDA's
dense-stride contract makes ``_flashkda_prefill`` copy all three
(``.contiguous()``, 3 x [T, 4096] bf16 per KDA layer).  The conv is
independent per channel, so running it once per slice is the same arithmetic
(vLLM's own comment says so of the merge), and each call's ``empty_like``
output on a channel-last slice is dense token-major, so FlashKDA's copies
become no-ops.  :func:`install_kda_conv_split` rebinds
``Glm5NextLinearAttention._forward`` to the stock method compiled from its own
source with that one block replaced (:data:`KDA_STOCK_CONV_BLOCK`), only when
both touched files are byte-identical to an inspected interface and the block
occurs exactly once; the stock decorator is applied to the result.  A conv
bias (stock passes ``q_conv1d.bias`` with the merged 3P-channel weight, so it
is ``None`` on every inspected serve) keeps the stock merged call.

Environment:

- ``TESSERA_GLM53_KDA_CONV_SPLIT``: ``on`` or ``off`` (default ``off`` until a
  served A/B lands).
- ``TESSERA_GLM53_SP_MHC``: ``off`` (default: the stock forward), ``auto``
  (measured ``T*``; research, see above), or ``force`` (SP at every token
  count where it is exact; a measurement arm, logged as such).  SP is always
  exact; there is no inexact mode.
- ``TESSERA_GLM53_SP_MHC_SPEC=1``: allow SP with speculative decoding (the MTP
  arm).  Without it a serve with a speculative config declines until an MTP
  row shows tolerance and acceptance hold.
- ``TESSERA_GLM53_ONORM_CUDA``: ``1`` appends the op as above; ``0`` (the
  default) leaves ``custom_ops`` as the serve set it.

Decline (stock behaviour, one warning): a touched module's sha256 is not an
inspected interface's, or TP != 2, PP > 1, DP > 1, EP, sequence-parallel MoE
already on, decode or prefill context parallelism, mHC off, or speculative
decoding without the opt-in.  A recognized interface whose objects do not look
as inspected (``o_proj`` not reducing, a MoE whose output is already reduced
or whose final reduction is already skipped, a zero-expert MoE) also declines:
each layer is checked, before anything on it changes, on its first forward in
vLLM's profile run, which never takes SP; a layer that fails runs the stock
forward and no later pass takes SP.
"""
from __future__ import annotations

# vLLM is optional and absent from the device-less development interpreter.
# pyright: reportMissingImports=false

import contextlib
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Callable

from .stock_interface import InspectedInterface as _Interface
from .stock_interface import import_modules, match_modules, module_digest as _sha256, stock_attribute

_log = logging.getLogger(__name__)

ONORM_OP = "fused_rms_norm_gated"
GLM5NEXT_ARCHITECTURES = frozenset({"Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"})

#: The modules the SP rebind reads or depends on, in ``_Interface.digests`` order.
SP_MODULES = (
    "vllm.models.glm5next.common.model",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner",
    "vllm.model_executor.layers.linear",
    "vllm.models.common.ops.sequence_parallel",
    "vllm.distributed.communication_op",
    # Exact SP: the split rule and the dispatch that reads it at call time.
    "vllm.model_executor.kernels.mhc.tilelang_kernels",
    "vllm.model_executor.kernels.mhc.tilelang",
    "vllm.model_executor.layers.mhc",
    "vllm.utils.deep_gemm",
    # SP's selected collective must have an active writer, not merely an object.
    "vllm.distributed.device_communicators.cuda_communicator",
    "vllm.distributed.device_communicators.all_reduce_utils",
    "vllm.distributed.device_communicators.pynccl",
    "vllm.distributed.parallel_state",
    "vllm.utils.torch_utils",
)

#: The token tile both pinned ``compute_num_split`` call sites pass as its grid,
#: ``cdiv(num_tokens, 64)``: ``_hc_prenorm_gemm_outputs`` (``kernels/mhc/tilelang.py``)
#: and the warmup compile key (``kernels/mhc/tilelang_kernels.py``).  Re-check it
#: whenever a digest in ``_INTERFACES`` changes.
PRENORM_BLOCK_M = 64


#: sha256 of each module in SP_MODULES order, read inside
#: localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a
#: (vLLM 0.30.1rc1.dev336+gaf5b4857e, nightly-20260929).
_INTERFACES = (
    _Interface("nightly-20260929", (
        "9ad4952e048bdec4991327e2877fae8d1385743a371bf69a2de90610d4d3b68e",
        "2c6e1fffba1f4fd2e4c5d3981c4d26f7c0de802fc9c35865748b113c30235da5",
        "7a9b90937865fa35d955ffc2017f23d3997bc4fba954e2667870b1b4854a38e0",
        "699fde98360fc8d604055e2987d93108ee489e87cc5563949d2a3dcd838eb8bd",
        "3c3f8ac60db38ece891d39dc1e2f6f1947ad438685893d2e5777d072320bd1b8",
        "61a384a091344d8207a04b2b7c74d30a81626eec4c068cf1dd034699cbfc34b5",
        "204e19209db5b19eb3dd25ea6ae10c84425dee02ef5bc896eca3f83da6149a8e",
        "433d750a4d669ad468679b1faf0a3f1e846410198fb4a1eeee2d14cfe55e6a55",
        "3c6652c4fcf2a6aa2e8c95d52056c2d86355a057d1234715b2d4b14c052dfc61",
        "2a0695d8b46757be83b38fe657df605ba0bfb2ef2dfa690c78c57419bbad0762",
        "b41e3ab17e21d81c7516cbef29850b0096ad3efd73d23538300af652ffce42d6",
        "4c51becc2ebfd41910526b93d7bd8e9e36312df444fdfb55fa2fbc92b0f8f88e",
        "a33fc846e0f4e682a644ce712d862806ecfa12dca6e5f641806e2e7991cd0087",
        "eab9ea0a3d3b9792fd7e3076cca1a3484031b22ce662baa7709cb5a68dc2115f",
    )),
)

#: Token counts the ``T*`` measurement visits (capped at max_num_batched_tokens).
T_GRID = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_WARMUP, _REPS = 3, 7

_INSTALL_LOCK = threading.Lock()
#: The configs this hook appended the op to, by id.  Each is held, so a dead config's id
#: cannot pass to a new config and read as this hook's decision.
_ONORM_DONE: dict[int, Any] = {}


# --------------------------------------------------------------------------- config


def _text_config(config: Any) -> Any:
    hf = getattr(getattr(config, "model_config", None), "hf_config", None)
    return getattr(hf, "text_config", None) or hf


def is_glm5next(config: Any) -> bool:
    hf = getattr(getattr(config, "model_config", None), "hf_config", None)
    archs = set(getattr(hf, "architectures", None) or ())
    return bool(archs & GLM5NEXT_ARCHITECTURES)


def onorm_cuda_enabled() -> bool:
    """``TESSERA_GLM53_ONORM_CUDA``: ``1`` opts in; ``0`` or unset (the default) leaves the serve's ops."""
    raw = os.environ.get("TESSERA_GLM53_ONORM_CUDA", "0").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"TESSERA_GLM53_ONORM_CUDA must be 0 or 1; got {raw!r}")
    return raw == "1"


def enable_onorm_cuda(config: Any) -> bool:
    """Append ``+fused_rms_norm_gated`` for a Glm5Next serve that did not name the op.

    Returns True when this call (or an earlier one on the same config) enabled it.
    Must run before the first ``FusedRMSNormGated`` is constructed: the op picks
    its forward at construction.
    """
    if config is None or not is_glm5next(config):
        return False
    if not onorm_cuda_enabled():
        return False
    compilation = getattr(config, "compilation_config", None)
    ops = getattr(compilation, "custom_ops", None)
    if not isinstance(ops, list):
        return False
    if _ONORM_DONE.get(id(config)) is config:
        return True
    if f"+{ONORM_OP}" in ops or f"-{ONORM_OP}" in ops:
        return False  # the serve decided; respect it either way
    ops.append(f"+{ONORM_OP}")
    _ONORM_DONE[id(config)] = config
    _log.warning("tessera.glm53_prefill: custom_ops += +%s (Glm5Next runs eager; the KDA "
                 "output norm takes vLLM's forward_cuda instead of its eager decomposition)",
                 ONORM_OP)
    return True


def sp_mode() -> str:
    mode = os.environ.get("TESSERA_GLM53_SP_MHC", "off").strip().lower()
    if mode not in ("auto", "off", "force"):
        raise ValueError(f"TESSERA_GLM53_SP_MHC must be auto, off or force; got {mode!r}")
    return mode


def sp_decline_reasons(config: Any) -> list[str]:
    """Why this serve must keep the stock layer forward; empty when eligible."""
    reasons = []
    if config is None:
        return ["no current vLLM config"]
    if not is_glm5next(config):
        return ["not a Glm5Next model"]
    par = getattr(config, "parallel_config", None)

    def p(name, default=None):
        return getattr(par, name, default)

    if p("tensor_parallel_size") != 2:
        reasons.append(f"tensor_parallel_size {p('tensor_parallel_size')} (measured at 2 only)")
    if p("pipeline_parallel_size", 1) != 1:
        reasons.append(f"pipeline_parallel_size {p('pipeline_parallel_size')}")
    if p("data_parallel_size", 1) != 1:
        reasons.append(f"data_parallel_size {p('data_parallel_size')}")
    if p("enable_expert_parallel", False):
        reasons.append("expert parallelism on")
    if p("use_sequence_parallel_moe", False):
        reasons.append("vLLM's own sequence-parallel MoE is on")
    for name in ("decode_context_parallel_size", "prefill_context_parallel_size"):
        if p(name, 1) not in (None, 1):
            reasons.append(f"{name} {p(name)}")
    text = _text_config(config)
    hf = getattr(getattr(config, "model_config", None), "hf_config", None)
    if not (getattr(text, "mhc", None) or getattr(hf, "mhc", None)):
        reasons.append("mHC off in the model config")
    if getattr(config, "speculative_config", None) is not None and \
            os.environ.get("TESSERA_GLM53_SP_MHC_SPEC", "0") != "1":
        reasons.append("speculative decoding on (TESSERA_GLM53_SP_MHC_SPEC=1 opts in)")
    return reasons


# ------------------------------------------------------------------ source identity


def _match(modules: tuple[Any, ...], names: tuple[str, ...],
           interfaces: tuple[_Interface, ...]) -> tuple[_Interface | None, str]:
    return match_modules(modules,names,interfaces)


def _match_interface(modules: tuple[Any, ...]) -> tuple[_Interface | None, str]:
    return _match(modules, SP_MODULES, _INTERFACES)


def _import_all(names: tuple[str, ...] = SP_MODULES) -> tuple[tuple[Any, ...] | None, str]:
    return import_modules(names)


# ------------------------------------------------------------------------ the forward


class SplitForcer:
    """Stand-in for ``tilelang_kernels.compute_num_split``: stock, except inside :meth:`full_batch`.

    Inside ``full_batch(tokens)`` every call on this thread answers for the
    ``tokens``-token grid (``cdiv(tokens, PRENORM_BLOCK_M)``) instead of the
    caller's own, so an SP rank's mHC call on its shard runs the pre-norm GEMM
    at the split the full batch's call takes.  Outside it, and on every other
    thread, the stock (cached) rule answers unchanged.
    """

    def __init__(self, stock: Callable[[int, int | None, int], int]):
        self.stock = stock
        self._local = threading.local()

    def __call__(self, block_k: int, k: int | None, grid_size: int) -> int:
        tokens = getattr(self._local, "tokens", None)
        if tokens is not None:
            grid_size = -(-tokens // PRENORM_BLOCK_M)
        return self.stock(block_k, k, grid_size)

    @contextlib.contextmanager
    def full_batch(self, tokens: int):
        previous = getattr(self._local, "tokens", None)
        self._local.tokens = int(tokens)
        try:
            yield
        finally:
            self._local.tokens = previous


_NO_FORCE = contextlib.nullcontext()


class SpState:
    """Per-process SP state: the measured threshold, its table, and the per-pass decision.

    SP is decided once per model pass, at its first layer, so every layer of a
    pass agrees.  No pass takes SP before one complete pass has prepared every
    layer it ran (vLLM's profile run): a layer that cannot be prepared then
    declines the whole serve to stock before any activation was sharded.
    """

    def __init__(self, mode: str, max_tokens: int, tp_size: int):
        self.mode = mode
        self.max_tokens = int(max_tokens)
        self.tp_size = int(tp_size)
        self.t_star: float | None = float(tp_size) if mode == "force" else None
        self.table: list[dict] = []
        self.lock = threading.Lock()
        self.declined: str | None = None  # why a layer could not be prepared
        self.ready = False                # a complete pass prepared every layer it ran
        self.pass_sp = False              # the decision for the pass in flight
        self._pass_open = False
        self._pass_ok = True

    def use_sp(self, num_tokens: int) -> bool:
        return self.t_star is not None and num_tokens >= self.t_star

    def begin_pass(self, num_tokens: int, capturing: bool, *, exact: bool) -> bool:
        """At a pass's first layer: settle the previous pass, then decide this one.

        ``exact``: the full batch and its shard both reach the split-k path, so
        the shard's mHC calls can run at the full batch's split.  A pass that
        would not be exact runs stock, whatever ``T*`` is.
        """
        if self._pass_open and self._pass_ok and self.declined is None and not self.ready:
            self.ready = True
            _log.warning("tessera.glm53_prefill: SP mHC armed (T*=%s, mode %s)", self.t_star, self.mode)
        self._pass_open, self._pass_ok = True, True
        # A captured graph always holds the stock op sequence, whatever T* is.
        self.pass_sp = (self.ready and self.declined is None and not capturing
                        and exact and self.use_sp(num_tokens))
        return self.pass_sp

    def prepare(self, layer: Any) -> bool:
        """Prepare ``layer``; False (and the serve declines to stock) when it cannot be."""
        if layer.__dict__.get("_tessera_sp_ready"):
            return True
        try:
            prepare_layer(layer)
        except RuntimeError as exc:
            if self.pass_sp:
                raise  # activations of this pass are already sharded: no stock path back
            self._pass_ok = False
            if self.declined is None:
                self.declined = str(exc)
                _log.warning("tessera.glm53_prefill: %s; SP mHC declines to stock for this serve", exc)
            return False
        return True

    def wants_measurement(self) -> bool:
        """Measure on the first non-capturing pass, whatever its size (the timings use their own tensors)."""
        return self.mode == "auto" and self.t_star is None


def choose_t_star(rows: list[dict]) -> float:
    """Smallest grid count from which the saving beats the cost at every larger count."""
    t_star = math.inf
    for row in sorted(rows, key=lambda r: -r["tokens"]):
        if row["saving_ms_per_layer"] > row["cost_ms_per_layer"]:
            t_star = row["tokens"]
        else:
            break
    return t_star


def prepare_layer(layer: Any) -> None:
    """Turn off the two reductions the rebound forward now performs itself.

    Runs on a layer's first forward (vLLM's profile run, before any capture).
    Raises ``RuntimeError``, before changing anything, when the objects are not
    the inspected ones; :meth:`SpState.prepare` turns that into a decline.
    """
    if layer.__dict__.get("_tessera_sp_ready"):
        return
    o_proj = layer.self_attn.o_proj
    if getattr(o_proj, "reduce_results", None) is not True:
        raise RuntimeError(f"tessera SP mHC: layer {layer.layer_idx} o_proj.reduce_results is "
                           f"{getattr(o_proj, 'reduce_results', None)!r}, expected True")
    if layer._mlp_is_moe:
        runner = layer.mlp.experts
        cfg = runner.moe_config
        if getattr(cfg, "skip_final_all_reduce", None) is not False:
            raise RuntimeError(f"tessera SP mHC: layer {layer.layer_idx} MoE final all-reduce "
                               f"already skipped ({getattr(cfg, 'skip_final_all_reduce', None)!r})")
        if getattr(runner, "_fused_output_is_reduced", True):
            raise RuntimeError(f"tessera SP mHC: layer {layer.layer_idx} MoE kernel reduces its "
                               "own output; the late all-reduce is not the one skipped")
        if getattr(runner, "zero_expert_type", None) is not None or \
                getattr(cfg, "is_sequence_parallel", False) or \
                getattr(getattr(cfg, "moe_parallel_config", None), "use_all2all_kernels", False):
            raise RuntimeError(f"tessera SP mHC: layer {layer.layer_idx} MoE parallel layout "
                               "is not the inspected TP-sharded one")
        cfg.skip_final_all_reduce = True
    else:
        down = layer.mlp.down_proj
        if getattr(down, "reduce_results", None) is not True:
            raise RuntimeError(f"tessera SP mHC: layer {layer.layer_idx} mlp.down_proj."
                               f"reduce_results is {getattr(down, 'reduce_results', None)!r}")
        down.reduce_results = False
    o_proj.reduce_results = False
    layer._tessera_sp_ready = True


def _median_ms(fn: Callable[[], Any], torch: Any) -> float:
    for _ in range(_WARMUP):
        fn()
    times = []
    for _ in range(_REPS):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    times.sort()
    return times[len(times) // 2]


def measure_t_star(layer: Any, state: SpState, ops: Any, torch: Any, device: Any) -> None:
    """Fill ``state.table`` and ``state.t_star`` from timings on this serve (see module doc)."""
    hidden, n = layer.hidden_size, layer.n
    rows = []
    for t in T_GRID:
        if t > state.max_tokens or t < 2 * state.tp_size:
            continue
        if not ops.sp_exact(t, hidden, n):
            continue  # no pass of this size takes SP (the shard would take the fused small-batch kernel)
        half = -(-t // state.tp_size)

        def mhc(tokens):
            x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
            res = torch.randn(tokens, n, hidden, device=device, dtype=torch.bfloat16)
            post = torch.full((tokens, n, 1), 1.0, device=device, dtype=torch.float32)
            comb = torch.full((tokens, n, n), 1.0 / n, device=device, dtype=torch.float32)
            return lambda: layer.hc_fused_post_pre(
                x, res, post, comb, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                norm_weight=layer.post_attention_layernorm.weight.data,
                norm_eps=layer.post_attention_layernorm.variance_epsilon)

        full = torch.randn(t, hidden, device=device, dtype=torch.bfloat16)
        shard = full[:half].contiguous()
        m_full = _median_ms(mhc(t), torch)
        with ops.full_split(t):  # the shard's call as an SP pass runs it: at the full batch's split
            m_half = _median_ms(mhc(half), torch)
        ar = _median_ms(lambda: ops.all_reduce(full), torch)
        ag = _median_ms(lambda: ops.sp_all_gather(shard), torch)
        rs = _median_ms(lambda: ops.sp_reduce_scatter(full), torch)
        rows.append({"tokens": t, "mhc_ms": m_full, "mhc_half_ms": m_half, "all_reduce_ms": ar,
                     "all_gather_ms": ag, "reduce_scatter_ms": rs,
                     "saving_ms_per_layer": 2 * (m_full - m_half),
                     "cost_ms_per_layer": 2 * (ag + rs - ar)})
    local = choose_t_star(rows)
    agreed = ops.max_across_tp(local)
    state.table = rows
    state.t_star = agreed
    _log.warning("tessera.glm53_prefill: SP mHC threshold T*=%s (this rank measured %s) from %s",
                 agreed, local, rows)


def make_forward(stock_forward: Callable, ops: Any, state: SpState, torch: Any) -> Callable:
    """The rebound ``Glm5NextDecoderLayer.forward`` (stock line for line except the collectives)."""

    def forward(self, positions, hidden_states, residual=None, post=None, comb=None):
        if not self.mhc or self.is_mtp_layer:
            return stock_forward(self, positions, hidden_states, residual, post, comb)
        num_tokens = positions.shape[0]
        if post is None and self.layer_idx == 0:
            capturing = torch.cuda.is_current_stream_capturing()
            available = not capturing and ops.sp_available()
            if not capturing and not available:
                raise RuntimeError("SP collective unavailable before activation")
            if state.wants_measurement() and available and state.declined is None:
                with state.lock:
                    if state.t_star is None:
                        measure_t_star(self, state, ops, torch, hidden_states.device)
            state.begin_pass(num_tokens, capturing,
                             exact=available and ops.sp_exact(num_tokens, self.hidden_size, self.n))
        if not state.prepare(self):
            return stock_forward(self, positions, hidden_states, residual, post, comb)
        sp = state.pass_sp
        # On an SP pass the shard's mHC calls run at the full batch's pre-norm split (exact SP).
        # (a factory: a generator context manager is single-use, and a layer enters it twice).
        split = (lambda: ops.full_split(num_tokens)) if sp else (lambda: _NO_FORCE)

        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                if sp:
                    x = ops.sp_shard(x)
                x = ops.hc_expand(x, self.n)
            residual = x
            with split():
                post, comb, x = self.hc_pre(
                    x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                    norm_weight=self.input_layernorm.weight.data,
                    norm_eps=self.input_layernorm.variance_epsilon)
        else:
            with split():
                residual, post, comb, x = self.hc_fused_post_pre(
                    x, residual, post, comb, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                    norm_weight=self.input_layernorm.weight.data,
                    norm_eps=self.input_layernorm.variance_epsilon)

        if sp:
            x = ops.sp_all_gather(x)[:num_tokens]
        x = self.self_attn(hidden_states=x, positions=positions)
        x = ops.sp_reduce_scatter(x) if sp else ops.all_reduce(x)

        with split():
            residual, post, comb, x = self.hc_fused_post_pre(
                x, residual, post, comb, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
                norm_weight=self.post_attention_layernorm.weight.data,
                norm_eps=self.post_attention_layernorm.variance_epsilon)

        if sp:
            x = ops.sp_all_gather(x)[:num_tokens]
        x = self.mlp(x)
        x = ops.sp_reduce_scatter(x) if sp else ops.all_reduce(x)

        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = ops.hc_contract(x, self.n)
            if sp:
                x = ops.sp_all_gather(x)[:num_tokens]
            return x, None, None, None
        return x, residual, post, comb

    forward._tessera_sp_mhc = True  # type: ignore[attr-defined]
    return forward


def install_split_forcer(kernels: Any) -> SplitForcer:
    """Put a :class:`SplitForcer` in place of ``kernels.compute_num_split`` (once per process).

    ``_hc_prenorm_gemm_outputs`` imports the name from the module on every call,
    so the module attribute is the one every runtime caller reads.
    """
    current = kernels.compute_num_split
    if isinstance(current, SplitForcer):
        return current
    forcer = SplitForcer(current)
    kernels.compute_num_split = forcer
    return forcer


def shard_split_exact(kernels: Any, deep_gemm: Any, tp_size: int, num_tokens: int,
                      hidden: int, n: int) -> bool:
    """True when the full batch and this rank's shard both run the split-k pre-norm GEMM.

    That is the DeepGEMM path, with vLLM's fused small-batch kernel declined for
    both token counts by vLLM's own dispatch rule.  Only there does the split
    the shard's call takes come from ``compute_num_split``, so only there can
    :class:`SplitForcer` make it the full batch's.
    """
    if not deep_gemm.is_deep_gemm_supported():
        return False
    shard = -(-num_tokens // tp_size)
    return all(kernels.mhc_fused_post_pre_split_config(t, hidden, n) is None
               for t in (num_tokens, shard))


def sp_collective_decline(dc: Any, *, symmetric_ag_rs: Any) -> str | None:
    """Require the pinned ordinary SP route's active TP2 PyNccl writer.

    Custom/symmetric SP routes are outside this inspected route's qualification.
    PyNccl's disabled methods are no-ops; suspended communicators cannot be used
    before their collective resume. Unknown state is unavailable.
    """
    if dc is None or stock_attribute(dc, "world_size") != 2:
        return "no TP2 device communicator"
    if stock_attribute(dc, "ca_comm", object()) is not None:
        return "custom SP communicator present or unknown"
    if type(symmetric_ag_rs) is not bool or symmetric_ag_rs:
        return "symmetric-memory SP enabled or unknown"
    nccl = stock_attribute(dc, "pynccl_comm")
    if (nccl is None or stock_attribute(nccl, "world_size") != 2
            or stock_attribute(nccl, "available") is not True
            or stock_attribute(nccl, "disabled") is not False
            or stock_attribute(nccl, "_suspended") is not False):
        return "PyNccl writer missing, disabled, suspended or unknown"
    if any(not callable(stock_attribute(nccl, name)) for name in ("reduce_scatter", "all_gather")):
        return "PyNccl SP methods unavailable"
    return None


class SpCollectiveGuard:
    """Agree availability on the existing TP CPU group before each SP boundary.

    The control group remains usable when the selected PyNccl writer is
    disabled. A pass refuses before sharding; a transient fault after that
    refuses both ranks before either enters a mismatched/no-op collective.
    Capture uses stock all-reduces and never enters the SP control exchange.
    The active writer's inputs, outputs and NCCL operation order are unchanged.
    """
    def __init__(self, *, route: Callable, agree: Callable, capturing: Callable):
        self.route, self.agree, self.capturing = route, agree, capturing

    def available(self) -> bool:
        if self.capturing():
            return False
        try:
            why = self.route()
        except (AttributeError, OSError, TypeError, ValueError, AssertionError):
            why = "unavailable communicator facts"
        return self.agree(float(why is not None)) == 0

    def call(self, fn: Callable, x: Any) -> Any:
        if self.capturing():
            raise RuntimeError("SP collective is unavailable under graph capture")
        if not self.available():
            raise RuntimeError("SP collective unavailable before writer enqueue")
        return fn(x)


def _vllm_ops(modules: tuple[Any, ...], tp_size: int) -> Any:
    (model, _runner, _linear, seqpar, comm, kernels, _tilelang, _mhc_ops, deep_gemm,
     _cuda_comm, ar_utils, _pynccl, _parallel_state, _torch_utils) = modules
    import torch

    from vllm.distributed import get_tp_group

    forcer = install_split_forcer(kernels)

    def route():
        return sp_collective_decline(get_tp_group().device_communicator,
                                     symmetric_ag_rs=ar_utils.should_nccl_symm_mem_ag_rs())

    def agree(value: float) -> float:
        # A CPU control exchange must not call the disabled writer it checks,
        # introduce a CUDA capture node, or change the SP NCCL reduction order.
        control = torch.tensor([int(value)], dtype=torch.int32)
        torch.distributed.all_reduce(control, op=torch.distributed.ReduceOp.MAX,
                                     group=get_tp_group().cpu_group)
        return float(control.item())

    guard = SpCollectiveGuard(route=route, agree=agree,
                              capturing=torch.cuda.is_current_stream_capturing)

    def sp_exact(num_tokens: int, hidden: int, n: int) -> bool:
        return shard_split_exact(kernels, deep_gemm, tp_size, num_tokens, hidden, n)

    def max_across_tp(value: float) -> float:
        group = get_tp_group()
        big = float(2 ** 62)
        t = torch.tensor([big if math.isinf(value) else float(value)], dtype=torch.float64,
                         device=f"cuda:{torch.cuda.current_device()}")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group.device_group)
        out = float(t.item())
        return math.inf if out >= big else out

    return SimpleNamespace(
        sp_shard=seqpar.sp_shard,
        sp_all_gather=lambda x: guard.call(seqpar.sp_all_gather, x),
        sp_reduce_scatter=lambda x: guard.call(seqpar.sp_reduce_scatter, x),
        all_reduce=comm.tensor_model_parallel_all_reduce,
        hc_expand=model.hc_expand, hc_contract=model.hc_contract,
        max_across_tp=max_across_tp, sp_exact=sp_exact, sp_available=guard.available,
        full_split=forcer.full_batch)


_INSTALLED: dict[str, Any] = {}


def install_sp_mhc(config: Any) -> bool:
    """Rebind the layer forward when this serve is the inspected one; True when active."""
    mode = sp_mode()
    if mode == "off":
        return False
    decided = _INSTALLED.get(("decided", id(config)))
    if decided is not None:
        return decided
    active = _install_sp_mhc(config, mode)
    _INSTALLED[("decided", id(config))] = active
    return active


def _install_sp_mhc(config: Any, mode: str) -> bool:
    reasons = sp_decline_reasons(config)
    if not reasons:
        modules, why = _import_all()
        if modules is None:
            reasons = [why]
        else:
            interface, why = _match_interface(modules)
            if interface is None:
                reasons = [why]
    if reasons:
        _log.warning("tessera.glm53_prefill: SP mHC declined, stock layer forward: %s",
                     "; ".join(reasons))
        return False
    model = modules[0]
    layer_cls = model.Glm5NextDecoderLayer
    if getattr(layer_cls.forward, "_tessera_sp_mhc", False):
        return True
    import torch

    sched = getattr(config, "scheduler_config", None)
    state = SpState(mode, getattr(sched, "max_num_batched_tokens", T_GRID[-1]),
                    config.parallel_config.tensor_parallel_size)
    layer_cls.forward = make_forward(layer_cls.forward, _vllm_ops(modules, state.tp_size), state, torch)
    _INSTALLED["state"] = state
    _log.warning("tessera.glm53_prefill: SP mHC installed (interface %s, mode %s, "
                 "max_num_batched_tokens %s; exact: the shard's mHC runs at the full batch's "
                 "pre-norm split)", interface.name, mode, state.max_tokens)
    return True


# ------------------------------------------------------------- KDA conv per slice

#: The modules the KDA conv-split rebind reads or depends on, in digest order: the layer whose
#: ``_forward`` is recompiled, and the conv whose wrapper and kernel must honour slice strides.
KDA_MODULES = (
    "vllm.models.glm5next.common.kda",
    "vllm.model_executor.layers.mamba.ops.causal_conv1d",
)

#: sha256 of each module in KDA_MODULES order, read inside the image named at _INTERFACES.
_KDA_INTERFACES = (
    _Interface("nightly-20260929", (
        "efd6fdca3110176e7e4aa8b5a7543b6840690b3d4c01dee48d45d246611b4eea",
        "cb16cc9250c4195c09d6d83b43bba607c2d749b621d8d135a920443d1577265f",
    )),
)

KDA_CLASS, KDA_METHOD, KDA_DECORATOR = "Glm5NextLinearAttention", "_forward", "eager_break_during_capture"

#: The stock prefill conv block of ``Glm5NextLinearAttention._forward``, verbatim with its
#: indentation; it must occur exactly once in the method's source.
KDA_STOCK_CONV_BLOCK = """\
            qkv_ns = causal_conv1d_fn(
                qkv_ns.transpose(0, 1),
                conv_weights,
                conv_bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)
"""

KDA_SPLIT_CONV_BLOCK = """\
            q_ns, k_ns, v_ns = _tessera_glm53_conv_split(
                causal_conv1d_fn,
                qkv_ns,
                conv_weights,
                conv_bias,
                conv_state,
                has_initial_state,
                non_spec_state_indices_tensor,
                non_spec_query_start_loc,
                attn_metadata_narrowed,
                self.local_projection_size,
            )
"""


def kda_conv_split_mode() -> str:
    mode = os.environ.get("TESSERA_GLM53_KDA_CONV_SPLIT", "off").strip().lower()
    if mode not in ("on", "off"):
        raise ValueError(f"TESSERA_GLM53_KDA_CONV_SPLIT must be on or off; got {mode!r}")
    return mode


def conv_split(conv_fn: Callable, qkv: Any, weight: Any, bias: Any, conv_state: Any,
               has_initial_state: Any, cache_indices: Any, query_start_loc: Any, metadata: Any,
               p: int) -> tuple[Any, Any, Any]:
    """The KDA prefill conv once per q/k/v slice: three dense token-major [T, p] outputs.

    ``qkv`` is the merged [T, 3p] projection.  Each call sees a channel-last (p, T) slice of
    it, the matching rows of the merged weight and the matching channels of ``conv_state``
    (which it updates in place, as the merged call does), and allocates its own output with
    ``empty_like``, which on that slice is dense token-major.  With a conv bias, or a shape
    that is not 3p channels, this is the stock merged call and split.
    """
    x = qkv.transpose(0, 1)
    if bias is not None or x.shape[0] != 3 * p or weight.shape[0] != 3 * p or conv_state.shape[1] != 3 * p:
        out = conv_fn(x, weight, bias, activation="silu", conv_states=conv_state,
                      has_initial_state=has_initial_state, cache_indices=cache_indices,
                      query_start_loc=query_start_loc, metadata=metadata).transpose(0, 1)
        return out.split(p, dim=-1)
    outs = []
    for i in range(3):
        sl = slice(i * p, (i + 1) * p)
        outs.append(conv_fn(x[sl], weight[sl], None, activation="silu", conv_states=conv_state[:, sl],
                            has_initial_state=has_initial_state, cache_indices=cache_indices,
                            query_start_loc=query_start_loc, metadata=metadata).transpose(0, 1))
    return outs[0], outs[1], outs[2]


def recompile_kda_forward(module: Any) -> tuple[Callable | None, str]:
    """``KDA_CLASS.KDA_METHOD`` compiled from ``module``'s source with the conv block replaced.

    Returns ``(function, "")`` with the stock decorator applied, or ``(None, why)``.  Reads the
    file the digest check covered; the function's globals are a copy of the module's plus the
    helper, so nothing is added to vLLM's namespace.  :func:`.method_rebuild.rebuild_method`
    compiles it (see that module for why the read and the compile live apart).
    """
    from .method_rebuild import rebuild_method

    path = getattr(module, "__file__", None)
    try:
        src = Path(path).read_text() if path else None
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"
    if src is None:
        return None, "module has no source file"
    ns = dict(vars(module))
    ns["_tessera_glm53_conv_split"] = conv_split
    method, why = rebuild_method(src, ns, KDA_CLASS, KDA_METHOD, KDA_DECORATOR, KDA_STOCK_CONV_BLOCK,
                                 KDA_SPLIT_CONV_BLOCK,
                                 f"<tessera glm53_prefill: {KDA_CLASS}.{KDA_METHOD} conv split>",
                                 block_name="the stock conv block")
    if method is None:
        return None, why
    decorator = getattr(module, KDA_DECORATOR, None)
    if decorator is None:
        return None, f"{KDA_DECORATOR} not in {module.__name__}"
    new = decorator(method)
    new._tessera_kda_conv_split = True  # type: ignore[attr-defined]
    return new, ""


def install_kda_conv_split(config: Any) -> bool:
    """Rebind ``Glm5NextLinearAttention._forward`` when this serve is the inspected one."""
    if kda_conv_split_mode() == "off":
        return False
    decided = _INSTALLED.get(("kda", id(config)))
    if decided is not None:
        return decided
    _INSTALLED[("kda", id(config))] = active = _install_kda_conv_split(config)
    return active


def _install_kda_conv_split(config: Any) -> bool:
    reasons = [] if config is not None and is_glm5next(config) else ["not a Glm5Next model"]
    modules = None
    if not reasons:
        modules, why = _import_all(KDA_MODULES)
        if modules is None:
            reasons = [why]
        else:
            interface, why = _match(modules, KDA_MODULES, _KDA_INTERFACES)
            if interface is None:
                reasons = [why]
    if not reasons:
        cls = getattr(modules[0], KDA_CLASS, None)
        if cls is None:
            reasons = [f"{KDA_CLASS} not in {KDA_MODULES[0]}"]
        elif getattr(getattr(cls, KDA_METHOD, None), "_tessera_kda_conv_split", False):
            return True
        else:
            new, why = recompile_kda_forward(modules[0])
            if new is None:
                reasons = [why]
    if reasons:
        _log.warning("tessera.glm53_prefill: KDA conv split declined, stock %s.%s: %s",
                      KDA_CLASS, KDA_METHOD, "; ".join(reasons))
        return False
    setattr(cls, KDA_METHOD, new)
    _log.warning("tessera.glm53_prefill: KDA conv split installed (interface %s): the prefill conv "
                 "runs per q/k/v slice, so FlashKDA's q/k/v copies are no-ops", interface.name)
    return True


def install_for_current_config() -> None:
    """Called from ``TesseraConfig.get_quant_method`` during model construction."""
    try:
        from vllm import config as vllm_config
    except Exception:  # noqa: BLE001 - no vLLM, nothing to install
        return
    getter = getattr(vllm_config, "get_current_vllm_config_or_none", None)
    current = getter() if getter is not None else None
    if current is None:
        return
    with _INSTALL_LOCK:
        enable_onorm_cuda(current)
        install_sp_mhc(current)
        install_kda_conv_split(current)
