"""The native window MoE adapter: two grouped window GEMMs around the
activation, for the FP8 and BF16 routed-expert families.

WHAT IT IS.  ``PreparedGroupedWindowGemm`` (``window_gemm_grouped``) computes
one projection under a token->expert routing.  A routed MoE layer is two of
them -- gate/up, then down -- with the activation between, and that is what
this module owns:

* ``gate/up`` runs in ``preserve=True`` mode: per-route outputs
  ``[T, top_k, 2I]`` (or two ``[T, top_k, I]`` stacks if gate and up are
  separate).  The routing weight multiplies this projection's fp32
  accumulator iff ``apply_router_weight_on_input``, which is the legacy
  ``fused_moe.py`` placement.  The MODULAR path that serves instead applies the
  weight in ``prepare`` -- to the hidden states, BEFORE the activation
  quantization (``prepare_finalize/no_dp_ep.py``) -- and that is what this
  adapter now reproduces, so both are named here rather than conflated.
* the activation is applied per route.  ``silu`` is served, with the model's
  SwiGLU clamp (``swiglu_limit`` / vLLM's ``gemm1_clamp_limit``) reproduced by
  clamping both branches in fp32 before the activation -- the gate saturates at
  ``+limit`` and the up branch at ``+-limit``, the arithmetic vLLM's own
  ``silu_and_mul``/``apply_moe_activation`` performs.  ``swiglu_alpha``,
  ``swiglu_beta`` and another activation **fail closed** until an
  implementation reproduces vLLM's exact arithmetic -- no silent
  substitution.
* ``down`` runs with ``route_input=True``: it consumes the per-route
  activations, and weights each route's fp32 accumulator iff the input did
  not already carry the weights (vLLM's gemm2 / ``topk_weight_and_reduce``
  placement), summing over ``top_k``.

ARITHMETIC.  The weight-side contract is the grouped bundle's and is explicit
at preparation: ``"epilogue"`` (dense BF16 row scale on the fp32 accumulator,
or the FP8 ``acc * a_scale * w_scale`` contract) or ``"folded"`` (the
research BF16 contract, one bf16 rounding of ``value * row_scale`` in
registers before ``tl.dot``).  The adapter inherits it from the bundles; the
FP8 family has no folded form.

WHAT REMAINS THE OWNER'S.  Routing (top-k ids and weights), shared experts,
the TP all-reduce and the final scale/dtype presentation are the serving
layer's; this adapter returns the routed-expert result ``[T, rows]`` bf16.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch

from .errors import GrammarError
from .kernel_window_gemv import WindowGemvUnit
from .window_gemm_grouped import PreparedGroupedWindowGemm, prepare_grouped_window_gemm

__all__ = ["NativeWindowMoE", "prepare_native_window_moe", "PackedWindowUnits",
           "SUPPORTED_ACTIVATIONS", "checked_swiglu_limit"]

#: The activations this adapter reproduces exactly.  Everything else refuses.
SUPPORTED_ACTIVATIONS = ("silu",)


@dataclasses.dataclass(frozen=True)
class PackedWindowUnits:
    """Rank-local per-expert units in GLOBAL expert order, one entry per
    expert: what a loader hands the adapter without a second weight walk."""

    gate: tuple
    up: tuple
    down: tuple
    family: str

    @property
    def experts(self) -> int:
        return len(self.down)

    def resident_bytes(self) -> int:
        total = 0
        for unit in (*self.gate, *self.up, *self.down):
            rep = unit.rep
            total += rep.words.numel() * rep.words.element_size()
            for t in (unit.table, unit.scale, unit.initial_state, unit.codes_of_state, unit.native):
                if t is not None:
                    total += t.numel() * t.element_size()
        return total

    def prepare(self, *, block_m: int = 64, block_n: int = 64, block_k: int = 64,
                arithmetic: "str | None" = None, activation: str = "silu",
                quantizer: "str | None" = "native") -> "NativeWindowMoE":
        """Build the adapter.  The default arithmetic is each family's
        published contract: folded for the research BF16 wire, epilogue for
        the FP8 wire.  An explicit value must match the family's served
        contract; nothing here silently swaps them."""
        if arithmetic is None:
            arithmetic = "folded" if self.family == "value" else "epilogue"
        up = list(self.up) if self.up else None
        return prepare_native_window_moe(
            list(self.gate), list(self.down), up=up,
            block_m=block_m, block_n=block_n, block_k=block_k,
            quantizer=quantizer, arithmetic=arithmetic, activation=activation)


@dataclasses.dataclass(frozen=True)
class NativeWindowMoE:
    """Two grouped projections around the activation.  ``gate_up`` is the
    fused ``[2I]`` stack; ``gate``/``up`` are the separate-stack spelling."""

    gate_up: "PreparedGroupedWindowGemm | None"
    gate: "PreparedGroupedWindowGemm | None"
    up: "PreparedGroupedWindowGemm | None"
    down: PreparedGroupedWindowGemm
    activation: str = "silu"

    def __call__(self, x: torch.Tensor,
                 expert_ids: torch.Tensor,
                 routing_weights: torch.Tensor,
                 *,
                 apply_router_weight_on_input: bool = False,
                 swiglu_limit: "float | None" = None) -> torch.Tensor:
        limit = checked_swiglu_limit(swiglu_limit)
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise GrammarError(
                f"the native window MoE serves {SUPPORTED_ACTIVATIONS}; "
                f"activation {self.activation!r} has no exact implementation here "
                "and refusing beats approximating vLLM's arithmetic"
            )
        # ``apply_router_weight_on_input`` is the MODULAR kernel's prepare-time
        # placement, and that is a different operation from the legacy
        # monolithic one.  ``prepare_finalize/no_dp_ep.py`` multiplies the
        # HIDDEN STATES by ``topk_weights`` and only THEN calls
        # ``_quantize_input``: the route weight is applied to x BEFORE the
        # per-token FP8 activation quantization, so the activation scale is
        # computed from the SCALED magnitudes.  A lane that quantizes the
        # unscaled x and scales the gemm1 accumulator afterwards is quantizing
        # different numbers, which is a different function, not a rounding
        # difference.  Legacy ``fused_moe.py`` scales the accumulator; the
        # modular path is what serves, and this lane follows the modular one.
        #
        # vLLM's prepare asserts the placement is only implemented for topk=1;
        # refuse rather than silently serving something else.
        inter = self.down.cols
        if apply_router_weight_on_input:
            if int(expert_ids.shape[1]) != 1:
                raise GrammarError(
                    f"apply_router_weight_on_input is only implemented for topk=1 "
                    f"(vLLM's own prepare asserts this); got topk="
                    f"{int(expert_ids.shape[1])}"
                )
            # BEFORE the activation quantization, as the modular prepare does:
            # ``no_dp_ep.py`` computes ``a1 = a1 * topk_weights.to(a1.dtype)``
            # and only then quantizes.  The weight is narrowed to the ACTIVATION
            # dtype first (not left fp32 to promote the product) and the product
            # keeps that dtype, so the per-token scale is computed from exactly
            # the magnitudes the stock prepare feeds its quantizer.
            weights_in = routing_weights.reshape(-1, 1).to(x.dtype)
            x = x * weights_in
        # gemm1 consumes the ALREADY-SCALED activations: the weight is in x,
        # so neither gemm1 nor gemm2 may apply it again.
        if self.gate_up is not None:
            route = self.gate_up(x, expert_ids, routing_weights, preserve=True,
                                 apply_router_weight_on_input=False)
            if route.shape[-1] != 2 * inter:
                raise GrammarError(
                    f"the fused gate/up stack produces {route.shape[-1]} rows, "
                    f"the down stack consumes {inter} per half"
                )
            gate, up = route[..., :inter], route[..., inter:]
        else:
            gate = self.gate(x, expert_ids, routing_weights, preserve=True,
                             apply_router_weight_on_input=False)
            up = self.up(x, expert_ids, routing_weights, preserve=True,
                         apply_router_weight_on_input=False)
        act = _silu_and_mul(gate, up, clamp_limit=limit)
        t_tokens, top_k, _ = act.shape
        # The down stage's weight application is decided by
        # ``MUL_WEIGHT=(apply_router_weight_on_input == preserve)``
        # (``window_gemm_grouped.py``) with ``preserve=False`` here, so the flag
        # must be FORWARDED unchanged: True suppresses the second
        # multiplication once x already carries the routes, False lets the down
        # stage weight them when nothing upstream did.  Passing False
        # unconditionally would multiply by the routes a second time.
        return self.down(act.reshape(t_tokens * top_k, inter), expert_ids, routing_weights,
                         route_input=True,
                         apply_router_weight_on_input=apply_router_weight_on_input,
                         round_routes=True)


def checked_swiglu_limit(limit: "float | None", *, where: str = "") -> "float | None":
    """Validate a model's SwiGLU clamp limit, or ``None`` when absent.

    vLLM carries this as ``gemm1_clamp_limit`` and refuses, in its own words,
    to let a backend "silently select one and drop the clamp"; a value that is
    not a finite positive number is a model fact we cannot reproduce, so it
    fails closed here instead of arming a clamp that does nothing.
    """
    if limit is None:
        return None
    try:
        value = float(limit)
    except (TypeError, ValueError) as exc:
        raise GrammarError(
            f"{where}swiglu_limit {limit!r} is not a number; refusing") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise GrammarError(
            f"{where}swiglu_limit {value!r} is not finite and positive; refusing")
    return value


def _silu_and_mul(gate: torch.Tensor, up: torch.Tensor, *,
                  clamp_limit: "float | None" = None) -> torch.Tensor:
    """``silu(gate) * up`` with one bf16 rounding, the same arithmetic vLLM's
    ``SiluAndMul`` performs on the gemm output (fp32 activation, cast once).

    ``clamp_limit`` is vLLM's SwiGLU clamp: the gate saturates at ``+limit``
    (no lower clamp), the up branch at ``+-limit``.

    DTYPE.  In the production adapter these branches are the ``preserve=True``
    grouped outputs, which ``window_gemm_grouped`` allocates as **bfloat16**
    (``result = torch.empty(..., dtype=torch.bfloat16)``), so the clamp
    saturates a bf16 tile -- NOT an fp32 accumulator.  The saturation points are
    identical in fp32 (that is the stock op's arithmetic) and the cast below
    keeps the product in fp32, so the result is the single bf16 rounding of
    ``silu(clamp(gate)) * clamp(up)`` either way; what must not be claimed is
    that the clamp sees fp32 accumulators on this path.

    The stock stage this lane replaces:
    ``vllm/model_executor/layers/fused_moe/activation.py``
    ``silu_and_mul_with_clamp`` (SILU + a clamp resolves to
    ``torch.ops._C.silu_and_mul_with_clamp``), whose XPU branch spells the
    directions out as ``clamp(gate, max=limit)`` / ``clamp(up, -limit, limit)``.
    The quantised-activation Triton kernel
    ``.../layers/quantization/utils/fp8_utils.py`` saturates identically before
    narrowing.  What differs here is only the *narrowing*: the stock op clamps
    and narrows the bf16 gemm output, this adapter clamps the fp32 accumulator
    and keeps the lane's documented single rounding of the product."""
    if gate.shape != up.shape:
        raise GrammarError(f"gate {tuple(gate.shape)} and up {tuple(up.shape)} must match")
    limit = checked_swiglu_limit(clamp_limit)
    if limit is not None:
        gate = torch.clamp(gate.float(), max=limit)
        up = torch.clamp(up.float(), min=-limit, max=limit)
    act = torch.nn.functional.silu(gate.float()) * up.float()
    return act.to(torch.bfloat16)


class WindowUnitAxis:
    """One group's preallocated per-expert SoA, filled callback by callback.

    The first ``put`` for a projection allocates every stacked tensor ONCE
    from that unit's layout and fills slot 0; later experts verify the layout
    and fill their own slot; the temporary ``WindowGemvUnit`` is dropped as
    soon as its slot is written, so finalization copies nothing and the
    model's packed weights are never held twice (the intake's memory rule).
    Layouts that differ between experts are refused by name rather than
    silently re-strided.
    """

    def __init__(self, experts: int, parts: Sequence[str], *, family: str):
        self.experts = int(experts)
        self.parts = tuple(str(p) for p in parts)
        self.family = family
        self._slots: dict = {}
        self._layout: dict = {}
        self._filled: dict = {}
        self._meta: dict = {}

    def _alloc(self, part: str, unit: WindowGemvUnit) -> dict:
        e = self.experts
        rep = unit.rep
        words = rep.words
        runs = rep.runs
        rows, cols, L = int(unit.rows), int(unit.cols), int(unit.window_bits)
        device = words.device
        slot = {
            "words": torch.empty((e, words.numel()), dtype=torch.int32, device=device),
            "runs": torch.empty((e, int(runs.shape[0]), 4), dtype=torch.int32, device=device),
            "scale": torch.empty((e, rows), dtype=torch.float32, device=device),
            "perm": torch.empty((e, cols), dtype=torch.int32, device=device),
            "init": torch.empty((e, cols), dtype=torch.int32, device=device),
        }
        if self.family == "value":
            slot["table"] = torch.empty((e, unit.table.numel()), dtype=torch.bfloat16, device=device)
            slot["codes"] = torch.zeros(0, dtype=torch.uint8, device=device)
            slot["native"] = torch.zeros(0, dtype=torch.uint8, device=device)
        else:
            slot["table"] = torch.zeros(0, dtype=torch.bfloat16, device=device)
            slot["codes"] = torch.empty((e, unit.codes_of_state.numel()), dtype=torch.uint8,
                                        device=device)
            slot["native"] = torch.empty((e, unit.native.numel()), dtype=torch.uint8,
                                         device=device)
        slot["tile_words"] = torch.empty(e, dtype=torch.int32, device=device)
        slot["total_words"] = torch.empty(e, dtype=torch.int32, device=device)
        slot["has_init"] = torch.empty(e, dtype=torch.int32, device=device)
        signature = (
            self.family, rows, cols, L, int(words.numel()), int(runs.shape[0]),
            tuple(int(r) for r in rep.rates), int(unit.table.numel()),
            int(unit.scale.numel()), int(rep.tile_words), int(rep.n_tiles),
        )
        self._slots[part] = slot
        self._layout[part] = signature
        self._filled[part] = set()
        self._meta[part] = (rows, cols, L)
        return slot

    def put(self, part: str, expert: int, unit: WindowGemvUnit) -> None:
        part = str(part)
        expert = int(expert)
        if not 0 <= expert < self.experts:
            raise GrammarError(f"expert {expert} outside the axis of {self.experts}")
        if part in self._slots and expert in self._filled[part]:
            raise GrammarError(f"expert {expert} of {part!r} already placed; a second put "
                               "would overwrite packed weights")
        slot = self._slots.get(part) or self._alloc(part, unit)
        rep = unit.rep
        signature = (
            self.family, int(unit.rows), int(unit.cols), int(unit.window_bits),
            int(rep.words.numel()), int(rep.runs.shape[0]),
            tuple(int(r) for r in rep.rates), int(unit.table.numel()),
            int(unit.scale.numel()), int(rep.tile_words), int(rep.n_tiles),
        )
        if signature != self._layout[part]:
            raise GrammarError(
                f"{part!r} expert {expert}: packed layout differs from the first expert's; "
                "one grouped stack needs one layout per projection")
        slot["words"][expert] = rep.words
        slot["runs"][expert] = rep.runs
        slot["scale"][expert] = unit.scale
        slot["perm"][expert] = rep.perm
        init = unit.initial_state
        if init is None:
            slot["init"][expert] = 0
            slot["has_init"][expert] = 0
        else:
            slot["init"][expert] = init.to(torch.int32)
            slot["has_init"][expert] = 1
        if self.family == "value":
            slot["table"][expert] = unit.table.to(torch.bfloat16)
        else:
            slot["codes"][expert] = unit.codes_of_state
            slot["native"][expert] = unit.native
        slot["tile_words"][expert] = int(rep.tile_words)
        slot["total_words"][expert] = int(rep.words.numel())
        self._filled[part].add(expert)

    def finish(self) -> dict:
        """The finished SoA per part; no copying, and incomplete slots refuse."""
        out = {}
        for part, slot in self._slots.items():
            missing = [e for e in range(self.experts) if e not in self._filled[part]]
            if missing:
                raise GrammarError(f"{part!r} is missing experts {missing} at finish")
            word_width = slot["words"].shape[1]
            out[part] = {
                **slot,
                "rows": self._meta[part][0],
                "cols": self._meta[part][1],
                "window_bits": self._meta[part][2],
                "word_off": (torch.arange(self.experts, dtype=torch.int32, device=slot["words"].device)
                             * word_width),
                "run_off": torch.cat([
                    torch.zeros(1, dtype=torch.int32, device=slot["runs"].device),
                    torch.cumsum(torch.full((self.experts,), int(slot["runs"].shape[1]),
                                            dtype=torch.int32, device=slot["runs"].device), 0)]),
            }
        self._slots = {}
        self._filled = {}
        return out

    def filled(self) -> int:
        """How many (part, expert) slots have been placed."""
        return sum(len(s) for s in self._filled.values())

    def resident_bytes(self) -> int:
        """Bytes the allocated slots hold (packed constants only)."""
        return sum(t.numel() * t.element_size()
                   for slot in self._slots.values() for t in slot.values()
                   if isinstance(t, torch.Tensor))


@dataclasses.dataclass(frozen=True)
class PackedWindowMoeBundles:
    """A loader-filled grouped stack: gate/up/down SoA bundles plus the
    adapter they build.  ``resident_bytes`` counts the packed constants."""

    gate: PreparedGroupedWindowGemm
    up: PreparedGroupedWindowGemm
    down: PreparedGroupedWindowGemm
    family: str

    @property
    def experts(self) -> int:
        return self.down.experts

    @property
    def device(self) -> torch.device:
        return self.down.device

    def resident_bytes(self) -> int:
        total = 0
        for bundle in (self.gate, self.up, self.down):
            for t in (bundle.words_all, bundle.table_all, bundle.codes_all, bundle.native_all,
                      bundle.scale_all, bundle.runs_all, bundle.init_all, bundle.has_init,
                      bundle.word_off, bundle.tile_words, bundle.total_words, bundle.run_off,
                      bundle.perm_all):
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        return total

    def adapter(self) -> NativeWindowMoE:
        return native_window_moe_from_bundles(
            self.down, gate=self.gate, up=self.up, activation="silu")


def native_window_moe_from_bundles(
    down: PreparedGroupedWindowGemm,
    *,
    gate_up: "PreparedGroupedWindowGemm | None" = None,
    gate: "PreparedGroupedWindowGemm | None" = None,
    up: "PreparedGroupedWindowGemm | None" = None,
    activation: str = "silu",
) -> NativeWindowMoE:
    """The adapter from already-prepared grouped stacks (the loader path).

    ``gate_up`` is the fused ``[2I]`` stack; ``gate``/``up`` are the separate
    spelling.  Families, weight arithmetic and expert counts must agree.
    """
    if activation not in SUPPORTED_ACTIVATIONS:
        raise GrammarError(f"activation {activation!r} is not served")
    if gate_up is not None and (gate is not None or up is not None):
        raise GrammarError("the fused gate_up stack cannot be combined with gate/up stacks")
    if gate_up is None and (gate is None or up is None):
        raise GrammarError("the adapter needs either one fused gate/up stack or both gate and up")
    for name, bundle in (("gate_up", gate_up), ("gate", gate), ("up", up), ("down", down)):
        if bundle is None:
            continue
        if bundle.family != down.family:
            raise GrammarError(f"{name} family {bundle.family!r} differs from down's {down.family!r}")
        if bundle.arithmetic != down.arithmetic:
            raise GrammarError(f"{name} arithmetic {bundle.arithmetic!r} differs from down's")
        if bundle.experts != down.experts:
            raise GrammarError(f"{name} has {bundle.experts} experts, down has {down.experts}")
    if gate_up is not None and gate_up.rows != 2 * down.cols:
        raise GrammarError(
            f"the fused gate/up stack has {gate_up.rows} rows for {down.cols} intermediate columns")
    return NativeWindowMoE(gate_up=gate_up, gate=gate, up=up, down=down, activation=activation)


def prepare_native_window_moe(
    gate: Sequence[WindowGemvUnit],
    down: Sequence[WindowGemvUnit],
    *,
    up: "Sequence[WindowGemvUnit] | None" = None,
    initial_state: "torch.Tensor | None" = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    quantizer: "str | None" = "native",
    arithmetic: str = "epilogue",
    activation: str = "silu",
) -> NativeWindowMoE:
    """Prepare both stages once.

    ``gate`` is the fused gate/up stack (``rows == 2 * down.cols``) unless
    ``up`` is given, in which case ``gate`` and ``up`` are separate stacks of
    ``down.cols`` rows each.  ``arithmetic`` is the grouped bundles' explicit
    weight-side contract (``"epilogue"`` default, ``"folded"`` for the
    research BF16 contract); the down stack must agree with the gate/up
    stacks.  ``activation`` must be in :data:`SUPPORTED_ACTIVATIONS`.
    """
    if activation not in SUPPORTED_ACTIVATIONS:
        raise GrammarError(
            f"activation {activation!r} is not served; supported: {SUPPORTED_ACTIVATIONS}"
        )
    if up is None:
        gate_up = prepare_grouped_window_gemm(
            gate, initial_state=initial_state, block_m=block_m, block_n=block_n,
            block_k=block_k, quantizer=quantizer, arithmetic=arithmetic)
        gate_bundle = up_bundle = None
    else:
        gate_up = None
        gate_bundle = prepare_grouped_window_gemm(
            gate, initial_state=initial_state, block_m=block_m, block_n=block_n,
            block_k=block_k, quantizer=quantizer, arithmetic=arithmetic)
        up_bundle = prepare_grouped_window_gemm(
            up, initial_state=initial_state, block_m=block_m, block_n=block_n,
            block_k=block_k, quantizer=quantizer, arithmetic=arithmetic)
    down_bundle = prepare_grouped_window_gemm(
        down, initial_state=initial_state, block_m=block_m, block_n=block_n,
        block_k=block_k, quantizer=quantizer, arithmetic=arithmetic)
    first = gate_up or gate_bundle
    if first.family != down_bundle.family:
        raise GrammarError(
            f"gate/up family {first.family!r} and down family {down_bundle.family!r} "
            "must agree; a mixed-family MoE has no single arithmetic contract"
        )
    if first.arithmetic != down_bundle.arithmetic:
        raise GrammarError(
            "the gate/up and down stacks must share one weight arithmetic "
            f"({first.arithmetic!r} vs {down_bundle.arithmetic!r})"
        )
    if first.experts != down_bundle.experts:
        raise GrammarError(
            f"gate/up has {first.experts} experts, down has {down_bundle.experts}"
        )
    for bundle in (gate_bundle, up_bundle):
        if bundle is not None and bundle.experts != down_bundle.experts:
            raise GrammarError(
                f"a gate/up stack has {bundle.experts} experts, down has "
                f"{down_bundle.experts}"
            )
    return NativeWindowMoE(gate_up=gate_up, gate=gate_bundle, up=up_bundle,
                           down=down_bundle, activation=activation)
