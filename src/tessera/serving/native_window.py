"""Dense native window GEMM serving: the compact loader's unit, served packed.

WHAT THIS IS.  The dense FP8 and BF16 routes decode a Tessera window wire to a
stock tile today (``prepare_tessera_fp8_module`` / ``prepare_tessera_bf16_module``
+ ``materialize_fp8`` / ``materialize_bf16``), verify that tile against the
reference decoder once at load, and then run ``torch._scaled_mm`` or the stock
BF16 GEMM over it.  This module is the native replacement for that tile:

* loading goes through the compact reader (``scheme.parse_compact_blob_for_scheme``
  -> ``compact_prep.prepare_window_compact``): the container framing, role
  list, per-role grid/body/plane/span/rung facts, digests and slack are the
  same checks through the same helpers, and no weight plane is expanded;
* the compute is ``tessera.window_gemm``'s packed bitstream GEMM, whose
  ``PreparedWindowGemm`` freezes the constants once -- table/codes/native,
  the permuted ``initial_state``, the row scale and the run table -- and does
  no tensor-content validation, synchronisation, dtype cast or reallocation
  per call, so it is capturable;
* the call is a **functional custom op** (``tessera::window_gemm_dense``) with
  a fake implementation, so vLLM's compiled forward traces one opaque node
  with one output it owns and the graph key is stable.  Every constant and
  every static config value is an explicit argument and the bundle is rebuilt
  inside the op, so no module-level registry holds weights past a layer's
  life.

FAMILIES.  The BF16 family is bf16 activations in, the bf16 table decoded
in-register and FOLDED there -- each weight becomes ``bf16(value *
row_scale)``, one rounding, before the dot (tessera#614) -- fp32 accumulate,
no epilogue scale, and one bf16 cast.  That is the tile
``decode.materialize_bf16_folded`` renders and the arithmetic the routed BF16
stack serves, so the route's dense and routed modules compute one function of
the wire; it stamps ``native_window_gemm_folded``.  The FP8 family is **not**
only a different epilogue: the activation is per-token dynamic E4M3 quantized
by vLLM's native op (``fp8_per_token_dynamic`` preserved), the wire decodes to
E4M3 bytes in-register, the mainloop is the fp8 dot, and the epilogue is
``y = a_scale[m] * w_scale[n] * acc`` -- the route quantizes before the call
(the same bytes the bundle attested) and passes the result plus its scale.
It stamps ``native_window_gemm``.  Each family's arithmetic is fixed here
(``NATIVE_WINDOW_ARITHMETIC``), not chosen by a caller: a serve that could
pick either would publish two functions under one route.

WEIGHTS STAY PACKED.  Nothing here materialises a ``[rows, cols]`` weight
tensor -- not at load, not at first use, not per forward.  The prepared
bundle holds the repacked wire words plus the small tables and the fp32 row
scale; ``packed_bytes()`` and ``fingerprints()`` are what a test (or a census)
reads to see that a forward changed none of it.

TWO LAUNCH IDENTITIES, ONE PER MODULE.  Since contract v43 a module whose every
role the fused window kernel reads (``tessera.routed_fused.
fused_dense_window_supported``: the wire's one- or two-rate run table at
rates 1..8 in the packer's column order since v45 -- rate 4 alone at v43/v44
-- window 14, rows a multiple of 4, the last 128-row block partial on an
N-tail) is served by that kernel's dense case instead:
the functional custom op ``tessera::fused_window_dense`` launches
``routed_fused_kernel<FP8, 2, DENSE>`` into each role's column slice of one
``[M, rows]`` output (no concatenation), splitting K at decode shapes so
every SM has an item -- once per role with a reduce launch after a split by
default, and under ``TESSERA_DENSE_MODULE_LAUNCH=1`` on the E4M3 libraries
once per module, every role and the split's reduction in one launch
(tessera#750 WP2; ``routed_fused.ENV_DENSE_MODULE``) -- and stamps
``native_fused_window_dense`` /
``native_fused_window_dense_folded``.  The lane is decided ONCE at
preparation for the whole module -- a module stamps one decoder -- and the
Triton op above stays the dispatch for every module the predicate refuses,
for a box whose toolchain cannot build the library (the ``when_unavailable``
substitution, logged), and for ``TESSERA_DENSE_FUSED=0``.  Both identities
compute the same function of the wire (the family's published arithmetic and
fp32 operation order); their MMA accumulation orders differ, so agreement is
held to the reference product's bound, not bitwise
(``tests/test_dense_fused_window.py``, ``experiments/dense_fused_oracle.py``).

THE REFERENCE STAYS.  ``prepare_tessera_fp8_module``/``prepare_tessera_bf16_module``
and the torch window decode in ``serving.window`` are retained unchanged as
the reference path; they are no longer reached from ``process_weights_after_loading``.
Deleting them is a separate assignment, after this path has served.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional, Sequence

import torch

from ..compact_prep import DENSE_WINDOW_RATE_MAX, CompactWire, prepare_window_compact
from .scheme import (FUSED_WINDOW_DENSE_SYMBOL, ROUTES, TESSERA_BF16, TESSERA_FP8,
                     WINDOW_GEMM_SYMBOL)
from .sharding import AXIS_ROWS, ShardPlan
from .telemetry import (DECODER_NATIVE_FUSED_WINDOW_DENSE,
                        DECODER_NATIVE_FUSED_WINDOW_DENSE_E4M3MMA,
                        DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED,
                        DECODER_NATIVE_WINDOW_GEMM, DECODER_NATIVE_WINDOW_GEMM_FOLDED)

__all__ = [
    "DENSE_FAMILIES",
    "DENSE_LANES",
    "FUSED_WINDOW_DENSE_DECODER",
    "LANE_FUSED",
    "LANE_TRITON",
    "NATIVE_WINDOW_ARITHMETIC",
    "NATIVE_WINDOW_DECODER",
    "NATIVE_WINDOW_FAMILY",
    "PreparedDenseNativeModule",
    "prepare_dense_native_module",
]

_log = logging.getLogger(__name__)

@torch.library.custom_op("tessera::routed_window_classes", mutates_args=("counters",))
def _routed_window_classes(
    x: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor,
    shared: Optional[torch.Tensor], words: List[torch.Tensor], tables: List[torch.Tensor],
    inits: List[torch.Tensor], has_inits: List[torch.Tensor], wscales: List[torch.Tensor],
    runs: List[torch.Tensor], bdescs: List[torch.Tensor], starts: List[int], ends: List[int],
    tile_words: List[int], slot_words: List[int], issue_order: List[int], counters: torch.Tensor,
    library: str, piece_major: bool, resource_key: str, input_weight: bool, swiglu_limit: float,
) -> torch.Tensor:
    """One opaque native routed operation, including the two-stream event DAG.

    Routing prefixes are computed on-device on every invocation/replay. Class
    launches seed preallocated counters from absolute prefixes, then share the
    entire sorted activation and flat output buffers. No weights are repacked.
    """
    from .. import routed_fused as rf
    from .. import routed_class_dispatch

    family = "value" if library == "value" else "e4m3"
    tokens, top_k = expert_ids.shape
    hidden, inter = wscales[2].shape[1], wscales[0].shape[1]
    if input_weight:
        x = x * routing_weights.reshape(-1, 1).to(x.dtype)
    resources = rf.resolve_dispatch_resources(resource_key)
    parameters = dict(words=words, tables=tables, inits=inits, has_inits=has_inits, wscales=wscales,
        runs=runs, bdescs=bdescs, tile_words=tile_words, slot_words=slot_words, piece_major=piece_major)
    widths = routed_class_dispatch.declared_route_widths(resources.kernel, tokens, issue_order, parameters)
    routing = rf._routing_tables(expert_ids, routing_weights, ends[-1], x.device, widths)
    xq, a1 = resources.kernel.prepare_input(x, None, tokens, family, x.device)
    act = torch.empty((routing.routes, inter), dtype=torch.bfloat16, device=x.device)
    args = dict(parameters=parameters, starts=starts, ends=ends, issue_order=issue_order,
                counters=counters, resources=resources)
    routed_class_dispatch.dispatch_class_projection(0, xq, a1, routing, **args, a_row_mode=0,
        mul_weight=False, limit=swiglu_limit, out=act)
    # Join gate/up before quantizing the full sorted-route activation. The
    # quantizer and all arithmetic/route boundaries are the existing ones.
    aq, a2 = resources.kernel.prepare_input(act, None, routing.routes, family, x.device)
    routed = torch.empty((routing.routes, hidden), dtype=torch.bfloat16, device=x.device)
    routed_class_dispatch.dispatch_class_projection(2, aq, a2, routing, **args, a_row_mode=1,
        mul_weight=not input_weight, limit=float("inf"), out=routed)
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=x.device)
    if shared is None:
        rf._ext(library).token_sum(routed, out, top_k)
    else:
        rf._ext(library).token_sum_shared(routed, shared, out, top_k)
    return out


@_routed_window_classes.register_fake
def _routed_window_classes_fake(x, expert_ids, routing_weights, shared, words, tables,
        inits, has_inits, wscales, runs, bdescs, starts, ends, tile_words, slot_words,
        issue_order, counters, library, piece_major, resource_key, input_weight, swiglu_limit):
    return torch.empty((x.shape[0], wscales[2].shape[1]), dtype=torch.bfloat16, device=x.device)



DENSE_FAMILIES = (TESSERA_FP8, TESSERA_BF16)

#: The window GEMM's family spelling for each route family.
NATIVE_WINDOW_FAMILY = {TESSERA_FP8: "e4m3", TESSERA_BF16: "value"}
#: The weight arithmetic each route family serves (``window_gemm``'s
#: ``arithmetic``): the FP8 family's row scale is on the fp32 epilogue beside
#: the per-token A scale; the BF16 family's is folded into each weight
#: (tessera#614).
NATIVE_WINDOW_ARITHMETIC = {TESSERA_FP8: "epilogue", TESSERA_BF16: "folded"}
#: The decoder each arithmetic stamps -- one symbol, two numerical functions.
NATIVE_WINDOW_DECODER = {"epilogue": DECODER_NATIVE_WINDOW_GEMM,
                         "folded": DECODER_NATIVE_WINDOW_GEMM_FOLDED}
#: The fused window kernel's dense identity, per arithmetic (contract v43).
FUSED_WINDOW_DENSE_DECODER = {"epilogue": DECODER_NATIVE_FUSED_WINDOW_DENSE,
                              "folded": DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED}
#: The dense identity's decoder per ``routed_fused`` library: the E4M3
#: instruction's library is its own numerical function of the wire (the same
#: exact products, another fp32 accumulation order), so it stamps its own.
FUSED_WINDOW_DENSE_LIBRARY_DECODER = {"value": DECODER_NATIVE_FUSED_WINDOW_DENSE_FOLDED,
                                      "e4m3": DECODER_NATIVE_FUSED_WINDOW_DENSE,
                                      "e4m3mma": DECODER_NATIVE_FUSED_WINDOW_DENSE_E4M3MMA}
#: The two lanes a prepared module may run, and the ``(symbol, decoder)`` each
#: stamps per arithmetic.  ``lane`` is a module fact decided at preparation.
LANE_TRITON = "triton"
LANE_FUSED = "fused"
DENSE_LANES = {
    LANE_TRITON: (WINDOW_GEMM_SYMBOL, NATIVE_WINDOW_DECODER),
    LANE_FUSED: (FUSED_WINDOW_DENSE_SYMBOL, FUSED_WINDOW_DENSE_DECODER),
}
#: The activation contract each route publishes, read off its ROUTES entry.
ACTIVATION_CONTRACT = {family: ROUTES[family]["activation_contract"]
                       for family in DENSE_FAMILIES}

@torch.library.custom_op("tessera::window_gemm_dense", mutates_args=())
def _window_gemm_dense(
    x: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    words: torch.Tensor, table: torch.Tensor, codes: torch.Tensor,
    native: torch.Tensor, scale: torch.Tensor, runs: torch.Tensor,
    init_perm: torch.Tensor, perm: torch.Tensor,
    rows: int, cols: int, window_bits: int, tile_words: int, total_words: int,
    has_init: bool, family_e4m3: bool,
    block_m: int, block_n: int, block_k: int, folded: bool,
) -> torch.Tensor:
    """One role's packed window GEMM, functional and opaque.

    Every input is an explicit argument -- the frozen tensors and the static
    geometry/config the launch reads -- and the validation-free
    ``PreparedWindowGemm`` is rebuilt from them inside the op.  Nothing is
    looked up in module state, so a compiled forward treats the constants as
    graph inputs and a model unload releases them with the layer: there is no
    registry to keep weights alive.

    The family spelling matches the bundle's: ``e4m3`` takes prequantized fp8
    ``x`` plus its per-token scale (or bf16, quantized by the native op inside
    the bundle), ``value`` takes bf16 and returns bf16.  ``folded`` is the
    bundle's ``arithmetic == "folded"``, an explicit argument like every other
    static value, so the rebuilt bundle cannot run a different arithmetic
    than the prepared one.
    """
    from .. import window_gemm as wg

    bundle = wg.PreparedWindowGemm(
        words=words, table=table, codes=codes, native=native, scale=scale,
        runs=runs, init_perm=init_perm, perm=perm, tile_words=int(tile_words),
        total_words=int(total_words), rows=int(rows), cols=int(cols),
        window_bits=int(window_bits), family=("e4m3" if family_e4m3 else "value"),
        has_init=bool(has_init), block_m=int(block_m), block_n=int(block_n),
        block_k=int(block_k), quantizer="native",
        arithmetic="folded" if folded else "epilogue")
    if family_e4m3:
        return bundle(x, a_scale=a_scale)
    return bundle(x)


@_window_gemm_dense.register_fake
def _window_gemm_dense_fake(
    x, a_scale, words, table, codes, native, scale, runs, init_perm, perm,
    rows, cols, window_bits, tile_words, total_words, has_init, family_e4m3,
    block_m, block_n, block_k, folded,
):
    return torch.empty((x.shape[0], rows), dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("tessera::fused_window_dense", mutates_args=())
def _fused_window_dense(
    x: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    words: List[torch.Tensor], tables: List[torch.Tensor], inits: List[torch.Tensor],
    has_inits: List[torch.Tensor], wscales: List[torch.Tensor],
    runs: List[torch.Tensor], bdescs: List[torch.Tensor],
    role_rows: List[int], tile_words: List[int], slot_words: List[int],
    cols: int, family_e4m3: bool, folded: bool,
) -> torch.Tensor:
    """A whole module through the fused window kernel's dense case: one node.

    The module's roles are explicit lists (one entry per role, row order), so
    a compiled forward traces one opaque node with one output it owns, exactly
    as ``tessera::window_gemm_dense`` does per role.  Each role is launched
    into its column slice of the one ``[M, sum(rows)]`` output; the work
    counter the kernel claims items through is allocated here per call and
    zeroed in-stream, so the op stays functional (nothing outside it is
    mutated) and a captured forward replays.  ``runs``, ``bdescs``,
    ``tile_words`` and ``slot_words`` are each role's run pair, block
    descriptor, tile stride and word-stage slot (tessera#694): the kernel
    reads the wire's rates from them, so they travel with the role like every
    other frozen input.  ``folded`` is carried for the same reason the Triton
    op carries it: the family fixes it, and a rebuilt role cannot run a
    different arithmetic than the prepared one.
    """
    from .. import routed_fused as rf

    family = "e4m3" if family_e4m3 else "value"
    if folded != (family == "value"):
        raise ValueError("the fused dense identity folds the value family and only it")
    m = int(x.shape[0])
    total = int(sum(role_rows))
    out = torch.empty((m, total), dtype=torch.bfloat16, device=x.device)
    if m == 0:
        return out
    roles = [rf.FusedDenseWindowRole(
        family=family, rows=int(rows), cols=int(cols), words=words[i], table16=tables[i],
        init=inits[i], has_init=has_inits[i], wscale=wscales[i],
        runs=runs[i], bdesc=bdescs[i], tile_words=int(tile_words[i]), slot_words=int(slot_words[i]))
        for i, rows in enumerate(role_rows)]
    if family == "e4m3" and rf.dense_module_launch_enabled():
        # The E4M3 libraries take the module's roles in one launch and reduce
        # a K split in-kernel (tessera#750 WP2): one fill and one kernel.
        # Opt-in until tessera#778's decode measurement: read per call, so a
        # captured forward keeps the launch it was captured with.
        rf.dense_forward_roles(roles, x, a_scale, out)
        return out
    # One in-stream fill zeroes every role's slot, so the launches add none
    # (``zeroed=True``): at small M this op's host time, not its kernels, sets
    # the eager forward's time.
    counter = torch.zeros(len(role_rows), dtype=torch.int32, device=x.device)
    offset = 0
    for i, role in enumerate(roles):
        rf.dense_forward(role, x, a_scale, out.narrow(1, offset, role.rows), counter[i:i + 1], zeroed=True)
        offset += role.rows
    return out


@_fused_window_dense.register_fake
def _fused_window_dense_fake(x, a_scale, words, tables, inits, has_inits, wscales, runs, bdescs,
                             role_rows, tile_words, slot_words, cols, family_e4m3, folded):
    return torch.empty((x.shape[0], int(sum(role_rows))), dtype=torch.bfloat16, device=x.device)


class PreparedDenseNativeModule:
    """One vLLM dense module's roles, prepared once and served packed.

    A module is one or more roles (q/k/v, gate/up) cut to this rank by the
    layer's ``ShardPlan``; each role is one ``PreparedWindowGemm`` over the
    rank-local window unit.  ``apply`` runs them and concatenates on the row
    axis, which is the arrangement the merged Linear expects.
    """

    __slots__ = ("__roles", "__rows", "__columns", "__device", "__family",
                 "__arithmetic", "__lane", "__fused", "__lane_reason", "__decoded")

    def __init__(self, roles, *, rows: int, columns: int, device: torch.device,
                 family: str, lane: str = LANE_TRITON, fused_roles=None,
                 lane_reason: "str | None" = None):
        self.__roles = tuple(roles)
        # Both dense custom ops carry legacy words without a layout argument.
        # Admit that invariant while roles still carry their immutable tags.
        from ..kernel_window_gemv import require_legacy_word_layout
        for role in self.__roles:
            require_legacy_word_layout(getattr(role.bundle, "word_layout", "legacy"),
                                       "the native dense custom-op owner")
        self.__rows = int(rows)
        self.__columns = int(columns)
        self.__device = device
        self.__family = str(family)
        if sum(role.rows for role in self.__roles) != self.__rows:
            raise ValueError("prepared native roles do not stack to the module's rows")
        if any(role.bundle.cols != self.__columns for role in self.__roles):
            raise ValueError("every role of a module shares its input width")
        arithmetics = {role.bundle.arithmetic for role in self.__roles}
        if len(arithmetics) != 1:
            raise ValueError(
                f"the roles of one module run one weight arithmetic, got {sorted(arithmetics)}; "
                "the module stamps one decoder")
        self.__arithmetic = arithmetics.pop()
        if lane not in DENSE_LANES:
            raise ValueError(f"unknown dense lane {lane!r}; one of {sorted(DENSE_LANES)}")
        fused = tuple(fused_roles) if fused_roles is not None else ()
        if (lane == LANE_FUSED) != bool(fused):
            raise ValueError("the fused lane carries one prepared kernel role per module role")
        if fused and (len(fused) != len(self.__roles)
                      or any(f.rows != r.rows for f, r in zip(fused, self.__roles))):
            raise ValueError("the fused roles do not match the module's roles row for row")
        self.__lane = lane
        self.__fused = fused
        self.__lane_reason = lane_reason
        self.__decoded = None

    @property
    def rows(self): return self.__rows
    @property
    def columns(self): return self.__columns
    @property
    def device(self): return self.__device
    @property
    def family(self): return self.__family
    @property
    def arithmetic(self): return self.__arithmetic
    @property
    def lane(self):
        """``"fused"`` or ``"triton"``: which launch identity serves this module."""
        return self.__lane
    @property
    def lane_reason(self):
        """Why the module kept the Triton lane (``None`` on the fused lane)."""
        return self.__lane_reason
    @property
    def symbol(self): return DENSE_LANES[self.__lane][0]
    @property
    def decoder(self):
        if self.__lane == LANE_FUSED:
            return FUSED_WINDOW_DENSE_LIBRARY_DECODER[self.__fused[0].library]
        return DENSE_LANES[self.__lane][1][self.__arithmetic]
    @property
    def launch_pair(self):
        """The ``(symbol, decoder)`` of this module's window lane (every M
        without a decoded copy; M below ``e4m3_prefill.MIN_M`` with one)."""
        return (self.symbol, self.decoder)

    @property
    def decoded(self):
        """The decode-once E4M3 copy (``e4m3_prefill.DecodedE4M3``), or ``None``."""
        return self.__decoded

    def attach_decoded(self, decoded) -> None:
        """Hold ``decoded`` (made from THIS module) and serve large M from it.

        Once per module, E4M3 family only, shape-checked against the module;
        the route decides whether to attach (``e4m3_prefill.enabled`` and the
        resident mode)."""
        if self.__family != "e4m3":
            raise ValueError(f"a decode-once copy serves the E4M3 family, not {self.__family!r}")
        if self.__decoded is not None:
            raise ValueError("this module already holds a decode-once copy")
        if tuple(decoded.weight.shape) != (self.__rows, self.__columns) \
                or tuple(decoded.scale.shape) != (self.__rows,):
            raise ValueError(
                f"decode-once copy {tuple(decoded.weight.shape)}/{tuple(decoded.scale.shape)} "
                f"does not fit a [{self.__rows}, {self.__columns}] module")
        self.__decoded = decoded

    def launch_pair_for(self, m: int):
        """The ``(symbol, decoder)`` ``apply`` runs for an ``m``-row input."""
        from .e4m3_prefill import MIN_M
        from .scheme import DECODE_ONCE_DENSE_SYMBOL
        from .telemetry import DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3

        if self.__decoded is not None and int(m) >= MIN_M:
            return (DECODE_ONCE_DENSE_SYMBOL, DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3)
        return self.launch_pair
    @property
    def role_names(self): return tuple(role.name for role in self.__roles)
    @property
    def role_bundles(self):
        """Each role's frozen ``PreparedWindowGemm``, in row order (read-only)."""
        return tuple(role.bundle for role in self.__roles)

    def layout_facts(self):
        """Each role's lightweight layout facts, in row order.

        Provenance and tests read these (the wire's rates, the repack's padded
        rows, the cut's row offset and whether it carries history); a serve
        reads the bundles.  They are ints and bools on purpose: the compact
        ``WindowGemvUnit`` that produced a bundle also holds its GEMV item
        tables and its UNPERMUTED start state, and keeping it here would
        retain storage the bundle does not count (Astra residency review).
        The start state itself stays reachable through
        ``bundle.init_perm``/``bundle.perm``, which the kernel needs and
        ``packed_bytes`` counts.
        """
        return tuple(role.facts for role in self.__roles)

    def row_scale(self) -> torch.Tensor:
        """The per-row fp32 scale, ``[rows]``, in role order.

        The same expression the reference decoder applies
        (``scale_rows * global`` in fp32), carried here for the route record
        and the retained reference checks; where it is applied (the fp32
        epilogue, or folded into each weight) is the bundles' ``arithmetic``.
        """
        return torch.cat([role.bundle.scale for role in self.__roles]).contiguous()

    def apply(self, x: torch.Tensor, a_scale: "torch.Tensor | None" = None) -> torch.Tensor:
        """``x [M, columns]`` in original column order -> ``bf16 [M, rows]``.

        ``x`` is bf16 for the value family and prequantized fp8 plus its
        per-token scale for the e4m3 family (the route quantizes before this
        call, so the contract's quantizer is the one that ran).  On the fused
        lane one custom-op node serves the whole module; on the Triton lane one
        node per role.  No host-side data-dependent work on either.  A module
        holding a decode-once copy serves M >= ``e4m3_prefill.MIN_M`` from it
        (:meth:`launch_pair_for` names which ran).  That branch reads M on the
        host, so the decode-once lane is EAGER-ONLY.  The gate is the route's
        LOAD (``fp8_route`` refuses to attach a copy when vLLM compiles the
        forward); the raise below is a backstop for a direct caller, and under
        ``torch.compile`` without ``fullgraph`` it is a graph break Dynamo may
        run around.  A CUDA-graph capture sees a concrete M and records the
        branch that ran for it.
        """
        if self.__decoded is not None:
            from .e4m3_prefill import FLAG, MIN_M, prefill_apply

            if torch.compiler.is_compiling():
                raise RuntimeError(
                    f"the decode-once E4M3 lane is eager-only ({FLAG}=1); serve with "
                    "compilation mode NONE or unset the flag")
            if int(x.shape[0]) >= MIN_M:
                return prefill_apply(self.__decoded, x, a_scale)
        if self.__lane == LANE_FUSED:
            fused = self.__fused
            return _fused_window_dense(
                x, a_scale, [f.words for f in fused], [f.table16 for f in fused],
                [f.init for f in fused], [f.has_init for f in fused], [f.wscale for f in fused],
                [f.runs for f in fused], [f.bdesc for f in fused],
                [int(f.rows) for f in fused], [int(f.tile_words) for f in fused],
                [int(f.slot_words) for f in fused], int(self.__columns), self.__family == "e4m3",
                self.__arithmetic == "folded")
        parts = []
        for role in self.__roles:
            bundle = role.bundle
            parts.append(_window_gemm_dense(
                x, a_scale, bundle.words, bundle.table, bundle.codes, bundle.native,
                bundle.scale, bundle.runs, bundle.init_perm, bundle.perm,
                int(bundle.rows), int(bundle.cols), int(bundle.window_bits),
                int(bundle.tile_words), int(bundle.total_words), bool(bundle.has_init),
                self.__family == "e4m3", int(bundle.block_m), int(bundle.block_n),
                int(bundle.block_k), self.__arithmetic == "folded"))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    # -- residency accounting ------------------------------------------------

    def named_tensors(self):
        """References to the exact frozen kernel inputs; no copies or mutation.

        These tensors are held by slotted prepared bundles rather than registered
        buffers. Resource observers use these names beneath the owning Linear;
        exposing them does not change module loading or device movement.
        """
        for index, role in enumerate(self.__roles):
            for name in ("words", "table", "codes", "native", "scale", "runs",
                         "init_perm", "perm"):
                yield f"roles.{index}.{name}", getattr(role.bundle, name)
        # The fused lane's own storage beyond the bundles, per role: the composed
        # 16-bit table (32 KB), the has_init flag, the run pair and the 32-column
        # block descriptors (FusedDenseWindowRole.named_tables); its words, start
        # state and row scale are views of the bundle tensors already yielded.
        for index, fused in enumerate(self.__fused):
            for name, tensor in fused.named_tables():
                yield f"roles.{index}.{name}", tensor
        # The decode-once copy, when the route attached one: one E4M3 byte per
        # weight and the fp32 row scale it is served with (tessera#931).
        if self.__decoded is not None:
            yield "decoded.weight", self.__decoded.weight
            yield "decoded.scale", self.__decoded.scale

    def packed_bytes(self) -> int:
        """Device bytes the prepared weights occupy: the packed wire half, plus
        the decode-once copy when one is attached (``named_tensors``)."""
        return sum(tensor.numel() * tensor.element_size()
                   for _, tensor in self.named_tensors())

    def fingerprints(self):
        """Identity of every frozen tensor, for a load-time/after-forward check."""
        return tuple((tensor.data_ptr(), tensor._version, tuple(tensor.shape), tensor.dtype)
                     for _, tensor in self.named_tensors())


@dataclasses.dataclass(frozen=True)
class _RoleFacts:
    """What a role's compact unit says about its layout, without its tensors:
    the wire's per-column rates, the repack's padded row count, the input
    width, the cut's row offset and whether the cut carries history."""

    rates: tuple
    rows_p: int
    cols: int
    row_offset: int
    has_history: bool


@dataclasses.dataclass(frozen=True)
class _NativeRole:
    name: str
    rows: int
    bundle: object                     # window_gemm.PreparedWindowGemm (import kept lazy)
    facts: _RoleFacts


def _require_route_unit(route: str, name: str, metadata) -> None:
    """The route-table facts, checked before any plane is packed.

    The same four conditions the materialising preparers state (grid, body,
    scale plane, span), read off ``scheme.ROUTES`` rather than restated, with
    the route's own words.  The grid-object half (a scalar hardware grid) is
    the compact window preparer's own refusal, which asks the grid directly.
    """
    entry = ROUTES[route]
    if metadata.grid.name not in entry["grids"]:
        raise ValueError(
            f"role {name!r}: {route} decodes {entry['grid_kind']} grid "
            f"{entry['grids']} (tessera.serving.scheme.ROUTES[{route!r}]), "
            f"not {metadata.grid.name}")
    if metadata.body.name != entry["body"]:
        raise ValueError(
            f"role {name!r}: {route} decodes the {entry['body']} body "
            f"(tessera.serving.scheme.ROUTES[{route!r}]); this unit carries "
            f"{metadata.body.name}, which has no in-forward decoder here")
    plane = metadata.manifest.scale_plane.kind.name
    if plane != entry["plane"]:
        raise ValueError(
            f"role {name!r}: {route} takes the {entry['plane']} plane "
            f"(tessera.serving.scheme.ROUTES[{route!r}]); this unit carries "
            f"{plane}")
    if int(metadata.span) != entry["span"]:
        raise ValueError(
            f"role {name!r}: {route} decodes span-{entry['span']} "
            f"{entry['body']} (tessera.serving.scheme.ROUTES[{route!r}]); "
            f"this unit carries span {int(metadata.span)}")


def _role_cut(plan: ShardPlan, name: str):
    """This rank's ``(rows, cols)`` for one role, off the layer's own plan."""
    role = plan.role(name)
    if role.is_whole:
        return {}
    if plan.axis == AXIS_ROWS:
        return {"rows": (role.lo, role.hi)}
    return {"cols": (role.lo, role.hi)}


def prepare_dense_native_module(
    compact_roles: "Sequence[tuple[str, CompactWire]]",
    plan: ShardPlan,
    *,
    family: str,
    device="cuda",
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
) -> PreparedDenseNativeModule:
    """``[(role, CompactWire)]`` + the layer's plan -> the packed bundles.

    Each role is cut to this rank off the plan (a row cut for a
    column-parallel module, a column cut for a row-parallel one), packed by
    ``compact_prep.prepare_window_compact`` -- which refuses every cut the
    reference cutter would refuse and derives the row cut's incoming state
    from the packed wire -- and frozen by ``window_gemm.prepare_window_gemm``,
    which attests the native per-token FP8 quantizer once for the e4m3 family
    and fixes the family's weight arithmetic (``NATIVE_WINDOW_ARITHMETIC``).
    No reference decode runs here; the expanded reference preparations remain
    the test oracle.

    The lane is decided here, once per module: the fused window kernel's dense
    identity when ``routed_fused.fused_dense_window_supported`` admits every
    role and ``TESSERA_DENSE_FUSED`` is not ``0`` (its libraries are built or
    loaded HERE, so a toolchain that cannot compile them fails at load into
    the Triton lane -- the ``when_unavailable`` substitution -- and not on a
    serve's first forward); the Triton GEMM otherwise, with the reason kept
    on the module (``lane_reason``) and logged.
    """
    from .. import window_gemm as wg

    if family not in DENSE_FAMILIES:
        raise ValueError(f"{family!r} is not a dense native window family ({DENSE_FAMILIES})")
    window_family = NATIVE_WINDOW_FAMILY[family]
    device = torch.device("cuda" if device is None else device)
    if not compact_roles:
        raise ValueError("a Tessera module needs at least one role")
    roles: List[_NativeRole] = []
    offset = 0
    columns = None
    for name, wire in compact_roles:
        metadata = wire.metadata
        _require_route_unit(family, name, metadata)
        cut = _role_cut(plan, name)
        # a dense unit serves every rate its window holds (tessera#750 item
        # 4); the fused lane below admits the rates its library decodes
        unit = prepare_window_compact(
            wire, device=device, family=window_family,
            rate_max=DENSE_WINDOW_RATE_MAX, **cut)
        if columns is None:
            columns = int(unit.cols)
        elif int(unit.cols) != columns:
            raise ValueError(f"role {name!r} has {unit.cols} input columns, the module {columns}")
        facts = _RoleFacts(
            rates=tuple(unit.rep.rates), rows_p=int(unit.rep.rows_p),
            cols=int(unit.cols), row_offset=int(unit.row_offset),
            has_history=bool(unit.initial_state is not None
                             and unit.initial_state.any()))
        bundle = wg.prepare_window_gemm(
            unit, block_m=block_m, block_n=block_n, block_k=block_k,
            quantizer="native" if window_family == "e4m3" else None,
            arithmetic=NATIVE_WINDOW_ARITHMETIC[family])
        roles.append(_NativeRole(name=wire.name or name, rows=int(unit.rows),
                                 bundle=bundle, facts=facts))
        offset += int(unit.rows)
    lane, fused, reason = _decide_lane(roles, window_family)
    return PreparedDenseNativeModule(roles, rows=offset, columns=columns,
                                     device=device, family=window_family,
                                     lane=lane, fused_roles=fused, lane_reason=reason)


def _decide_lane(roles, window_family: str):
    """``(lane, fused_roles, reason)`` for a module's prepared roles.

    Every role must pass the predicate (one decoder per module); the first
    refusal is the module's reason.  A build or load failure of the family's
    library after the predicate admitted the module is the substitution the
    extension entry's ``when_unavailable`` publishes: the Triton lane, with a
    warning naming the cause.
    """
    from .. import routed_fused as rf
    from ..errors import GrammarError

    for role in roles:
        reason = rf.fused_dense_window_supported(role.bundle)
        if reason is not None:
            _log.info("Triton dense window GEMM kept for a %s module of %d role(s): role %r: %s",
                      window_family, len(roles), role.name, reason)
            return LANE_TRITON, None, f"role {role.name!r}: {reason}"
    try:
        fused = tuple(rf.prepare_dense_role(role.bundle) for role in roles)
    except GrammarError:
        raise
    except Exception as exc:  # noqa: BLE001 -- the native build is what may fail here
        reason = f"native build unavailable ({type(exc).__name__}: {exc})"
        _log.warning("fused dense window identity unavailable for a %s module of %d role(s); "
                     "the Triton GEMM serves it: %s", window_family, len(roles), reason)
        return LANE_TRITON, None, reason
    return LANE_FUSED, fused, None
