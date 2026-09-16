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
in-register, fp32 accumulate, the row scale on the fp32 epilogue and one bf16
cast.  The FP8 family is **not** only a different epilogue: the activation is
per-token dynamic E4M3 quantized by vLLM's native op (``fp8_per_token_dynamic``
preserved), the wire decodes to E4M3 bytes in-register, the mainloop is the fp8
dot, and the epilogue is ``y = a_scale[m] * w_scale[n] * acc`` -- the route
quantizes before the call (the same bytes the bundle attested) and passes the
result plus its scale.

WEIGHTS STAY PACKED.  Nothing here materialises a ``[rows, cols]`` weight
tensor -- not at load, not at first use, not per forward.  The prepared
bundle holds the repacked wire words plus the small tables and the fp32 row
scale; ``packed_bytes()`` and ``fingerprints()`` are what a test (or a census)
reads to see that a forward changed none of it.

THE REFERENCE STAYS.  ``prepare_tessera_fp8_module``/``prepare_tessera_bf16_module``
and the torch window decode in ``serving.window`` are retained unchanged as
the reference path; they are no longer reached from ``process_weights_after_loading``.
Deleting them is a separate assignment, after this path has served.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

import torch

from ..compact_prep import CompactWire, prepare_window_compact
from .scheme import ROUTES, TESSERA_BF16, TESSERA_FP8, WINDOW_GEMM_SYMBOL
from .sharding import AXIS_ROWS, ShardPlan
from .telemetry import DECODER_NATIVE_WINDOW_GEMM

__all__ = [
    "DENSE_FAMILIES",
    "NATIVE_WINDOW_FAMILY",
    "PreparedDenseNativeModule",
    "prepare_dense_native_module",
]

DENSE_FAMILIES = (TESSERA_FP8, TESSERA_BF16)

#: The window GEMM's family spelling for each route family.
NATIVE_WINDOW_FAMILY = {TESSERA_FP8: "e4m3", TESSERA_BF16: "value"}
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
    block_m: int, block_n: int, block_k: int,
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
    the bundle), ``value`` takes bf16 and returns bf16.
    """
    from .. import window_gemm as wg

    bundle = wg.PreparedWindowGemm(
        words=words, table=table, codes=codes, native=native, scale=scale,
        runs=runs, init_perm=init_perm, perm=perm, tile_words=int(tile_words),
        total_words=int(total_words), rows=int(rows), cols=int(cols),
        window_bits=int(window_bits), family=("e4m3" if family_e4m3 else "value"),
        has_init=bool(has_init), block_m=int(block_m), block_n=int(block_n),
        block_k=int(block_k), quantizer="native")
    if family_e4m3:
        return bundle(x, a_scale=a_scale)
    return bundle(x)


@_window_gemm_dense.register_fake
def _window_gemm_dense_fake(
    x, a_scale, words, table, codes, native, scale, runs, init_perm, perm,
    rows, cols, window_bits, tile_words, total_words, has_init, family_e4m3,
    block_m, block_n, block_k,
):
    return torch.empty((x.shape[0], rows), dtype=torch.bfloat16, device=x.device)


class PreparedDenseNativeModule:
    """One vLLM dense module's roles, prepared once and served packed.

    A module is one or more roles (q/k/v, gate/up) cut to this rank by the
    layer's ``ShardPlan``; each role is one ``PreparedWindowGemm`` over the
    rank-local window unit.  ``apply`` runs them and concatenates on the row
    axis, which is the arrangement the merged Linear expects.
    """

    __slots__ = ("__roles", "__rows", "__columns", "__device", "__family")

    def __init__(self, roles, *, rows: int, columns: int, device: torch.device,
                 family: str):
        self.__roles = tuple(roles)
        self.__rows = int(rows)
        self.__columns = int(columns)
        self.__device = device
        self.__family = str(family)
        if sum(role.rows for role in self.__roles) != self.__rows:
            raise ValueError("prepared native roles do not stack to the module's rows")
        if any(role.bundle.cols != self.__columns for role in self.__roles):
            raise ValueError("every role of a module shares its input width")

    @property
    def rows(self): return self.__rows
    @property
    def columns(self): return self.__columns
    @property
    def device(self): return self.__device
    @property
    def family(self): return self.__family
    @property
    def decoder(self): return DECODER_NATIVE_WINDOW_GEMM
    @property
    def role_names(self): return tuple(role.name for role in self.__roles)

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
        and the retained reference checks; the epilogue itself lives in the
        bundles.
        """
        return torch.cat([role.bundle.scale for role in self.__roles]).contiguous()

    def apply(self, x: torch.Tensor, a_scale: "torch.Tensor | None" = None) -> torch.Tensor:
        """``x [M, columns]`` in original column order -> ``bf16 [M, rows]``.

        ``x`` is bf16 for the value family and prequantized fp8 plus its
        per-token scale for the e4m3 family (the route quantizes before this
        call, so the contract's quantizer is the one that ran).  One custom-op
        node per role; no host-side data-dependent work.
        """
        parts = []
        for role in self.__roles:
            bundle = role.bundle
            parts.append(_window_gemm_dense(
                x, a_scale, bundle.words, bundle.table, bundle.codes, bundle.native,
                bundle.scale, bundle.runs, bundle.init_perm, bundle.perm,
                int(bundle.rows), int(bundle.cols), int(bundle.window_bits),
                int(bundle.tile_words), int(bundle.total_words), bool(bundle.has_init),
                self.__family == "e4m3", int(bundle.block_m), int(bundle.block_n),
                int(bundle.block_k)))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    # -- residency accounting ------------------------------------------------

    def packed_bytes(self) -> int:
        """Device bytes the prepared weights occupy: the packed wire half.

        The repacked words, the tables and the fp32 row scale -- never a
        ``[rows, cols]`` decoded tile.
        """
        total = 0
        for role in self.__roles:
            bundle = role.bundle
            for tensor in (bundle.words, bundle.table, bundle.codes, bundle.native,
                           bundle.scale, bundle.runs, bundle.init_perm, bundle.perm):
                total += tensor.numel() * tensor.element_size()
        return total

    def fingerprints(self):
        """Identity of every frozen tensor, for a load-time/after-forward check."""
        out = []
        for role in self.__roles:
            bundle = role.bundle
            for tensor in (bundle.words, bundle.table, bundle.codes, bundle.native,
                           bundle.scale, bundle.runs, bundle.init_perm, bundle.perm):
                out.append((tensor.data_ptr(), tensor._version, tuple(tensor.shape),
                            tensor.dtype))
        return tuple(out)


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
    which attests the native per-token FP8 quantizer once for the e4m3 family.
    No reference decode runs here; the expanded reference preparations remain
    the test oracle.
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
        unit = prepare_window_compact(
            wire, device=device, family=window_family, **cut)
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
            quantizer="native" if window_family == "e4m3" else None)
        roles.append(_NativeRole(name=wire.name or name, rows=int(unit.rows),
                                 bundle=bundle, facts=facts))
        offset += int(unit.rows)
    return PreparedDenseNativeModule(roles, rows=offset, columns=columns,
                                     device=device, family=window_family)
