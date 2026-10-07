"""The routed WINDOW class plugin for E4M3 and folded BF16.

Every routed scheme declares storage-to-global expert IDs and a contiguous class
partition. Load callbacks place storage-named projection wires directly into the
exact packed axes. The plugin constructs one inverse on the load device and
remaps the router's IDs once without changing top-k positions, weights or EP's
expert_map. The native class adapter owns the existing quantization, SwiGLU,
route boundary and fixed-order token sum.

There is one routed WINDOW execution path, not a materializing compatibility
reader. Standalone CPU materialization helpers remain reference/measurement
utilities and are not selected by this serving method. Execution telemetry names
the actual class operation; it does not qualify an artifact or promote a cell.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

from ..errors import GrammarError
from ..moe_execution import ResearchSelectedMoeConfig
from ..moe_layout import W13_PROJECTIONS, validate_moe_wire_lengths
from .glm53_shared_fold import native_call
from .lane import MODE_RESIDENT, MODES
from .residency import named_resident_tensors
from .scheme import (MOE_GROUP_SHARDS, MOE_GROUPS, ROUTES, TESSERA_BF16, TESSERA_FP8,
                     moe_census_symbol_base as census_symbol_base,
                     expert_rungs_mixed, expert_role_declarations,
                     stack_effective_rungs,
                     parse_tessera_expert_blob,
                     validate_tessera_moe_scheme)
from .telemetry import emit_route, route_shape

#: Opt-in: re-lay the compact routed window body piece-major (tessera#739).
#: Default off.  Only the single-rate rate-4 body the fused R4 reader
#: addresses is eligible (``kernel_window_gemv.piece_major_eligible``); every
#: other unit keeps legacy words, and no reader is re-strided to read either.
ENV_PIECE_MAJOR = "TESSERA_ROUTED_PIECE_MAJOR"


def _piece_major_requested() -> bool:
    return os.environ.get(ENV_PIECE_MAJOR, "0").strip().lower() not in ("", "0", "false", "no")


def _piece_major_admissible(family: str) -> bool:
    """The full intake gate for the piece-major resident layout (tessera#739).

    All three hold before the transient is re-laid:

    * the piece-major experiment is requested;
    * the family is E4M3 -- BF16 keeps canonical word placement;
    * the E4M3 library this process builds is the MMA one, since the
      piece-major reader is instantiated only there.  ``library_for`` reads
      ``TESSERA_FUSED_E4M3_MMA``: an explicit ``f16`` keeps every body legacy.
    """
    if not _piece_major_requested():
        return False
    if family != "e4m3":
        return False
    from ..routed_fused import library_for, library_mma8
    return library_mma8(library_for(family))


__all__ = [
    "ACTIVATION_CONTRACT",
    "SHARD_TO_GROUP",
    "PreparedTesseraMoeExperts",
    "PreparedTesseraPackedMoeExperts",
    "PreparedTesseraPackedBf16MoeExperts",
    "ResearchSelectedMoeConfig",
    "census_expected",
    "census_symbol_base",
    "prepare_tessera_moe_experts",
    "prepare_tessera_packed_moe_experts",
    "prepare_tessera_packed_bf16_moe_experts",
    "build_tessera_moe_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_FP8]["activation_contract"]




#: The payload family each window expert route's census expectation is read
#: under, keyed by route: the contract's ``executes`` table and
#: ``census.platform_expectation`` speak payload families.
_CENSUS_PAYLOAD_FAMILY = {TESSERA_FP8: "TESSERA_E4M3_K1", TESSERA_BF16: "TESSERA_BF16_K1"}


def native_decoder(family: str) -> str:
    """The actual class decoder selected for this routed window family."""
    from ..routed_fused import library_for, routed_class_launch_pair

    if family not in _CENSUS_PAYLOAD_FAMILY:
        raise KeyError(f"{family!r} has no native WINDOW expert route")
    library = library_for("value" if family == TESSERA_BF16 else "e4m3")
    return routed_class_launch_pair(library)[1]


def _profiler_label(adapter) -> str:
    """Use the class operation's own profiler identity."""
    return adapter.PROFILER_LABEL


def census_expected(*, compiled: bool = False, platform=None,
                    family: str = TESSERA_FP8) -> dict:
    """Expected actual execution pair, not a serving-cell qualification."""
    from ..routed_fused import library_for, routed_class_launch_pair
    from .census import platform_expectation

    del compiled
    if family not in _CENSUS_PAYLOAD_FAMILY:
        raise KeyError(f"{family!r} has no expert stack on this window route")
    pair = routed_class_launch_pair(library_for("value" if family == TESSERA_BF16 else "e4m3"))
    pairs = {"decode": {pair}, "batch": {pair}}
    return platform_expectation(_CENSUS_PAYLOAD_FAMILY[family], platform, pairs)


#: The runtime's shard name -> (group, row block).  DERIVED from
#: ``MOE_GROUP_SHARDS``, so the loader's dispatch and the sidecar's group
#: vocabulary are one table: ``w1`` is block 0 of ``w13`` because ``w1`` is
#: first in that group's shard tuple, which is the order
#: ``RoutedExperts._load_w13`` narrows to.
SHARD_TO_GROUP: dict[str, tuple[str, int]] = {
    shard: (group, index)
    for group, shards in MOE_GROUP_SHARDS.items()
    for index, shard in enumerate(shards)
}


class PreparedTesseraMoeExperts:
    """The four stock tensors an expert stack's wires decode to."""

    __slots__ = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")

    def __init__(self, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale):
        self.w13_weight = w13_weight
        self.w2_weight = w2_weight
        self.w13_weight_scale = w13_weight_scale
        self.w2_weight_scale = w2_weight_scale

    @property
    def experts(self) -> int:
        return int(self.w13_weight.shape[0])


class PreparedTesseraPackedMoeExperts:
    """The shared FP8/window owners for both groups, without decoded weights."""

    def __init__(self, first, second):
        self.__first, self.__second = first, second
        self.experts, self.device = first.experts, first.device

    def resident_bytes(self) -> int:
        return self.__first.resident_bytes() + self.__second.resident_bytes()

    def decode(self, expert_ids, *, max_experts_per_chunk, backend="torch") -> PreparedTesseraMoeExperts:
        return PreparedTesseraMoeExperts(
            self.__first.decode(expert_ids, max_experts_per_chunk=max_experts_per_chunk, backend=backend).view(torch.float8_e4m3fn),
            self.__second.decode(expert_ids, max_experts_per_chunk=max_experts_per_chunk, backend=backend).view(torch.float8_e4m3fn),
            self.__first.row_scale(expert_ids).unsqueeze(-1),
            self.__second.row_scale(expert_ids).unsqueeze(-1))


class PreparedTesseraFoldedBf16MoeExperts:
    """Selected stock BF16 tiles matching the joint screen's PWC render."""

    def __init__(self, w13_weight, w2_weight):
        self.w13_weight, self.w2_weight = w13_weight, w2_weight


class PreparedTesseraPackedBf16MoeExperts:
    """Two selected compressed BF16 groups; no resident decoded expert pool."""

    def __init__(self, first, second):
        self.__first, self.__second = first, second
        self.experts, self.device = first.experts, first.device

    def resident_bytes(self) -> int:
        return self.__first.resident_bytes() + self.__second.resident_bytes()

    def decode_folded(self, expert_ids, *, max_experts_per_chunk, backend="torch"):
        return PreparedTesseraFoldedBf16MoeExperts(
            self.__first.decode_folded(expert_ids, max_experts_per_chunk=max_experts_per_chunk,
                                       backend=backend),
            self.__second.decode_folded(expert_ids, max_experts_per_chunk=max_experts_per_chunk,
                                        backend=backend))


def _parsed_experts(blobs, declared_group, target, device):
    """One parsing/role-validation seam for resident and research owners."""
    for expert, expert_blobs in enumerate(blobs):
        role_declarations = expert_role_declarations(declared_group, expert=expert)
        if len(expert_blobs) != len(role_declarations):
            raise ValueError(
                f"{target} expert {expert}: {len(expert_blobs)} container(s) for "
                f"{len(role_declarations)} declared projection(s) "
                f"{[r['roles'][0][0] for r in role_declarations]}")
        roles = []
        for blob, role in zip(expert_blobs, role_declarations):
            roles.extend(parse_tessera_expert_blob(blob, role, f"{target} expert {expert}", device=device))
        yield roles


def _decode_group(blobs: Sequence[Sequence[bytes]], declared_group: Mapping, target: str,
                  device) -> "tuple[torch.Tensor, torch.Tensor]":
    """One group's E x P containers -> ``([E, rows, cols] uint8, [E, rows, 1] fp32)``.

    ``blobs[e]`` is that expert's containers in the group's ROW order -- gate
    then up for ``w13``, down alone for ``w2`` -- which is the order
    ``RoutedExperts._load_w13`` narrows to, so the stack lands where the
    kernel reads it.

    The reference decoder (``tessera.decode.materialize_fp8``) is what produces
    the served bytes here, so -- unlike the dense route, whose forward runs a
    second, packed-window decoder and must therefore cross-check the two --
    there is no second decoder to disagree with.  What guards the bytes is the
    reader: ``parse_unit_artifact`` verifies the manifest and the payload
    digest from the blob alone before anything is decoded.
    """
    from tessera.decode import materialize_fp8

    rows, columns = int(declared_group["rows"]), int(declared_group["columns"])
    weights, scales = [], []
    for expert, roles in enumerate(_parsed_experts(blobs, declared_group, target, device)):
        role_w, role_s = [], []
        for _name, parsed in roles:
            tile, scale = materialize_fp8(parsed.unit, parsed.forests, parsed.code)
            role_w.append(tile.to(device))
            role_s.append(scale.to(device, torch.float32).reshape(-1))
        weight = torch.cat(role_w, 0) if len(role_w) > 1 else role_w[0]
        scale = torch.cat(role_s, 0) if len(role_s) > 1 else role_s[0]
        if tuple(weight.shape) != (rows, columns):
            raise ValueError(
                f"{target} expert {expert}: the group's roles decode to {tuple(weight.shape)}, "
                f"the sidecar declares ({rows}, {columns})")
        weights.append(weight)
        scales.append(scale)
    return torch.stack(weights, 0), torch.stack(scales, 0).unsqueeze(-1)


def _require_expert_groups(blobs, declared, target):
    experts = int(declared["experts"])
    for group in MOE_GROUPS:
        if len(blobs[group]) != experts:
            raise ValueError(
                f"{target}: group {group!r} carries {len(blobs[group])} expert row(s) for "
                f"{experts} experts; every expert must have its own row of projection containers")


def _packed_group_shard_plan(declared, group, target, tp_rank, tp_size):
    from .sharding import plan_shard

    hidden, inter = int(declared['hidden_size']), int(declared['intermediate_size'])
    local_inter = inter // tp_size
    declaration = declared['groups'][group]
    first = group == 'w13'
    return plan_shard(f"{target}.{group}", roles=declaration['roles'],
            columns=int(declaration['columns']),
            out_partitions=[local_inter, local_inter] if first else [hidden],
            in_size=hidden if first else local_inter, tp_rank=tp_rank, tp_size=tp_size,
            input_size=hidden if first else inter, output_size=2 * inter if first else hidden)


def prepare_tessera_packed_moe_experts(blobs, declared, target, device=None, *, tp_rank=0, tp_size=1):
    """Research load: validate original containers into existing packed owners.

    Each expert is placed on its group's expert axis as soon as it is
    prepared, and the axis checks layout compatibility as ``stack`` does.
    Per-expert bodies/scales/alphabets may differ; heterogeneous gather
    layouts refuse. TP2 validates each original full container before the
    existing dense shard planner/slicer derives rank-local roles. Global
    expert IDs are unchanged. Only a transient per-role FP8 reference is
    materialized during preparation.
    """
    from .fp8_route import PreparedTesseraFp8Module, prepare_tessera_fp8_module
    from .sharding import shard_parsed_roles

    if type(tp_size) is not int or tp_size not in (1, 2):
        raise ValueError(f"{target}: research packed experts cover TP1 or TP2 only")
    if type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
        raise ValueError(f"{target}: invalid research tensor-parallel rank")
    inter = int(declared["intermediate_size"])
    if inter % tp_size:
        raise ValueError(f"{target}: intermediate size must divide the tensor-parallel size")
    device = torch.device("cuda" if device is None else device)
    _require_expert_groups(blobs, declared, target)
    prepared = {}
    for group in MOE_GROUPS:
        declaration = declared['groups'][group]
        plan = _packed_group_shard_plan(declared, group, target, tp_rank, tp_size)
        axis = PreparedTesseraFp8Module.axis(len(blobs[group]),
                                           heterogeneous=expert_rungs_mixed(declaration))
        for expert, roles in enumerate(_parsed_experts(blobs[group], declaration,
                                                       f"{target} {group}", device)):
            axis.put(expert, prepare_tessera_fp8_module(shard_parsed_roles(roles, plan),
                                                        device=device))
        prepared[group] = axis.finish()
    return PreparedTesseraPackedMoeExperts(prepared['w13'], prepared['w2'])


def prepare_tessera_packed_bf16_moe_experts(blobs, declared, target, device=None,
                                            *, tp_rank=0, tp_size=1):
    """Research-only selected BF16 owner over original verified expert wires.

    The selected folded tile matches ``read_unit_artifact(...).to(bfloat16)``,
    the current PrismaQuant joint screen. It is distinct from Tessera's dense
    BF16 route, which applies row scale after the GEMM instead.
    """
    from .bf16_route import PreparedTesseraBf16Module, prepare_tessera_bf16_module
    from .sharding import shard_parsed_roles

    if type(tp_size) is not int or tp_size not in (1, 2):
        raise ValueError(f"{target}: research packed BF16 experts cover TP1 or TP2 only")
    if type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
        raise ValueError(f"{target}: invalid research tensor-parallel rank")
    inter = int(declared["intermediate_size"])
    if inter % tp_size:
        raise ValueError(f"{target}: intermediate size must divide the tensor-parallel size")
    device = torch.device("cuda" if device is None else device)
    _require_expert_groups(blobs, declared, target)
    prepared = {}
    for group in MOE_GROUPS:
        declaration = declared['groups'][group]
        plan = _packed_group_shard_plan(declared, group, target, tp_rank, tp_size)
        axis = PreparedTesseraBf16Module.axis(len(blobs[group]),
                                            heterogeneous=expert_rungs_mixed(declaration))
        for expert, roles in enumerate(_parsed_experts(blobs[group], declaration,
                                                       f"{target} {group}", device)):
            axis.put(expert, prepare_tessera_bf16_module(shard_parsed_roles(roles, plan),
                                                         device=device))
        prepared[group] = axis.finish()
    return PreparedTesseraPackedBf16MoeExperts(prepared['w13'], prepared['w2'])


def _compact_role_units(blob, declared_role, target, device):
    """Read one projection through the grammar-owned compressed reader."""
    from .scheme import parse_compact_tessera_expert_blob

    return parse_compact_tessera_expert_blob(blob, declared_role, target, device=device)


def _compact_expert_units(blob, declared_role, plan, target, *, device, family,
                          scratch=None):
    """ONE projection container -> this rank's ``(role, WindowGemvUnit)``.

    Each load callback carries one projection, so the boundary must return
    exactly one role and it must be the declared member; the plan's own cut
    is applied through ``prepare_window_compact`` (via the dense lane's
    ``_role_cut``, one home for the plan-to-cut mapping), which validates it
    with ``slicing``'s predicates. Wire validation belongs to the shared reader.
    """
    from ..compact_prep import prepare_window_compact
    from .native_window import _role_cut

    roles = _compact_role_units(blob, declared_role, target, device)
    declared = declared_role["roles"][0][0]
    if len(roles) != 1 or roles[0][0] != declared:
        raise ValueError(
            f"{target}: the projection container carries "
            f"{[name for name, _ in roles]} roles; this callback declared the "
            f"single projection {declared!r}")
    name, wire = roles[0]
    return name, prepare_window_compact(
        wire, device=device, family=family, scratch=scratch,
        **_role_cut(plan, name))



def _mixed_axis_word_runs(declared, plans):
    """Exact flat word/run sizes per (group, part, expert) for a mixed stack.

    The sizes are the wire's own arithmetic, read off the DECLARED rungs and
    this rank's shard plan -- and ``WindowUnitAxis.put`` re-validates every
    unit's words and runs against them, so a declared size is a binding check,
    never a tolerated guess.  Per unit: the grammar's canonical schedule over
    the unit's SOURCE columns (``grammar.bresenham_rate_schedule`` of the
    per-code root, capped at the routed window lane's own rate bound), cut to
    the rank's column range where the plan cuts one.  A slice aligned to whole
    quota superblocks is exact for every legal placement -- the manifest
    enforces exactly ``grammar.superblock_quota_ok`` on the wire -- and any
    other slice carries the writer's canonical placement, which the put-time
    check makes binding; the exporter independently refuses a non-aligned
    importance placement whose sizes would differ, before the shard write.
    ``w13`` is row-cut only and keeps the whole schedule; rows pad to the
    512-row tile, so words = ``n_tiles * 16 * sum(local rates)`` and runs are
    one per distinct local rate.

    Returns ``{group: {part: [(words, runs) per expert]}}``, or ``None`` when
    no group is mixed -- a uniform stack predeclares nothing and keeps the
    legacy rectangular allocation.
    """
    if not any(expert_rungs_mixed(declared["groups"][g]) for g in MOE_GROUPS):
        return None
    from fractions import Fraction

    from ..alphabet import grid_for_name
    from ..compact_prep import WINDOW_GEMM_RATE_MAX
    from ..grammar import bresenham_rate_schedule
    from ..kernel_window_gemv import TILE_ROWS
    from .sharding import AXIS_COLUMNS, AXIS_ROWS

    sizes = {}
    for group in MOE_GROUPS:
        group_decl = declared["groups"][group]
        matrix = group_decl.get("expert_role_q256")
        if matrix is None:
            continue
        arity = int(grid_for_name(group_decl["grid"]).arity)
        columns = int(group_decl["columns"])
        plan = plans[group]
        out = {}
        for index, (name, role_rows) in enumerate(group_decl["roles"]):
            name = str(name)
            if plan.axis is None:
                rows_local, c0, c1 = int(role_rows), 0, columns
            elif plan.axis == AXIS_ROWS:
                shard = plan.role(name)
                rows_local, c0, c1 = shard.hi - shard.lo, 0, columns
            else:
                if plan.axis != AXIS_COLUMNS:
                    raise ValueError(
                        f"the {group} plan cuts axis {plan.axis!r}; a routed group cuts rows "
                        "(w13) or columns (w2) only")
                shard = plan.role(name)
                rows_local, c0, c1 = int(role_rows), shard.lo, shard.hi
            if rows_local % arity:
                raise ValueError(
                    f"{group} role {name!r}: this rank's {rows_local} row(s) do not divide "
                    f"into whole {arity}-row codes; the cut is not one the wire can start from")
            n_tiles = -(-(rows_local // arity) // TILE_ROWS)
            per_expert = []
            for e, row in enumerate(matrix):
                root = Fraction(int(row[index]) * arity, 256)
                local = bresenham_rate_schedule(root, columns,
                                                cap=WINDOW_GEMM_RATE_MAX)[c0:c1]
                per_expert.append((n_tiles * 16 * sum(local), len(set(local))))
            out[name] = per_expert
        sizes[group] = out
    return sizes


class _RankLocalPackedIntake:
    """Load storage-ordered expert projections into one exact packed axis."""

    def __init__(self, declared, target, device, tp_rank, tp_size):
        from ..native_window_moe import WindowUnitAxis

        self.declared, self.target, self.device = declared, target, device
        self.family = declared['family']
        if self.family not in (TESSERA_FP8, TESSERA_BF16):
            raise ValueError(f"{target}: packed routed intake has no {self.family} decoder")
        self.plans = {g: _packed_group_shard_plan(declared, g, target, tp_rank, tp_size)
                      for g in MOE_GROUPS}
        self.roles = {g: expert_role_declarations(declared['groups'][g]) for g in MOE_GROUPS}
        self._has_loaded = False
        self._scratch = {}
        self._non_uniform = len(stack_effective_rungs(declared)) > 1
        if self._non_uniform and _piece_major_requested():
            raise ValueError(
                f"{target}: {ENV_PIECE_MAJOR} supports one-run rate-4 stacks only; "
                "a non-uniform class stack must use the common packed word layout")
        self._piece_major = _piece_major_admissible(
            "value" if self.family == TESSERA_BF16 else "e4m3")
        if len(self.roles['w13']) != 2:
            raise ValueError(f"{target}: routed w13 must carry gate and up")
        word_runs = _mixed_axis_word_runs(declared, self.plans)
        self.axis = {g: WindowUnitAxis(
            int(declared['experts']),
            tuple(str(role['roles'][0][0]) for role in self.roles[g]),
            family="value" if self.family == TESSERA_BF16 else "e4m3",
            word_runs=None if word_runs is None else word_runs.get(g)) for g in MOE_GROUPS}

    def placed_projections(self) -> int:
        return sum(axis.filled() for axis in self.axis.values())

    def resident_bytes(self) -> int:
        return sum(axis.resident_bytes() for axis in self.axis.values())

    def load(self, group, index, expert, wire, *, device):
        device = torch.device(device)
        if self._has_loaded and device != self.device:
            raise ValueError(f'{self.target}: packed intake cannot change device after loading starts')
        self.device = device
        logical_expert = self.declared['expert_ids'][expert]
        target = f'{self.target} {group} expert {logical_expert} (storage {expert})'
        if device.type != "cuda":
            raise ValueError(f"{target}: the routed class lane requires a CUDA load device; got {device}")
        blob = wire.detach().cpu().contiguous().numpy().tobytes()
        name, unit = _compact_expert_units(
            blob, expert_role_declarations(self.declared['groups'][group], expert=expert)[index],
            self.plans[group], target, device=device,
            family="value" if self.family == TESSERA_BF16 else "e4m3", scratch=self._scratch)
        from ..kernel_window_gemv import WORD_LAYOUT_PIECE_MAJOR, piece_major_eligible
        if self._piece_major and piece_major_eligible(unit.rep):
            unit = replace(unit, rep=unit.rep.with_word_layout(WORD_LAYOUT_PIECE_MAJOR))
        self.axis[group].put(name, expert, unit)
        self._has_loaded = True

    def finish(self, w13_lengths, w2_lengths):
        from ..native_window_moe import PackedWindowMoeBundles
        from ..window_gemm_grouped import prepare_grouped_window_gemm_from_soa

        validate_moe_wire_lengths(w13_lengths, w2_lengths,
            experts=self.declared['experts'],
            stride13=self.declared['groups']['w13']['wire_stride'],
            stride2=self.declared['groups']['w2']['wire_stride'])
        family = "value" if self.family == TESSERA_BF16 else "e4m3"
        arithmetic = "folded" if self.family == TESSERA_BF16 else "epilogue"
        soa = {g: self.axis[g].finish() for g in MOE_GROUPS}
        self._scratch.clear()
        names = {g: [str(role['roles'][0][0]) for role in self.roles[g]] for g in MOE_GROUPS}

        def bundle(group, part):
            slot = soa[group][part]
            return prepare_grouped_window_gemm_from_soa(
                words_all=slot["words"], table_all=slot["table"],
                codes_all=slot["codes"], native_all=slot["native"],
                scale_all=slot["scale"], runs_all=slot["runs"],
                init_all=slot["init"], has_init=slot["has_init"],
                word_off=slot["word_off"], tile_words=slot["tile_words"],
                total_words=slot["total_words"], run_off=slot["run_off"],
                perm_all=slot["perm"], rows=slot["rows"], cols=slot["cols"],
                experts=int(self.declared['experts']), window_bits=slot["window_bits"],
                family=family, arithmetic=arithmetic,
                word_layout=str(slot.get("word_layout", "legacy")))

        self.axis = {}
        return PackedWindowMoeBundles(
            gate=bundle('w13', names['w13'][0]), up=bundle('w13', names['w13'][1]),
            down=bundle('w2', names['w2'][0]), family=family,
            expert_classes=self.declared['expert_classes'])


def prepare_tessera_moe_experts(blobs: Mapping[str, Sequence[Sequence[bytes]]],
                                declared: Mapping, target: str,
                                device=None) -> PreparedTesseraMoeExperts:
    """``{"w13": [[gate, up]]*E, "w2": [[down]]*E}`` -> the stock per-channel FP8 stack.

    ``declared`` is ``validate_tessera_moe_scheme``'s output.  Every blob is
    parsed against its group's declaration (grid, body, plane, span, roles,
    per-role rung, geometry) before a byte is decoded, so a container that is
    not what the sidecar promised is a refusal rather than a wrong tile.
    """
    device = torch.device("cuda" if device is None else device)
    _require_expert_groups(blobs, declared, target)
    w13, w13_scale = _decode_group(blobs["w13"], declared["groups"]["w13"], f"{target} w13", device)
    w2, w2_scale = _decode_group(blobs["w2"], declared["groups"]["w2"], f"{target} w2", device)
    return PreparedTesseraMoeExperts(
        w13_weight=w13.view(torch.float8_e4m3fn), w2_weight=w2.view(torch.float8_e4m3fn),
        w13_weight_scale=w13_scale, w2_weight_scale=w2_scale)


def _bind_module_prefix(layer, prefix: str) -> bool:
    """Give the routed experts layer the module identity its trace needs.

    The route trace names a module by ``layer.prefix`` (``telemetry.py``,
    ``_RouteTrace.count``), and this layer arrives WITHOUT one: vLLM's fused
    MoE takes ``prefix`` as a constructor argument and stores no attribute, so
    every routed dispatch was counted UNNAMED -- ``module_names: []``,
    ``unnamed_modules: 1``, per entry.  That is a missing module identity, not
    an observed collapse: in the one traced fixture the two routed layers carry
    DIFFERENT policies (``TESSERA_FP8`` on L3, ``TESSERA_BF16`` on L4), so the
    entry key separated them by accident.  Nothing guarantees it.  Entries are
    keyed by policy, shape, symbol, decoder and contract, so two routed modules
    of the SAME policy and shape share ONE ENTRY -- and that entry's identity is
    then names-only.  The count does NOT collapse: unnamed objects are still
    counted per distinct private ``id`` (``telemetry.py``), so two live unnamed
    modules read ``modules=2`` with ``unnamed_modules=2`` and an empty
    ``module_names``.  What is lost is which module each was, not how many ran;
    per-module qualification fails because the names are missing, and a lane
    policy is not a substitute for them.

    The builder is handed the module's real name and already prints it in every
    refusal here, so that is the identity it binds -- and ONLY where the layer
    has no name of its own.  A legitimate ``layer.prefix`` that vLLM set (the
    dense ``LinearBase`` path does set one) is somebody else's fact: it is
    preserved, never overwritten, and never merged with ours.

    Returns True when this call wrote the name.
    """
    if not isinstance(prefix, str) or not prefix:
        return False
    existing = getattr(layer, "prefix", None)
    if existing not in (None, ""):
        return False
    try:
        layer.prefix = prefix
    except (AttributeError, TypeError):  # a slotted/immutable layer: no name
        return False
    return True



def _require_eager_selected_context(config, prefix):
    """Keep model and explicit standalone operator eager evidence distinct."""
    model=getattr(config,"model_config",None)
    if model is not None:
        if getattr(model,"enforce_eager",None) is not True:
            raise ValueError(f"{prefix}: research selected experts require enforce_eager")
        return
    # The native whole-operator factory has no ModelConfig: it never loads a
    # model. Its real CompilationConfig explicitly disables both compilation
    # and graph capture. Do not invent a ModelConfig merely to pass this gate.
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    compilation=getattr(config,"compilation_config",None)
    if (compilation is None or getattr(compilation,"mode",None) is not CompilationMode.NONE
            or getattr(compilation,"cudagraph_mode",None) is not CUDAGraphMode.NONE):
        raise ValueError(f"{prefix}: standalone research selected experts require explicit eager compilation and no CUDA graphs")


def build_tessera_moe_method(scheme: Mapping, prefix: str, mode: str, layer, *,
                             research_selected: ResearchSelectedMoeConfig | None = None):
    """Construct the one native routed class method over storage-ordered wires."""
    if mode not in MODES:
        raise ValueError(f"unknown residency mode {mode!r}")
    if research_selected is not None and not isinstance(research_selected, ResearchSelectedMoeConfig):
        raise ValueError("research_selected requires an explicit ResearchSelectedMoeConfig")
    declared = validate_tessera_moe_scheme(scheme, prefix)
    family = declared["family"]
    from .scheme import refuse_a_family_with_no_expert_route
    refuse_a_family_with_no_expert_route(family, prefix)
    if family not in (TESSERA_FP8, TESSERA_BF16):
        raise ValueError(f"{prefix}: the routed class method has no {family} window decoder")
    _bind_module_prefix(layer, prefix)
    from .backend import require_platform_backs
    from .contract import PAYLOAD_FAMILY_BY_ROUTE
    from .telemetry import record_platform

    require_platform_backs(PAYLOAD_FAMILY_BY_ROUTE[family], f"tessera target {prefix!r}")
    record_platform()
    if mode != MODE_RESIDENT:
        raise ValueError(f"{prefix}: routed class weights require {MODE_RESIDENT!r} residency")
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.model_executor.utils import set_weight_attrs

    groups = declared["groups"]

    class TesseraMoEMethod(FusedMoEMethodBase):
        def __init__(self, moe):
            super().__init__(moe)
            self._mode = mode if research_selected is None else 'research_selected'
            self._research_phase = 'new'
            self._packed = self._native = self._rank_local_intake = self._expert_inverse = None
            self.fp8_backend = self.bf16_backend = self.experts_cls = None
            self._tp_size = (int(moe.moe_parallel_config.tp_size) if research_selected is None
                             else research_selected.expected_tensor_parallel_size)
            if not moe.is_act_and_mul:
                raise ValueError(f"{prefix}: routed class execution requires gated gate/up experts")
            if research_selected is not None:
                from vllm.config import get_current_vllm_config
                _require_eager_selected_context(get_current_vllm_config(), prefix)
                self._require_research_parallel_contract()
            self._tp_rank = int(moe.moe_parallel_config.tp_rank)

        @property
        def is_monolithic(self):
            return False

        @property
        def topk_indices_dtype(self):
            return None

        @property
        def mk_can_overlap_shared_experts(self):
            return False

        @property
        def supports_eplb(self):
            return False

        def _require_research_parallel_contract(self):
            parallel = self.moe.moe_parallel_config
            expected_tp = research_selected.expected_tensor_parallel_size
            if (type(parallel.tp_size) is not int or parallel.tp_size != expected_tp
                    or any(type(getattr(parallel, field, None)) is not int
                           or getattr(parallel, field) != 1
                           for field in ('ep_size', 'dp_size', 'pcp_size', 'sp_size'))
                    or type(getattr(parallel, 'tp_rank', None)) is not int
                    or not 0 <= parallel.tp_rank < expected_tp
                    or getattr(parallel, 'use_ep', None) is not False
                    or getattr(parallel, 'enable_eplb', None) is not False):
                raise ValueError(f"{prefix}: research selected classes require explicit "
                                 f"TP{expected_tp}/EP1/DP1/PCP1/SP1 and no EP/EPLB")
            if getattr(self.moe, 'defer_moe_finalize', False):
                raise ValueError(f"{prefix}: research selected classes refuse deferred finalize")
            if expected_tp > 1 and getattr(self.moe, 'skip_final_all_reduce', False):
                raise ValueError(f"{prefix}: selected TP2 requires the stock final all-reduce")

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype, **extra):
            if self._research_phase != 'new':
                raise RuntimeError(f"{prefix}: routed class owner is already constructed")
            experts = int(declared["experts"])
            global_experts = int(extra.get("global_num_experts", num_experts))
            if int(num_experts) != experts or global_experts != experts:
                raise ValueError(f"{prefix}: this rank holds {num_experts} of {global_experts} "
                                 f"experts; the class table requires all {experts}. EP is unsupported")
            if int(hidden_size) != int(declared["hidden_size"]):
                raise ValueError(f"{prefix}: hidden_size disagrees with the routed scheme")
            full_intermediate = int(declared["intermediate_size"])
            if (full_intermediate % self._tp_size
                    or int(intermediate_size_per_partition) != full_intermediate // self._tp_size):
                raise ValueError(f"{prefix}: intermediate_size_per_partition disagrees with "
                                 f"the scheme at TP{self._tp_size}")
            n_cols = full_intermediate // self._tp_size
            w13_wire = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
            w2_wire = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
            self._rank_local_intake = _RankLocalPackedIntake(
                declared, prefix, w13_wire.device, self._tp_rank, self._tp_size)
            layer.register_parameter("w13_wire", w13_wire)
            layer.register_parameter("w2_wire", w2_wire)
            set_weight_attrs(w13_wire, {"weight_loader": self._load_wire})
            set_weight_attrs(w2_wire, {"weight_loader": self._load_wire})
            layer.tessera_w13_wire_len = torch.zeros(experts, W13_PROJECTIONS, dtype=torch.long)
            layer.tessera_w2_wire_len = torch.zeros(experts, dtype=torch.long)
            self._w13_len, self._w2_len = layer.tessera_w13_wire_len, layer.tessera_w2_wire_len
            self._wire_ids = {'w13': id(w13_wire), 'w2': id(w2_wire)}
            layer.w13_input_scale = layer.w2_input_scale = None
            layer.tessera_mode, layer.tessera_family = self._mode, family
            layer.tessera_structure = declared["structure"]
            layer.tessera_activation_contract = ROUTES[family]["activation_contract"]
            layer.tessera_rows, layer.tessera_columns = 2 * n_cols, int(hidden_size)
            self._research_phase = 'loading'

        def _load_wire(self, param, loaded_weight, weight_name, shard_id, expert_id,
                       return_success=False):
            group, index = SHARD_TO_GROUP.get(str(shard_id), (None, None))
            if group is None:
                raise ValueError(f"{prefix}: shard_id {shard_id!r} does not name an expert projection")
            if self._research_phase != 'loading':
                raise RuntimeError(f"{prefix}: routed class owner is not loading")
            if type(expert_id) is not int or not 0 <= expert_id < int(declared['experts']):
                raise ValueError(f"{prefix}: invalid storage expert ID {expert_id!r}")
            if id(param) != self._wire_ids[group]:
                raise ValueError(f"{prefix}: wire parameter does not belong to group {group}")
            previous = self._w13_len[expert_id, index] if group == 'w13' else self._w2_len[expert_id]
            if int(previous) != 0:
                raise ValueError(f"{prefix}: storage expert {expert_id} shard {shard_id} already loaded")
            blob = loaded_weight.reshape(-1)
            if blob.dtype != torch.uint8:
                raise ValueError(f"{prefix}: expert wire must be uint8, got {blob.dtype}")
            length, stride = int(blob.numel()), int(groups[group]["wire_stride"])
            if length == 0 or length > stride:
                raise ValueError(f"{prefix}: {length}-byte expert wire does not fit wire_stride={stride}")
            try:
                device = param.device
                if device.type != 'cuda' and torch.cuda.is_available():
                    device = torch.device('cuda', torch.cuda.current_device())
                self._rank_local_intake.load(group, index, expert_id, blob, device=device)
            except Exception:
                self._research_phase = 'failed'
                self._rank_local_intake = None
                raise
            if group == 'w13':
                self._w13_len[expert_id, index] = length
            else:
                self._w2_len[expert_id] = length
            return True if return_success else None

        def process_weights_after_loading(self, layer):
            from ..expert_classes import inverse_expert_ids

            if self._research_phase != 'loading':
                raise RuntimeError(f"{prefix}: routed class owner is not loading")
            self._research_phase = 'failed'
            intake, self._rank_local_intake = self._rank_local_intake, None
            prepared = intake.finish(self._w13_len, self._w2_len)
            self._native = prepared.adapter()
            self._packed = prepared.native_owner()
            self._expert_inverse = torch.tensor(inverse_expert_ids(declared['expert_ids']),
                                                dtype=torch.int32, device=self._packed.device)
            del layer.w13_wire, layer.w2_wire
            layer.tessera_w13_wire_len = layer.tessera_w2_wire_len = None
            self._w13_len = self._w2_len = self._wire_ids = None
            layer.tessera_decoder = self._native.launch_pair[1]
            layer.tessera_backend = 'native'
            self._research_phase = 'ready'

        def get_fused_moe_quant_config(self, layer):
            return None

        def resident_tensors(self, layer):
            if self._packed is not None:
                yield from named_resident_tensors(self._packed, 'tessera_packed')
            if self._expert_inverse is not None:
                yield 'tessera_expert_inverse', self._expert_inverse

        def research_resident_bytes(self):
            if research_selected is None or self._research_phase != 'ready':
                raise RuntimeError(f"{prefix}: research class owner is not ready")
            return self._packed.resident_bytes() + self._expert_inverse.untyped_storage().nbytes()

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                  shared_experts_input):
            if self._research_phase != 'ready':
                raise RuntimeError(f"{prefix}: routed class weights have not finished loading")
            if x.ndim != 2 or x.shape[1] != int(declared['hidden_size']):
                raise ValueError(f"{prefix}: native input must be [tokens, {declared['hidden_size']}]")
            if (tuple(topk_ids.shape) != tuple(topk_weights.shape)
                    or topk_ids.ndim != 2 or topk_ids.shape[0] != x.shape[0]):
                raise ValueError(f"{prefix}: native routing must be [tokens, top_k]")
            if topk_ids.dtype not in (torch.int32, torch.int64) or not topk_weights.is_floating_point():
                raise ValueError(f"{prefix}: native routing needs integer IDs and floating weights")
            if any(t.device != self._packed.device for t in (x, topk_ids, topk_weights)):
                raise ValueError(f"{prefix}: routing and packed classes must share one device")
            if research_selected is not None:
                self._require_research_parallel_contract()
                if self.moe.moe_parallel_config.tp_rank != self._tp_rank:
                    raise ValueError(f"{prefix}: research rank changed after construction")
            limit = self._require_native_contract(layer)
            if x.shape[0] == 0:
                return x.new_empty((0, int(declared['hidden_size'])))
            # index_select rejects negative IDs rather than wrapping them like advanced indexing.
            stored_ids = self._expert_inverse.index_select(0, topk_ids.reshape(-1)).reshape_as(topk_ids)
            with torch.profiler.record_function(_profiler_label(self._native)):
                out = native_call(
                    self._native, x.contiguous(), stored_ids, topk_weights,
                    shared_experts=shared_experts, shared_experts_input=shared_experts_input,
                    swiglu_limit=limit,
                    apply_router_weight_on_input=bool(getattr(layer, 'apply_router_weight_on_input', False)))
            self._record(layer, x)
            return out

        def apply_monolithic(self, layer, x, router_logits, input_ids=None):
            raise ValueError(f"{prefix}: routed classes require external top-k routing")

        def _require_native_contract(self, layer):
            from ..native_window_moe import checked_swiglu_limit

            activation = getattr(layer, 'activation', 'silu')
            if str(getattr(activation, 'value', activation)).lower() != 'silu':
                raise ValueError(f"{prefix}: routed classes serve silu only")
            for field in ('swiglu_alpha', 'swiglu_beta'):
                if getattr(layer, field, None) is not None:
                    raise ValueError(f"{prefix}: {field} is not served by the routed class kernel")
            try:
                return checked_swiglu_limit(getattr(layer, 'swiglu_limit', None), where=f"{prefix}: ")
            except GrammarError as exc:
                raise ValueError(str(exc)) from exc

        def _record(self, layer, x):
            try:
                symbol, decoder = self._native.launch_pair
                emit_route(layer, kind="moe", policy=f"{family}:{layer.tessera_mode}",
                           symbol=symbol, tile_m=0,
                           shape=route_shape(x, layer.tessera_rows, layer.tessera_columns),
                           contract=layer.tessera_activation_contract, state="served", reason=None,
                           decoder=decoder, kernel_schedule=symbol)
            except Exception:  # telemetry must not break a request
                pass

    return TesseraMoEMethod(layer.moe_config)
