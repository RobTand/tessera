"""Tessera routed-MoE on the NVFP4 route: E2M1x2 expert wires served W4A4.

WHAT IT SERVES (tessera#492).  One ``tessera.fused`` container per expert per
projection, decoded ONCE at load through ``tessera.stock.materialize_stock``
-- the same decoder the dense NVFP4 route cross-checks its native kernel
against -- into exactly the parameter set vLLM's own ``ModelOptNvFp4FusedMoE``
builds for a modelopt NVFP4 checkpoint: ``w13_weight [E, 2N, K/2]`` and
``w2_weight [E, K, N/2]`` packed E2M1 nibbles, ``w13_weight_scale [E, 2N,
K/16]`` and ``w2_weight_scale [E, K, N/16]`` ue4m3 block scales, one fp32
global per expert per group (``w13_weight_scale_2 [E, 2]``,
``w2_weight_scale_2 [E]``) and one static fp32 activation scale per expert
per group (``w13_input_scale [E, 2]``, ``w2_input_scale [E]``).  From
``process_weights_after_loading`` onward this route IS that class: the same
``convert_to_nvfp4_moe_kernel_format``, the same
``make_nvfp4_moe_quant_config``, the same ``make_nvfp4_moe_kernel`` over the
backend the runtime's own ``select_nvfp4_moe_backend`` picked for
``(kNvfp4Static, kNvfp4Dynamic)`` on this box, and an ``apply`` that hands the
runtime's modular kernel the runtime's own tensors.  Nothing here writes a
kernel, and the A side -- group-16 dynamic E2M1 quantisation of every token
under the static per-expert scale -- is the kernel's, which is what makes the
executed contract ``ROUTES[TESSERA_NVFP4]["activation_contract"]`` and not a
claim this module makes.

THE GLOBAL IS SHARED PER EXPERT, NOT PER STACK.  The gate and up units of one
expert are encoded on their own LUT globals; ``w13`` is one tile per expert
whose two halves the kernel reads under ONE ``weight_scale_2`` (the stock
method warns and takes ``[:, 0]`` when the halves differ -- a silent wrong
answer, not a refusal).  So the loader waits for both halves of an expert,
joins them through ``fused.shared_lut_global`` -- the same power-of-two move
the dense route makes for q/k/v and gate/up -- and writes the joined
multiplier into both columns.  The down unit is its own tile and keeps its
own global.  A whole-stack global would move every expert's block scales onto
the range of the widest one and is not what the kernel needs.

THE A-SIDE SCALE IS A CHECKPOINT FACT.  W4A4 needs a static input scale per
expert projection, calibrated by the producer and written beside each wire as
``experts.{e}.{proj}.input_global_scale`` -- the SAME quantity the dense route
reads as ``trellis_input_global_scale``: capacity over amax, the value vLLM's
quantiser multiplies by.  modelopt stores its reciprocal (``amax / (448 * 6)``)
as ``input_scale``, so the loader inverts once at
``process_weights_after_loading`` and the kernel reads what it always reads.
A missing or non-positive scale is a refusal: an uninitialised input scale is
a stack that serves garbage at a plausible loss.

WHAT IS RANK-LOCAL.  At TP>1 the stock method slices ``w13`` rows and ``w2``
columns by rank at load.  Here each expert's FULL container is parsed and
verified (digest, geometry, role, recipe against the sidecar) and then cut by
``sharding.shard_parsed_roles`` on the group's plan -- a row cut for ``w13``,
a column cut for ``w2`` -- before decode, so a rank decodes and holds only its
own rows: the span-2 trellis register at the cut is threaded into the select
pad by ``lane_planes`` (tessera#492) and decodes ``torch.equal`` to the whole
unit's rows.  Global expert ids are unchanged; the final reduction is stock
vLLM's.

WHAT THIS ROUTE REFUSES.  Expert parallelism and EPLB (the stride invariant
needs every expert's blob and the parameter is ``[E, ...]`` by global id); a
residency mode other than ``resident``; a non-gated MoE; an expert count,
hidden size or intermediate size that disagrees with the sidecar, or a
rank-local intermediate width that is not a whole number of 16-wide groups;
an expert whose gate arrived without its up; a stock tensor name
(``experts.{e}.{proj}.weight`` and friends) in a Tessera checkpoint.

WHAT IS ATTESTED.  Being in ``scheme.MOE_BUILDERS`` is a dispatch fact.  The
served ``routed_moe`` cells for this family are ``lane_eligibility``'s to
publish, per image and per regime, from a container receipt.  Contract v28
publishes two, at q256 896, eager and resident, on the image the two-rank
GLM-5.3-Flash 4-layer stub served
(``docs/measurements/tessera-glm53-a4-stub-tp2-served-2026-09-14.md``).  An
export at another rung, or a serve outside that scope, is unattested, and an
export of it needs ``--allow-unserveable`` and says so in its manifest.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch

from ..errors import GrammarError
from ..moe_layout import W13_PROJECTIONS, validate_moe_wire_lengths
from .lane import MODE_RESIDENT, MODES
from .moe_route import SHARD_TO_GROUP, _packed_group_shard_plan
from .scheme import (GROUP_SIZE, MOE_GEMM_SYMBOL, MOE_GROUPS, ROUTES,
                     STRUCTURE_ROUTED_MOE, TESSERA_NVFP4, expert_role_declarations,
                     launch_pairs, moe_census_symbol_base as census_symbol_base,
                     parse_tessera_expert_blob, route_launches,
                     validate_tessera_moe_scheme)
from .telemetry import DECODER_TORCH_STOCK, emit_route, route_shape

__all__ = [
    "ACTIVATION_CONTRACT",
    "GEMM_SYMBOL",
    "INPUT_GLOBAL_SCALE_SUFFIX",
    "PAYLOAD_FAMILY",
    "census_expected",
    "census_symbol_base",
    "decode_expert_tile",
    "build_tessera_nvfp4_moe_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_NVFP4]["activation_contract"]
GEMM_SYMBOL = MOE_GEMM_SYMBOL
#: The contract's payload family for this route's wires -- the name the
#: platform gate and the census expectation are keyed by (#457).
PAYLOAD_FAMILY = "TESSERA_E2M1_K2"
#: The checkpoint suffix of the per-expert static A-side scale, written beside
#: each expert projection's ``wire`` by the exporter and routed by vLLM's
#: expert-parameter mapping onto ``{group}_input_global_scale`` exactly as
#: ``wire`` is routed onto ``{group}_wire``.
INPUT_GLOBAL_SCALE_SUFFIX = "input_global_scale"

#: The stock tensor names this route allocates and vLLM's kernels read.  Any
#: checkpoint tensor that lands on one of them is a modelopt/compressed-tensors
#: checkpoint being served through a Tessera scheme, and is refused by name.
_STOCK_TILE_NAMES = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
                     "w13_weight_scale_2", "w2_weight_scale_2",
                     "w13_input_scale", "w2_input_scale")


def census_expected(*, compiled: bool = False, platform=None) -> dict:
    """The ``(symbol, decoder)`` pairs an NVFP4 expert stack may report, by regime.

    The same shape as ``moe_route.census_expected`` for the same reason: one
    launch in both regimes (the tile is materialised once at load; every
    forward hands it to the runtime's modular kernel), so ``compiled`` changes
    nothing, and the symbol's backend suffix is the runtime's answer rather
    than this route's promise -- a census compares :func:`census_symbol_base`.
    Per ``(platform, family)`` (#457): the stack's payload family is the dense
    NVFP4 route's, so a platform that executes no E2M1_K2 route executes none
    for the experts either.
    """
    del compiled  # one launch has nothing to combine
    launches = route_launches(TESSERA_NVFP4, structure=STRUCTURE_ROUTED_MOE,
                              mode=MODE_RESIDENT)
    regimes = {regime for launch in launches for regime in launch["regimes"]}
    pairs = {regime: launch_pairs(TESSERA_NVFP4, structure=STRUCTURE_ROUTED_MOE,
                                  regime=regime, mode=MODE_RESIDENT)
             for regime in regimes}
    from .census import platform_expectation

    return platform_expectation(PAYLOAD_FAMILY, platform, pairs)


def decode_expert_tile(parsed_roles, device):
    """``[(name, parsed)]`` for ONE expert group -> ``(packed, scale, global)``.

    ``packed`` is ``[rows, cols/2]`` uint8 nibbles, ``scale`` is ``[rows,
    cols/16]`` float8_e4m3fn block scales and ``global`` is the fp32
    MULTIPLIER the kernel folds into its alpha (modelopt ``weight_scale_2``,
    the reciprocal of the stock tile's ``weight_global_scale`` divisor) --
    ONE value, shared across the roles handed in, so a two-role ``w13`` decodes
    onto the one global its tile carries.  The bytes are
    ``tessera.stock.materialize_stock``'s after ``fused.shared_lut_global``'s
    power-of-two move, the same pair the dense route's fallback tile is built
    from (``ops._torch_fallback_tile``), so what the fused-MoE kernel reads is
    what the stock lane was measured on.
    """
    from ..fused import shared_lut_global
    from .ops import _torch_fallback_tile

    shared, moved = shared_lut_global(
        [parsed.unit.scale_lut for _name, parsed in parsed_roles],
        [float(parsed.unit.scale_global) for _name, parsed in parsed_roles],
        [name for name, _parsed in parsed_roles])
    packed, scale = _torch_fallback_tile(parsed_roles, moved, shared, device)
    return packed, scale.view(torch.float8_e4m3fn), float(shared)


class _ExpertIntake:
    """Per-rank intake: parse the full container, cut to this rank, hold ``w13``
    halves until an expert has both, hand back what is ready to decode."""

    def __init__(self, declared, target, tp_rank, tp_size):
        self.target = target
        self.plans = {g: _packed_group_shard_plan(declared, g, target, tp_rank, tp_size)
                      for g in MOE_GROUPS}
        self.roles = {g: expert_role_declarations(declared["groups"][g]) for g in MOE_GROUPS}
        self.pending: dict[int, list] = {}

    def take(self, group, index, expert, blob: bytes, device):
        """One verified container -> this rank's native bundles for its role.

        The compact reader runs the same metadata verification the parsed
        reader does (digests, canonical padding, slack, geometry, the shard
        record) and repacks the packed BODY straight into the kernel planes:
        no parent-plane expansion and no decoded stock tile.  w13's halves are
        held until both arrive, then their 16-entry tables are moved onto one
        shared global exactly as the stock lane moves them
        (``fused.shared_lut_global``), because a fused tile carries one
        weight global.
        """
        from ..serving import scheme as scheme_module
        from .native_a4 import prepare_a4_unit

        target = self.target
        declared_role = self.roles[group][index]
        expected = declared_role["roles"][0][0]
        plan = self.plans[group]

        def _cuts(shard):
            if plan.axis == "row":
                return (shard.lo, shard.hi), None
            if plan.axis == "column":
                return None, (shard.lo, shard.hi)
            return None, None

        # The shared owner's factored compact validator (same container
        # framing, role list and per-role byte facts as the parsed reader,
        # with no weight-plane expansion).  Until it lands, the parsed
        # validator runs instead -- stricter and slower, never weaker; the
        # dependency is published in native-a4/INTEGRATION-REQUEST.md.
        validator = getattr(scheme_module, "parse_compact_tessera_expert_blob", None)
        if validator is not None:
            validated = validator(blob, declared_role, f"{target} {group} expert {expert}",
                                  device=device)
            if len(validated) != 1:
                raise GrammarError(
                    f"{target} {group} expert {expert}: an expert projection container "
                    f"holds one role, this one frames {len(validated)}")
            name, member = validated[0]
            rows, cols = _cuts(plan.role(name))
            unit = prepare_a4_unit(member, rows=rows, cols=cols)
        else:
            from ..kernel_a4 import A4Unit
            from ..lane_planes import prepare_span2_planes
            from .sharding import shard_parsed_roles

            validated = parse_tessera_expert_blob(
                blob, declared_role, f"{target} {group} expert {expert}", device=device)
            local = shard_parsed_roles(validated, plan)
            if local[0][0] != expected:
                raise GrammarError(
                    f"{target} {group} expert {expert}: the container frames role "
                    f"{local[0][0]!r}, the sidecar declares {expected!r}")
            unit = A4Unit.from_prepared(prepare_span2_planes(local[0][1], device=device))
        if group != "w13":
            return ([(expected, unit)], float(unit.global_scale))
        halves = self.pending.setdefault(expert, [None] * W13_PROJECTIONS)
        halves[index] = (expected, unit)
        if any(half is None for half in halves):
            return None
        del self.pending[expert]
        names = [half[0] for half in halves]
        shared, moved = shared_lut_global(
            [half[1].lut_bytes for half in halves],
            [half[1].global_scale for half in halves], names)
        import dataclasses

        joined = [
            (name, dataclasses.replace(unit, lut_bytes=table.view(torch.uint8)
                                       .view(torch.float8_e4m3fn).contiguous(),
                                       global_scale=float(shared)))
            for (name, unit), table in zip(halves, moved)
        ]
        return (joined, float(shared))


def build_tessera_nvfp4_moe_method(scheme: Mapping, prefix: str, mode: str, layer):
    """Construct the vLLM fused-MoE method serving a Tessera NVFP4 expert stack.

    ``layer`` is the ``RoutedExperts`` being built: its ``moe_config`` is what
    the runtime's backend oracle is asked about, so it is a constructor
    argument rather than something read back later.
    """
    if mode not in MODES:
        raise ValueError(f"unknown residency mode {mode!r}")
    declared = validate_tessera_moe_scheme(scheme, prefix)
    family = declared["family"]
    if family != TESSERA_NVFP4:
        raise ValueError(
            f"tessera target {prefix!r}: this builder serves {TESSERA_NVFP4} expert stacks "
            f"and the sidecar declares {family}; scheme.MOE_BUILDERS names each family's own")
    from .scheme import refuse_a_family_with_no_expert_route
    refuse_a_family_with_no_expert_route(family, prefix)
    # THE PLATFORM GATE FOR THE EXPERT ROUTE (#457), asked at this builder's
    # own front door and before the vLLM fused-MoE imports below, exactly as
    # ``moe_route.build_tessera_moe_method`` asks it.
    from .backend import require_platform_backs
    from .contract import PAYLOAD_FAMILY_BY_ROUTE
    from .telemetry import record_platform

    require_platform_backs(PAYLOAD_FAMILY_BY_ROUTE[family], f"tessera target {prefix!r}")
    record_platform()   # latched eagerly: see lane.build_tessera_method
    if mode != MODE_RESIDENT:
        raise ValueError(
            f"tessera target {prefix!r}: the expert route serves {MODE_RESIDENT!r} only. A "
            "streamed expert stack would decode E x 2 containers inside every forward, which "
            "is a different kernel story than the dense streamed route's and carries no "
            "measurement; refusing is what keeps 'streamed' meaning one thing.")

    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        convert_to_nvfp4_moe_kernel_format, make_nvfp4_moe_kernel,
        make_nvfp4_moe_quant_config, select_nvfp4_moe_backend)
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic, kNvfp4Static)
    from vllm.model_executor.utils import replace_parameter, set_weight_attrs

    groups = declared["groups"]
    experts = int(declared["experts"])
    hidden = int(declared["hidden_size"])
    full_intermediate = int(declared["intermediate_size"])

    class TesseraNvFp4MoEMethod(FusedMoEMethodBase):
        """NVFP4 W4A4 routed experts, decoded from Tessera E2M1x2 wires."""

        def __init__(self, moe) -> None:
            super().__init__(moe)
            parallel = moe.moe_parallel_config
            if not moe.is_act_and_mul:
                raise ValueError(
                    f"tessera target {prefix!r}: this MoE is not gated (is_act_and_mul is "
                    "False), so its w13 is one shard rather than the gate/up pair the "
                    "sidecar's groups describe. Refusing rather than loading a pair into a "
                    "single-shard tile.")
            if (getattr(parallel, "use_ep", False) or int(getattr(parallel, "ep_size", 1)) != 1
                    or getattr(parallel, "enable_eplb", False)):
                raise ValueError(
                    f"tessera target {prefix!r}: expert parallelism / EPLB is refused. The "
                    "wire stride is the maximum over EVERY expert's blob and the parameter "
                    "holds one row per global expert id, so a rank holding a subset or a "
                    "redundant physical expert cannot be checked against the sidecar.")
            self._tp_size = int(parallel.tp_size)
            self._tp_rank = int(parallel.tp_rank)
            if self._tp_size < 1 or not 0 <= self._tp_rank < self._tp_size:
                raise ValueError(f"tessera target {prefix!r}: invalid tensor-parallel "
                                 f"rank {self._tp_rank} of {self._tp_size}")
            # The runtime picks the backend, from the runtime's own predicate,
            # for the keys this route's tile actually is: static per-expert
            # NVFP4 weights, dynamically NVFP4-quantised activations.
            self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
                config=self.moe, weight_key=kNvfp4Static, activation_key=kNvfp4Dynamic)
            self._intake = None
            self._tiles = None
            self._w13_len = self._w2_len = None
            self._input_global = None

        @property
        def supports_eplb(self) -> bool:
            return False

        # -- load -------------------------------------------------------
        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype, **extra):
            global_experts = int(extra.get("global_num_experts", num_experts))
            if int(num_experts) != experts or global_experts != experts:
                raise ValueError(
                    f"tessera target {prefix!r}: this rank holds {num_experts} of "
                    f"{global_experts} experts and the sidecar declares {experts}. The wire "
                    "stride is the maximum over EVERY expert's blob, so a rank holding a "
                    "subset cannot check it -- expert parallelism is refused here rather "
                    "than served on an unverifiable stride.")
            if int(hidden_size) != hidden:
                raise ValueError(
                    f"tessera target {prefix!r}: vLLM builds hidden_size="
                    f"{hidden_size}, the sidecar declares {hidden}")
            if (full_intermediate % self._tp_size
                    or int(intermediate_size_per_partition) != full_intermediate // self._tp_size):
                raise ValueError(
                    f"tessera target {prefix!r}: this rank's intermediate size is "
                    f"{intermediate_size_per_partition} and the sidecar declares "
                    f"{full_intermediate} at TP{self._tp_size}. Runtime padding or a cut "
                    "the sidecar's geometry does not divide into is refused.")
            local = full_intermediate // self._tp_size
            if local % GROUP_SIZE or hidden % GROUP_SIZE:
                raise ValueError(
                    f"tessera target {prefix!r}: the rank-local tile is {2 * local}x{hidden} "
                    f"(w13) and {hidden}x{local} (w2); both widths must be whole "
                    f"{GROUP_SIZE}-wide block-scale groups for the NVFP4 mainloop.")
            n_rows = 2 * local

            # The wires and the A-side scales: zero-byte anchors whose only
            # job is to carry a loader.  ``RoutedExperts.load_weights`` calls
            # ``param.weight_loader``; the stack's own loader dispatches on
            # the substrings "weight"/"scale" and would drop ``wire`` in
            # silence (``docs/measurements/tessera-moe-wire-loader-2026-09-03.md``).
            w13_wire = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
            w2_wire = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
            layer.register_parameter("w13_wire", w13_wire)
            layer.register_parameter("w2_wire", w2_wire)
            set_weight_attrs(w13_wire, {"weight_loader": self._load_wire})
            set_weight_attrs(w2_wire, {"weight_loader": self._load_wire})
            # NaN until loaded: the finalizer refuses any expert projection
            # whose scale never arrived rather than serving it at 1.0.
            w13_input_global = torch.nn.Parameter(
                torch.full((experts, W13_PROJECTIONS), math.nan, dtype=torch.float32),
                requires_grad=False)
            w2_input_global = torch.nn.Parameter(
                torch.full((experts,), math.nan, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w13_input_global_scale", w13_input_global)
            layer.register_parameter("w2_input_global_scale", w2_input_global)
            set_weight_attrs(w13_input_global, {"weight_loader": self._load_input_global_scale})
            set_weight_attrs(w2_input_global, {"weight_loader": self._load_input_global_scale})
            self._input_global = {"w13": w13_input_global, "w2": w2_input_global}

            # The stock modelopt names stay registered as ZERO-SIZE anchors,
            # each with the refusing loader: a checkpoint carrying a stock
            # tensor is still refused by name, while the 4.5-bpp expanded pool
            # this route exists to avoid is never allocated.  The stock
            # ``ModelOptNvFp4FusedMoE`` parameter set is not needed because no
            # modular kernel runs on this route.
            self._tiles = {}
            for name in _STOCK_TILE_NAMES:
                param = torch.nn.Parameter(torch.empty(0), requires_grad=False)
                layer.register_parameter(name, param)
                set_weight_attrs(param, {"weight_loader": self._refuse_stock_tensor})
                self._tiles[name] = param
            # NOT parameters: what the loader learned from the blobs it was
            # handed, checked against the declared stride at finalize.
            layer.tessera_w13_wire_len = torch.zeros(experts, W13_PROJECTIONS, dtype=torch.long)
            layer.tessera_w2_wire_len = torch.zeros(experts, dtype=torch.long)
            self._w13_len = layer.tessera_w13_wire_len
            self._w2_len = layer.tessera_w2_wire_len
            self._intake = _ExpertIntake(declared, prefix, self._tp_rank, self._tp_size)
            # One axis per (group, role): each is one stacked allocation per
            # plane kind over the expert axis, filled as containers arrive.
            from .native_a4 import A4ExpertAxis

            self._axes = {
                (group_name, role.name): A4ExpertAxis(experts)
                for group_name in MOE_GROUPS
                for role in self._intake.roles[group_name]
            }
            self._shared_w13 = {}
            self._shared_w2 = {}
            layer.tessera_mode = mode
            layer.tessera_family = family
            layer.tessera_structure = declared["structure"]
            layer.tessera_activation_contract = ACTIVATION_CONTRACT
            layer.tessera_rows = n_rows
            layer.tessera_columns = hidden

        def _refuse_stock_tensor(self, param, loaded_weight, weight_name, shard_id, expert_id,
                                 return_success: bool = False):
            raise ValueError(
                f"tessera target {prefix!r} expert {expert_id} {shard_id}: the checkpoint "
                f"carries a stock tensor ({weight_name!r}) for a stack the sidecar declares "
                "as Tessera wires. A Tessera expert projection is ``wire`` plus "
                f"``{INPUT_GLOBAL_SCALE_SUFFIX}``; refusing rather than overwriting a "
                "decoded tile with checkpoint bytes.")

        def _group_of(self, shard_id, expert_id):
            group, index = SHARD_TO_GROUP.get(str(shard_id), (None, None))
            if group is None:
                raise ValueError(
                    f"tessera target {prefix!r}: shard_id {shard_id!r} is not one of the "
                    f"shards the expert groups hold ({sorted(SHARD_TO_GROUP)})")
            if type(expert_id) is not int or not 0 <= expert_id < experts:
                raise ValueError(
                    f"tessera target {prefix!r}: expert id {expert_id!r} is outside the "
                    f"{experts} experts the sidecar declares")
            return group, index

        def _decode_device(self):
            device = self._tiles["w13_weight"].device
            if device.type != "cuda" and torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            return device

        def _load_wire(self, param, loaded_weight, weight_name, shard_id, expert_id,
                       return_success: bool = False):
            """Validate one full container, cut it to this rank, decode it into the
            expert's rows of the stock tile, and drop the planes."""
            if self._intake is None:
                raise RuntimeError(f"tessera target {prefix!r}: wires arrived after finalize")
            group, index = self._group_of(shard_id, expert_id)
            blob = loaded_weight.reshape(-1)
            if blob.dtype != torch.uint8:
                raise ValueError(
                    f"tessera target {prefix!r} expert {expert_id} {shard_id}: a Tessera wire "
                    f"is uint8 bytes, the checkpoint holds {blob.dtype}")
            length = int(blob.numel())
            stride = int(groups[group]["wire_stride"])
            if length == 0 or length > stride:
                raise ValueError(
                    f"tessera target {prefix!r} expert {expert_id} {shard_id}: a {length}-byte "
                    f"wire does not fit the group's declared wire_stride={stride}; the sidecar "
                    "and the bytes disagree about the row width")
            previous = self._w13_len[expert_id, index] if group == "w13" else self._w2_len[expert_id]
            if int(previous) != 0:
                raise ValueError(
                    f"tessera target {prefix!r}: expert {expert_id} {shard_id} already loaded")
            device = self._decode_device()
            ready = self._intake.take(
                group, index, expert_id, blob.detach().cpu().contiguous().numpy().tobytes(),
                device)
            if ready is not None:
                units, shared = ready
                for name, unit in units:
                    self._axes[(group, name)].put(expert_id, unit)
                if group == "w13":
                    self._shared_w13[expert_id] = shared
                else:
                    self._shared_w2[expert_id] = shared
                del units, ready
            if group == "w13":
                self._w13_len[expert_id, index] = length
            else:
                self._w2_len[expert_id] = length
            return True if return_success else None

        def _load_input_global_scale(self, param, loaded_weight, weight_name, shard_id,
                                     expert_id, return_success: bool = False):
            if self._input_global is None:
                raise RuntimeError(f"tessera target {prefix!r}: scales arrived after finalize")
            group, index = self._group_of(shard_id, expert_id)
            value = loaded_weight.reshape(-1)
            if value.numel() != 1 or not value.is_floating_point():
                raise ValueError(
                    f"tessera target {prefix!r} expert {expert_id} {shard_id}: "
                    f"{INPUT_GLOBAL_SCALE_SUFFIX} is one floating scalar, the checkpoint "
                    f"holds {tuple(loaded_weight.shape)} {loaded_weight.dtype}")
            scale = float(value.float()[0])
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"tessera target {prefix!r} expert {expert_id} {shard_id}: "
                    f"{INPUT_GLOBAL_SCALE_SUFFIX}={scale!r} is not a finite positive value")
            target = self._input_global[group]
            if id(param) != id(target):
                raise ValueError(
                    f"tessera target {prefix!r}: {weight_name!r} does not belong to group {group}")
            slot = target.data[expert_id, index] if group == "w13" else target.data[expert_id]
            if not bool(torch.isnan(slot)):
                raise ValueError(
                    f"tessera target {prefix!r}: expert {expert_id} {shard_id} "
                    f"{INPUT_GLOBAL_SCALE_SUFFIX} already loaded")
            if group == "w13":
                target.data[expert_id, index] = scale
            else:
                target.data[expert_id] = scale
            return True if return_success else None

        def process_weights_after_loading(self, layer) -> None:
            if self._intake is None:
                raise RuntimeError(f"tessera target {prefix!r}: already finalized")
            validate_moe_wire_lengths(
                self._w13_len, self._w2_len, experts=experts,
                stride13=int(groups["w13"]["wire_stride"]),
                stride2=int(groups["w2"]["wire_stride"]))
            if self._intake.pending:
                raise ValueError(
                    f"tessera target {prefix!r}: expert(s) {sorted(self._intake.pending)[:8]} "
                    "loaded one half of w13 and not the other")
            for group in MOE_GROUPS:
                scales = self._input_global[group].data
                bad = ~(torch.isfinite(scales) & (scales > 0))
                if bool(bad.any()):
                    where = bad.nonzero().tolist()
                    raise ValueError(
                        f"tessera target {prefix!r}: {int(bad.sum())} expert projection(s) in "
                        f"{group} carry no {INPUT_GLOBAL_SCALE_SUFFIX} (first {where[:8]}). W4A4 "
                        "needs a static A-side scale per expert projection; refusing rather "
                        "than quantising activations at 1.0.")
            # modelopt's ``input_scale`` is the reciprocal of Tessera's
            # ``input_global_scale`` (amax / capacity vs capacity / amax).
            input_small = {group: 1.0 / self._input_global[group].data
                           for group in MOE_GROUPS}
            # The selected FlashInfer MoE backends aggregate the routed
            # activation scale per layer/projection rather than per expert
            # (``flashinfer_fp4_moe.py``'s
            # ``is_global_sf_supported_for_nvfp4_backend`` computes it through
            # ``amax_for_moe_activation_quant`` and repeats it across the
            # experts; the oracle collapses a disagreeing vector the same way),
            # so one quantizer scalar per GEMM is the served contract.  The
            # native path reproduces that reduction instead of quantising per
            # expert.
            gs13 = (1.0 / input_small["w13"].max()).to(torch.float32).reshape(())
            gs2 = (1.0 / input_small["w2"].max()).to(torch.float32).reshape(())

            device = self._decode_device()
            stacks = {}
            for key, axis in self._axes.items():
                stacks[key] = axis.finish()
            self._axes = None
            # Per-expert epilogues: the joined weight global over the A-side
            # global, frozen once so a forward reads no host scalar.
            shared13 = torch.tensor(
                [self._shared_w13[index] for index in range(experts)],
                dtype=torch.float32, device=device)
            shared2 = torch.tensor(
                [self._shared_w2[index] for index in range(experts)],
                dtype=torch.float32, device=device)
            layer.tessera_a4_gate_stack = stacks[("w13", "gate_proj")]
            layer.tessera_a4_up_stack = stacks[("w13", "up_proj")]
            layer.tessera_a4_down_stack = stacks[("w2", "down_proj")]
            layer.tessera_a4_gs13 = gs13
            layer.tessera_a4_gs2 = gs2
            layer.tessera_a4_gate_epilogues = shared13 / gs13
            layer.tessera_a4_up_epilogues = shared13 / gs13
            layer.tessera_a4_down_epilogues = shared2 / gs2
            del layer.w13_wire, layer.w2_wire
            del layer.w13_input_global_scale, layer.w2_input_global_scale
            layer.tessera_w13_wire_len = None
            layer.tessera_w2_wire_len = None
            self._intake = None
            self._tiles = None
            self._input_global = None
            self._w13_len = self._w2_len = None
            self._shared_w13 = self._shared_w2 = None
            layer.tessera_decoder = "native_span2_grouped"
            layer.tessera_backend = str(getattr(self.nvfp4_backend, "name",
                                                self.nvfp4_backend))

        def get_fused_moe_quant_config(self, layer):
            return make_nvfp4_moe_quant_config(
                backend=self.nvfp4_backend,
                w13_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
                w13_scale_2=layer.w13_weight_scale_2, w2_scale_2=layer.w2_weight_scale_2,
                a13_scale=layer.w13_input_scale, a2_scale=layer.w2_input_scale,
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                swiglu_alpha=getattr(layer, "swiglu_alpha", None),
                swiglu_beta=getattr(layer, "swiglu_beta", None),
                layer=layer, use_a16=False)

        # -- forward ----------------------------------------------------
        def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                  shared_experts_input):
            """Two native stages over one device dispatch, weights applied last.

            gate/up are separate grouped calls (per-role tables and globals
            stay per role; no artificial trellis merge), the activation is the
            layer's own, the down stage consumes the per-route rows, and the
            router weights are applied only in the final combine -- the
            routed-MoE contract.  ``apply_router_weight_on_input`` and an
            expert map are refused rather than approximated.
            """
            from .native_a4 import a4_grouped_apply

            assert not self.is_monolithic
            if layer.expert_map is not None:
                raise ValueError(
                    f"tessera target {prefix!r}: expert parallelism carries an expert "
                    "map and the native A4 route serves global expert ids only")
            if getattr(layer, "apply_router_weight_on_input", False):
                raise ValueError(
                    f"tessera target {prefix!r}: apply_router_weight_on_input is not "
                    "part of the native A4 contract (weights apply after the down "
                    "projection); refusing rather than approximating")
            if x.ndim != 2 or x.shape[1] != hidden:
                raise ValueError(
                    f"tessera target {prefix!r}: the routed activation must be "
                    f"[tokens, {hidden}], got {tuple(x.shape)}")
            top_k = int(self.moe.experts_per_token)
            if tuple(topk_ids.shape) != (x.shape[0], top_k):
                raise ValueError(
                    f"tessera target {prefix!r}: routing must be [tokens, {top_k}], "
                    f"got {tuple(topk_ids.shape)}")
            if x.shape[0] == 0:
                return x.new_empty((0, hidden))

            device = x.device
            # No host read of the routing tensor: ids are the router's device
            # output and its own contract (0 <= id < num_experts) is what the
            # modular kernels consume too.  An in-kernel gather of an
            # out-of-range id is a fault, not a served answer.
            flat_ids = topk_ids.to(torch.int64).reshape(-1)
            flat_tokens = torch.arange(x.shape[0], device=device,
                                       dtype=torch.int64).repeat_interleave(top_k)
            flat_weights = topk_weights.reshape(-1).to(torch.float32)
            order = torch.argsort(flat_ids, stable=True)
            counts = torch.zeros(experts, dtype=torch.int32, device=device)
            counts.scatter_add_(0, flat_ids[order],
                                torch.ones_like(flat_ids, dtype=torch.int32))
            offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
            offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
            route_tokens = flat_tokens[order].to(torch.int32)
            route_weights = flat_weights[order]
            routes = int(flat_ids.numel())

            gate = a4_grouped_apply(x, layer.tessera_a4_gate_stack, layer.tessera_a4_gs13,
                                    expert_offsets=offsets, route_ids=route_tokens,
                                    num_routes=routes,
                                    epilogues=layer.tessera_a4_gate_epilogues)
            up = a4_grouped_apply(x, layer.tessera_a4_up_stack, layer.tessera_a4_gs13,
                                  expert_offsets=offsets, route_ids=route_tokens,
                                  num_routes=routes,
                                  epilogues=layer.tessera_a4_up_epilogues)
            # The layer's activation is a ``MoEActivation`` ENUM, not a
            # callable: vLLM's own ``apply_moe_activation`` is the executor
            # (same op the modular kernels dispatch), driven by the layer's
            # own clamp/alpha/beta facts, and an unsupported activation is
            # refused by name rather than approximated.
            from vllm.model_executor.layers.fused_moe.activation import (
                ApplyMoEActivationConfig, apply_moe_activation,
                apply_moe_activation_supported)

            activation = layer.activation
            if not apply_moe_activation_supported(activation):
                raise ValueError(
                    f"tessera target {prefix!r}: the native A4 route does not "
                    f"serve the layer activation {activation!r}")
            activation_config = ApplyMoEActivationConfig(
                clamp_limit=getattr(layer, "swiglu_limit", None),
                alpha=float(getattr(layer, "swiglu_alpha", None) or 1.0),
                beta=float(getattr(layer, "swiglu_beta", None) or 0.0),
                activation_situ_beta=getattr(self.moe, "activation_situ_beta", None),
                activation_situ_linear_beta=getattr(
                    self.moe, "activation_situ_linear_beta", None))
            gate_up = torch.cat([gate, up], dim=-1)
            activated = torch.empty((gate_up.shape[0], gate_up.shape[1] // 2),
                                    dtype=gate_up.dtype, device=gate_up.device)
            apply_moe_activation(activation, activated, gate_up,
                                 activation_config=activation_config)
            identity = torch.arange(routes, dtype=torch.int32, device=device)
            down = a4_grouped_apply(activated, layer.tessera_a4_down_stack,
                                    layer.tessera_a4_gs2, expert_offsets=offsets,
                                    route_ids=identity, num_routes=routes,
                                    epilogues=layer.tessera_a4_down_epilogues)
            out = torch.zeros((x.shape[0], hidden), dtype=torch.float32, device=device)
            out.index_add_(0, route_tokens.to(torch.int64),
                           down * route_weights[:, None])
            out = out.to(x.dtype)
            # Shared experts are the RUNNER's: ``FusedMoERunner`` calls
            # ``SharedExperts`` once (``NO_OVERLAP``, or the multi-stream path
            # with its own wait) before this apply, and combines the stored
            # output with the routed result after it.  A quant method that
            # recomputed or re-added them here would count them twice; this
            # method is not a modular kernel, so the MK-internal order is
            # never its to run.  ``shared_experts``/``shared_experts_input``
            # are therefore intentionally unconsumed.
            self._record(layer, x)
            return out

        def apply_monolithic(self, layer, x, router_logits, input_ids=None):
            assert self.is_monolithic
            assert self.moe_kernel is not None
            out = self.moe_kernel.apply_monolithic(
                x, layer.w13_weight, layer.w2_weight, router_logits,
                activation=layer.activation, global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                num_expert_group=layer.num_expert_group, topk_group=layer.topk_group,
                e_score_correction_bias=layer.e_score_correction_bias,
                routed_scaling_factor=layer.routed_scaling_factor)
            self._record(layer, x)
            return out

        def _record(self, layer, x) -> None:
            try:
                x2 = x.reshape(-1, x.shape[-1])
                emit_route(
                    layer, kind="moe", policy=f"{family}:{layer.tessera_mode}",
                    symbol=f"{GEMM_SYMBOL}:{layer.tessera_backend}", tile_m=0,
                    shape=route_shape(x2, layer.tessera_rows, layer.tessera_columns),
                    contract=layer.tessera_activation_contract, state="served", reason=None,
                    decoder=layer.tessera_decoder)
            except Exception:  # noqa: BLE001 -- telemetry never breaks a request
                pass

    return TesseraNvFp4MoEMethod(layer.moe_config)
