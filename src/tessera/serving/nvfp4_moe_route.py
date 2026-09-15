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
        from .sharding import shard_parsed_roles

        parsed = parse_tessera_expert_blob(
            blob, self.roles[group][index], f"{self.target} {group} expert {expert}",
            device=device)
        local = shard_parsed_roles(parsed, self.plans[group])
        if group != "w13":
            return local
        halves = self.pending.setdefault(expert, [None] * W13_PROJECTIONS)
        halves[index] = local
        if any(half is None for half in halves):
            return None
        del self.pending[expert]
        return [role for half in halves for role in half]


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

            # The stock modelopt parameter set, allocated at create time with
            # the shapes ``ModelOptNvFp4FusedMoE.create_weights`` allocates,
            # so what the kernel sees is the runtime's own parameter set and
            # ``replace_parameter`` has something to replace.
            shapes = {
                "w13_weight": ((experts, n_rows, hidden // 2), torch.uint8),
                "w2_weight": ((experts, hidden, local // 2), torch.uint8),
                "w13_weight_scale": ((experts, n_rows, hidden // GROUP_SIZE), torch.float8_e4m3fn),
                "w2_weight_scale": ((experts, hidden, local // GROUP_SIZE), torch.float8_e4m3fn),
                "w13_weight_scale_2": ((experts, W13_PROJECTIONS), torch.float32),
                "w2_weight_scale_2": ((experts,), torch.float32),
                "w13_input_scale": ((experts, W13_PROJECTIONS), torch.float32),
                "w2_input_scale": ((experts,), torch.float32),
            }
            self._tiles = {}
            for name in _STOCK_TILE_NAMES:
                shape, dtype = shapes[name]
                param = torch.nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)
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
                packed, scale, shared = decode_expert_tile(ready, device)
                weight = self._tiles[f"{group}_weight"]
                block_scale = self._tiles[f"{group}_weight_scale"]
                if (tuple(packed.shape) != tuple(weight.shape[1:])
                        or tuple(scale.shape) != tuple(block_scale.shape[1:])):
                    raise ValueError(
                        f"tessera target {prefix!r} expert {expert_id} {group}: the rank-local "
                        f"decode is {tuple(packed.shape)} nibbles / {tuple(scale.shape)} "
                        f"scales, the tile is {tuple(weight.shape[1:])} / "
                        f"{tuple(block_scale.shape[1:])}")
                weight.data[expert_id].copy_(packed)
                block_scale.data[expert_id].copy_(scale)
                if group == "w13":
                    self._tiles["w13_weight_scale_2"].data[expert_id, :] = shared
                else:
                    self._tiles["w2_weight_scale_2"].data[expert_id] = shared
                del packed, scale, ready
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
            self._tiles["w13_input_scale"].data.copy_(1.0 / self._input_global["w13"].data)
            self._tiles["w2_input_scale"].data.copy_(1.0 / self._input_global["w2"].data)
            del layer.w13_wire, layer.w2_wire
            del layer.w13_input_global_scale, layer.w2_input_global_scale
            layer.tessera_w13_wire_len = None
            layer.tessera_w2_wire_len = None
            self._intake = None
            self._tiles = None
            self._input_global = None
            self._w13_len = self._w2_len = None

            # From here this IS ``ModelOptNvFp4FusedMoE.process_weights_after_loading``.
            if not torch.equal(layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]):
                raise ValueError(
                    f"tessera target {prefix!r}: w13 gate/up globals differ on some expert; "
                    "the loader joins them per expert, so this is a decode defect")
            w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()
            (w13, w13_scale, w13_scale_2, a13_scale,
             w2, w2_scale, w2_scale_2, a2_scale) = convert_to_nvfp4_moe_kernel_format(
                nvfp4_backend=self.nvfp4_backend, layer=layer,
                w13=layer.w13_weight, w13_scale=layer.w13_weight_scale,
                w13_scale_2=w13_weight_scale_2, a13_scale=layer.w13_input_scale,
                w2=layer.w2_weight, w2_scale=layer.w2_weight_scale,
                w2_scale_2=layer.w2_weight_scale_2, a2_scale=layer.w2_input_scale,
                is_act_and_mul=self.moe.is_act_and_mul, use_a16=False)
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
            replace_parameter(layer, "w13_input_scale", a13_scale)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w2_weight_scale", w2_scale)
            replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
            replace_parameter(layer, "w2_input_scale", a2_scale)

            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            assert self.experts_cls is not None
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config, moe_config=self.moe,
                experts_cls=self.experts_cls, backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables())
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)
            layer.tessera_decoder = DECODER_TORCH_STOCK
            # The oracle's enum names its members by their own name (value ==
            # name on the pinned image); the NAME is the stable identifier the
            # route symbol carries.
            layer.tessera_backend = str(getattr(self.nvfp4_backend, "name", self.nvfp4_backend))

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
            assert not self.is_monolithic
            assert self.moe_kernel is not None
            out = self.moe_kernel.apply(
                x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids,
                activation=layer.activation, global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts, shared_experts_input=shared_experts_input)
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
