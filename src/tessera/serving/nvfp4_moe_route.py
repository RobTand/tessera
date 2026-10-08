"""Native fused routed W4A4 over paired WINDOW L14 and LUT16 expert bytes.

Each projection keeps its own code table, group-16 scale nibbles, UE4M3
table and weight global. Compact preparation fills one WindowUnitAxis per
group, then FusedRoutedE2M1MoE owns the native eager and graph forwards.
No whole weight, TCQ reader, stock backend or compact FP8/BF16 substitute
is part of this serving route.

Static input globals remain checkpoint facts, reduced exactly as before:
invert each calibrated capacity-over-amax, take the maximum reciprocal over
all experts and projections of a GEMM input, then invert back. Weight globals
are never joined across gate/up or experts. Rank-local row cuts carry WINDOW
start states; column cuts preserve paired group-16 placement. Shared experts
and tensor-parallel reduction stay with vLLM's runner.

The declared reader/launch support is not a serving attestation. Historical
TCQ receipts cannot qualify new WINDOW bytes.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch

from ..errors import GrammarError
from ..moe_layout import W13_PROJECTIONS, validate_moe_wire_lengths
from .lane import MODE_RESIDENT, MODES
from .residency import layer_resident_tensors
from .moe_route import SHARD_TO_GROUP, _bind_module_prefix, _packed_group_shard_plan
from .scheme import (GROUP_SIZE, MOE_GROUPS, ROUTES, ROUTED_FUSED_WINDOW_E2M1_SYMBOL,
                     STRUCTURE_ROUTED_MOE, TESSERA_NVFP4, expert_role_declarations,
                     launch_pairs, moe_census_symbol_base as census_symbol_base,
                     route_launches, validate_tessera_moe_scheme)
from .telemetry import DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1, emit_route, route_shape

__all__ = [
    "ACTIVATION_CONTRACT",
    "GEMM_SYMBOL",
    "INPUT_GLOBAL_SCALE_SUFFIX",
    "PAYLOAD_FAMILY",
    "census_expected",
    "census_symbol_base",
    "build_tessera_nvfp4_moe_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_NVFP4]["activation_contract"]

# The owner declares its actual bundles, descriptors, ratios and counters.
RESIDENT_ATTRIBUTES = ("tessera_routed_fused",)
GEMM_SYMBOL = ROUTED_FUSED_WINDOW_E2M1_SYMBOL
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
    """Native launch pairs this routed owner can report, not qualification."""
    del compiled  # one launch has nothing to combine
    launches = route_launches(TESSERA_NVFP4, structure=STRUCTURE_ROUTED_MOE,
                              mode=MODE_RESIDENT)
    regimes = {regime for launch in launches for regime in launch["regimes"]}
    # Launch support and lane_eligibility receipts are separate declarations.
    from .scheme import experimental_launch_pairs

    pairs = {regime: launch_pairs(TESSERA_NVFP4, structure=STRUCTURE_ROUTED_MOE,
                                  regime=regime, mode=MODE_RESIDENT)
             | experimental_launch_pairs(TESSERA_NVFP4, structure=STRUCTURE_ROUTED_MOE,
                                         regime=regime, mode=MODE_RESIDENT)
             for regime in regimes}
    from .census import platform_expectation

    return platform_expectation(PAYLOAD_FAMILY, platform, pairs)


class _ExpertIntake:
    """Validate full expert containers before preparing the rank-local units."""

    def __init__(self, declared, target, tp_rank, tp_size):
        self.target = target
        self.plans = {g: _packed_group_shard_plan(declared, g, target, tp_rank, tp_size)
                      for g in MOE_GROUPS}
        self.roles = {g: expert_role_declarations(declared["groups"][g]) for g in MOE_GROUPS}
        self._scratch: dict = {}
        self._memo: dict = {}

    def take(self, group, index, expert, blob: bytes, device, axes=None):
        from ..compact_prep import prepare_window_lut_compact
        from .scheme import parse_compact_tessera_expert_blob

        declared_role = self.roles[group][index]
        validated = parse_compact_tessera_expert_blob(
            blob, declared_role, f"{self.target} {group} expert {expert}",
            device=device, memo=self._memo)
        if len(validated) != 1:
            raise GrammarError(
                f"{self.target} {group} expert {expert}: an expert projection container "
                f"holds one role, this one frames {len(validated)}")
        name, member = validated[0]
        plan = self.plans[group]
        shard = plan.role(name)
        rows = (shard.lo, shard.hi) if plan.axis == "row" else None
        cols = (shard.lo, shard.hi) if plan.axis == "column" else None
        unit = prepare_window_lut_compact(member, rows=rows, cols=cols,
                                          device=device, scratch=self._scratch)
        if axes is None:
            return name, unit
        # Scratch-backed temporary words are copied into their one resident
        # axis slot now, before the next compact preparation can reuse them.
        axes[group].put(name, expert, unit)
        return None

    def release(self):
        self._scratch.clear()
        self._memo.clear()


def _window_bundles(axes, experts, expert_classes):
    """Wrap finished e2m1 SoA tensors by reference, never choose an adapter.

    ``expert_classes`` is the scheme's declared class table. An E2M1 stack
    has one class: its validator refuses experts with different run tables.
    """
    from ..native_window_moe import PackedWindowMoeBundles
    from ..window_gemm_grouped import prepare_grouped_window_gemm_from_soa

    soa = {group: axis.finish() for group, axis in axes.items()}

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
            experts=experts, window_bits=slot["window_bits"], family="e2m1",
            scale_plane_all=slot["scale_plane"], scale_lut_all=slot["scale_lut"],
            global_all=slot["global_scale"],
            word_layout=str(slot.get("word_layout", "legacy")))

    return PackedWindowMoeBundles(gate=bundle("w13", "gate_proj"),
                                  up=bundle("w13", "up_proj"),
                                  down=bundle("w2", "down_proj"), family="e2m1",
                                  expert_classes=expert_classes)


def _static_input_global(scales, device):
    """The checkpoint reduction, retaining its reciprocal rounding sequence."""
    return (1.0 / (1.0 / scales).max()).to(device=device, dtype=torch.float32).reshape(())


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
    # The route trace names a module by ``layer.prefix`` and vLLM's fused MoE
    # stores none; ``moe_route._bind_module_prefix`` says why and what it keeps.
    _bind_module_prefix(layer, prefix)
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
    from vllm.model_executor.utils import set_weight_attrs

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
            # NO stock modular kernel and NO stock experts class: the native
            # grouped calls are this route's whole computation.  The runtime's
            # ``is_monolithic`` delegates to ``experts_cls`` whenever one is
            # present and ``moe_kernel`` is None
            # (fused_moe_method_base.py:137-143), so a selected stock backend
            # with a monolithic class would send this method to the obsolete
            # ``apply_monolithic`` path and assert a kernel that never exists
            # here.  Both are pinned to None instead of inherited from a
            # selection whose kernel this route never runs.
            self.moe_kernel = None
            self.moe_quant_config = None
            self._intake = None
            self._tiles = None
            self._w13_len = self._w2_len = None
            self._input_global = None

        @property
        def supports_eplb(self) -> bool:
            return False

        # -- the native modular protocol, stated explicitly -----------------
        # (The runtime dispatches on exactly these; see
        # ``FusedMoERunner._apply_quant_method``: ``is_monolithic`` False
        # means ``forward_modular``, which is the only path this route has.)
        @property
        def is_monolithic(self) -> bool:
            """The native lane is modular; no stock class may answer this.

            Overridden so the answer can never come from an
            ``experts_cls`` this route does not own: the base property
            returns ``self.experts_cls.is_monolithic()`` when a class is
            present, and the native grouped calls have no stock kernel to be
            monolithic about.
            """
            return False

        @property
        def topk_indices_dtype(self) -> "torch.dtype | None":
            """The router's own ids are consumed as given (int32/int64)."""
            return None

        @property
        def mk_can_overlap_shared_experts(self) -> bool:
            """The runner owns shared experts; this method runs none."""
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
            from .scheme import e2m1_shape_reason
            for projection, rows_local, cols_local in (("gate_proj", local, hidden),
                                                        ("down_proj", hidden, local)):
                reason = e2m1_shape_reason(rows_local, cols_local,
                                           structure=STRUCTURE_ROUTED_MOE, projection=projection)
                if reason is not None:
                    raise ValueError(f"tessera target {prefix!r} rank {self._tp_rank}: {reason}")
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
            from ..native_window_moe import WindowUnitAxis

            self._axes = {
                group_name: WindowUnitAxis(
                    experts, [role["roles"][0][0] for role in self._intake.roles[group_name]],
                    family="e2m1")
                for group_name in MOE_GROUPS
            }
            layer.tessera_mode = mode
            layer.tessera_family = family
            layer.tessera_structure = declared["structure"]
            layer.tessera_activation_contract = ACTIVATION_CONTRACT
            layer.tessera_rows = n_rows
            layer.tessera_columns = hidden

        def intake_axes(self) -> dict:
            """The actual e2m1 axes while loading; empty after native ownership."""
            return dict(getattr(self, "_axes", None) or {})

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
            """Validate one full container, cut it to this rank, prepare it into
            the expert's slot of the compact axis bundle, and drop the planes
            (no stock tile is built on this route)."""
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
            self._intake.take(
                group, index, expert_id, blob.detach().cpu().contiguous().numpy().tobytes(),
                device, axes=self._axes)
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
            # Reducing reciprocal scales preserves the checkpoint semantics.
            # The selected FlashInfer MoE backends aggregate the routed
            # activation scale per layer/projection rather than per expert
            # (``flashinfer_fp4_moe.py``'s
            # ``is_global_sf_supported_for_nvfp4_backend`` computes it through
            # ``amax_for_moe_activation_quant`` and repeats it across the
            # experts; the oracle collapses a disagreeing vector the same way),
            # so one quantizer scalar per GEMM is the served contract.  The
            # native path reproduces that reduction instead of quantising per
            # expert.
            device = self._decode_device()
            gs13 = _static_input_global(self._input_global["w13"].data, device)
            gs2 = _static_input_global(self._input_global["w2"].data, device)
            from ..routed_fused_e2m1 import FusedRoutedE2M1MoE
            from ..native_window_moe import checked_swiglu_limit

            activation = getattr(layer.activation, "value", layer.activation)
            if activation != "silu":
                raise ValueError(
                    f"tessera target {prefix!r}: native E2M1 serves silu, got {activation!r}")
            for fact, default in (("swiglu_alpha", 1.0), ("swiglu_beta", 0.0)):
                value = getattr(layer, fact, None)
                if value is not None and float(value) != default:
                    raise ValueError(
                        f"tessera target {prefix!r}: {fact}={value!r} has no exact native implementation")
            for fact in ("activation_situ_beta", "activation_situ_linear_beta"):
                if getattr(self.moe, fact, None) is not None:
                    raise ValueError(
                        f"tessera target {prefix!r}: {fact} has no exact native implementation")
            checked_swiglu_limit(getattr(layer, "swiglu_limit", None), where=f"{prefix}: ")
            if getattr(layer, "apply_router_weight_on_input", False) and int(self.moe.experts_per_token) != 1:
                raise ValueError(
                    f"tessera target {prefix!r}: apply_router_weight_on_input requires topk=1")
            bundles = _window_bundles(self._axes, experts, declared["expert_classes"])
            layer.tessera_routed_fused = FusedRoutedE2M1MoE.from_bundles(
                bundles.gate, bundles.up, bundles.down, gs13=gs13, gs2=gs2)
            self._axes = None
            self._intake.release()
            del layer.w13_wire, layer.w2_wire
            del layer.w13_input_global_scale, layer.w2_input_global_scale
            layer.tessera_w13_wire_len = None
            layer.tessera_w2_wire_len = None
            self._intake = None
            self._tiles = None
            self._input_global = None
            self._w13_len = self._w2_len = None
            # The runtime asks a method whose config is None for one before it
            # serves; build it here, once, from the model's own facts.
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            layer.tessera_decoder = DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1
            layer.tessera_backend = GEMM_SYMBOL


        def get_fused_moe_quant_config(self, layer):
            """The native lane's quant config: alphas only, no stock tensors.

            ``FusedMoEMethodBase`` declares this abstract and the runtime asks
            every method for it (``RoutedExperts._ensure_moe_quant_config_init``);
            the native grouped kernels take no quantised operands from it, so
            it carries the model's swiglu facts and nothing else.  It must not
            read the stock weight tensors (``w13_weight_scale`` and friends):
            this route deliberately never fills them, and a config assembled
            from empty anchors would be a lie about what serves.
            """
            from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

            return FusedMoEQuantConfig.make(
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None))

        # -- residency declaration (#580) -------------------------------
        def resident_tensors(self, layer):
            """The prepared tensors this route holds for ``layer`` outside
            registered state, by reference (``serving.residency``)."""
            return layer_resident_tensors(layer, RESIDENT_ATTRIBUTES)

        # -- forward ----------------------------------------------------
        def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                  shared_experts_input):
            """Native routed output; the runner alone evaluates shared experts."""
            assert not self.is_monolithic
            if layer.expert_map is not None:
                raise ValueError(
                    f"tessera target {prefix!r}: expert parallelism carries an expert "
                    "map and the native E2M1 route serves global expert ids only")
            if x.ndim != 2 or x.shape[1] != hidden:
                raise ValueError(
                    f"tessera target {prefix!r}: the routed activation must be "
                    f"[tokens, {hidden}], got {tuple(x.shape)}")
            top_k = int(self.moe.experts_per_token)
            if tuple(topk_ids.shape) != (x.shape[0], top_k):
                raise ValueError(
                    f"tessera target {prefix!r}: routing must be [tokens, {top_k}], "
                    f"got {tuple(topk_ids.shape)}")
            out = layer.tessera_routed_fused(
                x, topk_ids, topk_weights,
                apply_router_weight_on_input=bool(getattr(layer, "apply_router_weight_on_input", False)),
                swiglu_limit=getattr(layer, "swiglu_limit", None))
            self._record(layer, x)
            return out


        def _record(self, layer, x) -> None:
            try:
                x2 = x.reshape(-1, x.shape[-1])
                symbol = layer.tessera_backend
                emit_route(
                    layer, kind="moe", policy=f"{family}:{layer.tessera_mode}",
                    symbol=symbol, tile_m=0,
                    shape=route_shape(x2, layer.tessera_rows, layer.tessera_columns),
                    contract=layer.tessera_activation_contract, state="served", reason=None,
                    decoder=layer.tessera_decoder, kernel_schedule=symbol)
            except Exception:  # noqa: BLE001 -- telemetry never breaks a request
                pass

    return TesseraNvFp4MoEMethod(layer.moe_config)
