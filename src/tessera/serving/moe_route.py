"""The Tessera routed-MoE expert route: per-expert E4M3 wires as the stock FP8 stack.

WHAT IT SERVES. One ``tessera.fused`` container per expert per projection,
assembled into ``w13`` (gate then up, the row order
``RoutedExperts._load_w13`` narrows to) and ``w2`` (down), then decoded at
load into exactly the parameters vLLM's own fused-MoE kernels
read for a per-channel FP8 checkpoint: ``w13_weight [E, 2N, K]`` and
``w2_weight [E, K, N]`` in ``float8_e4m3fn``, with ``w13_weight_scale
[E, 2N, 1]`` and ``w2_weight_scale [E, K, 1]`` in fp32.  From
``process_weights_after_loading`` onward this route IS
``CompressedTensorsW8A8Fp8MoEMethod`` at ``strategy: channel``: the same
``convert_to_fp8_moe_kernel_format``, the same ``make_fp8_moe_quant_config``
(``per_out_ch_quant`` and ``per_act_token_quant`` both true), the same
``make_fp8_moe_kernel``, and an ``apply`` that hands the runtime's modular
kernel the runtime's own tensors.  Nothing here writes a kernel.

WHY PER-CHANNEL.  The CHANNEL scale plane is the wire's column structure and
it is not decoration: deleting it costs 0.77-0.84x, and folding 2N row scales
into one per expert is deleting it.  That the runtime's fused-MoE kernels
ACCEPT a per-channel weight scale on this hardware is not asserted here -- it
is read off the runtime's own ``is_supported_config`` predicate
(``experiments/moe_decode_target_probe.py`` /
``experiments/results/moe_decode_target_probe.json``: MARLIN, HUMMING, TRITON
and BATCHED_TRITON accept ``(kFp8StaticChannelSym, kFp8DynamicTokenSym)`` on
sm_121), and the backend is selected by the runtime's own
``select_fp8_moe_backend`` rather than pinned here.

THE WIRE PARAMETER, AND WHY IT HAS ITS OWN LOADER.  ``RoutedExperts``'
expert-parameter mapping is suffix-agnostic -- ``experts.{e}.gate_proj.wire``
routes to the ``w13_wire`` parameter with ``shard_id="w1"`` -- but its own
``weight_loader`` dispatches on the substrings "weight"/"scale" and returns
``False`` for anything else, writing nothing and saying nothing
(``docs/measurements/tessera-moe-wire-loader-2026-09-03.md``).  What
``load_weights`` actually calls is ``param.weight_loader``, so a wire
parameter carrying its own loader is the mechanism, and this route registers
one.

THE ORDINARY ROWS ARE PADDED AND THE LENGTHS RIDE BESIDE THEM.  A checkpoint stores one
tensor per expert projection at that blob's exact length; the PARAMETER is
rectangular, so ``create_weights`` allocates the group's declared
``wire_stride`` and the loader records each blob's true length.  What comes
back out is ``tessera.moe_layout.unpack_moe_wires``, whose refusals are the
integrity gate: a length past its row, a length tensor that disagrees with the
expert count, and -- the one that catches a mis-declared sidecar -- a stride
that is not the maximum its lengths imply.

WHAT THE PRODUCTION ROUTE REFUSES. Expert
parallelism and tensor parallelism inside an expert (the stride invariant
needs every expert's blob, and no expert slicer has been run); a residency
mode other than ``resident`` (a per-forward expert decode is a different
kernel story with no measurement); a family with no expert route
(``scheme.MOE_BUILDERS`` says which, and why the other two are absent); an
expert count, hidden size or intermediate size that disagrees with the
sidecar; and a non-gated MoE, whose ``w13`` is one shard rather than the pair
this route's groups describe.

The explicit ``ResearchSelectedMoeConfig`` separately permits eager
TP1 or TP2 selected decode. TP2 constructs zero-byte loader parameters and
validates each whole original wire during its load callback, then invokes
the existing role slicer and retains only local packed roles; it preserves global expert IDs and
leaves output reduction to stock vLLM. It is not a qualified runtime cell.

WHAT IS ATTESTED. The packaged contract publishes exactly two ``routed_moe`` cells:
E4M3/q1024, resident/eager on sm_121, for decode and batch on the exact EUGR
image named by each cell. The complete LFM artifact's census and source-bound
prefill KL comparison are recorded in
``docs/measurements/tessera-lfm-campaign-2026-09-04.md``. Other rungs/images,
compiled/streamed MoE and multi-rank execution remain unattested.

The earlier load-and-execute measurement
(``docs/measurements/tessera-moe-route-load-2026-09-04.md``) was taken twice --
once on the pin, once on the build that registers ``Glm5Next`` -- and every
recorded field, backend selection and error number is identical, so the route
does not depend on which of the two loads it. Later served GLM census
receipts cover 16-expert stacks, and the exact EUGR LFM construction receipt
(``docs/measurements/tessera-lfm-construction-2026-09-04.md``) covers model-level
wire delegation into a 32-expert stack without a forward. These receipts do
not establish full-model LFM served quality or a compiled MoE forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ..moe_execution import ResearchSelectedMoeConfig
from ..moe_layout import (W13_PROJECTIONS, MoePacked, unpack_moe_wires,
                          validate_moe_wire_lengths)
from .lane import MODE_RESIDENT, MODES
from .scheme import (MOE_GEMM_SYMBOL, MOE_GROUP_SHARDS, MOE_GROUPS, ROUTES,
                     STRUCTURE_ROUTED_MOE, TESSERA_FP8, launch_pairs, route_launches,
                     moe_census_symbol_base as census_symbol_base,
                     expert_role_declarations, parse_tessera_expert_blob,
                     validate_tessera_moe_scheme)
from .telemetry import DECODER_TORCH_STOCK, emit_route, route_shape

__all__ = [
    "ACTIVATION_CONTRACT",
    "GEMM_SYMBOL",
    "SHARD_TO_GROUP",
    "PreparedTesseraMoeExperts",
    "PreparedTesseraPackedMoeExperts",
    "ResearchSelectedMoeConfig",
    "census_expected",
    "census_symbol_base",
    "prepare_tessera_moe_experts",
    "prepare_tessera_packed_moe_experts",
    "build_tessera_moe_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_FP8]["activation_contract"]
GEMM_SYMBOL = MOE_GEMM_SYMBOL




def census_expected(*, compiled: bool = False, platform=None) -> dict:
    """The ``(symbol, decoder)`` pairs an expert stack may report, by regime.

    Owned here for the same reason ``fp8_gemv.census_expected`` is owned there:
    the dispatch is in this module, so a new path updates the expectation where
    the path was added rather than in a second spelling inside the census tool.
    Two things about this route are NOT the dense routes' shape.

    ONE LAUNCH, BOTH REGIMES.  There is no GEMV lane and no kernel decode here.
    ``process_weights_after_loading`` materialises the stack once and every
    forward, at any M, hands the runtime's modular fused-MoE kernel the tile
    that materialise produced -- so ``decode`` and ``batch`` admit the same
    single pair, where the window routes' two regimes admit different ones.
    ``compiled`` therefore changes nothing: the combined ``a+b`` symbol those
    routes stamp under a traced forward exists because their dispatch BRANCHES
    inside the graph, and one launch has nothing to combine.

    THE SYMBOL CARRIES A SUFFIX THIS ROUTE DOES NOT CHOOSE.  ``_record`` stamps
    ``vllm.fused_moe.modular_kernel:<backend>`` because which backend ran is a
    fact about the serve a receipt must not lose -- but the backend is
    ``select_fp8_moe_backend``'s answer, the RUNTIME's predicate over the
    kernels it finds on this box, not a promise this route makes.  So the pair
    below carries the entry point alone and a census compares
    :func:`census_symbol_base`, keeping the exact string in its histogram.
    Enumerating the backends we would accept would be a claim about vLLM's
    kernel roster written in our own prose, which the runtime-attestation rule forbids;
    pinning one would refuse a box whose runtime picked another.

    DERIVATION IS NOT ATTESTATION. The shared ``scheme.ROUTE_LAUNCHES`` table
    separates this expert structure from the dense FP8 launch set. Reading
    that table keeps the census and contract derivations together. It does
    not itself publish a served cell. The packaged contract's measured E4M3/q1024
    resident/eager cells name their exact EUGR image, toolchain and sm_121 scope;
    returning the same expected launch for compiled execution does not attest it.
    """
    del compiled  # documented above: one launch has nothing to combine
    launches = route_launches(TESSERA_FP8, structure=STRUCTURE_ROUTED_MOE,
                              mode=MODE_RESIDENT)
    regimes = {regime for launch in launches for regime in launch["regimes"]}
    pairs = {regime: launch_pairs(TESSERA_FP8, structure=STRUCTURE_ROUTED_MOE,
                                  regime=regime, mode=MODE_RESIDENT)
             for regime in regimes}
    # PER ``(platform, family)`` (#457).  The expert stack's family is the
    # dense FP8 route's -- same wire, same activation contract -- so a
    # platform that executes no E4M3 route executes none for the experts
    # either, and ``build_tessera_moe_method`` refuses such a stack at
    # construction.  ``platform=None`` is unchanged.
    from .census import platform_expectation

    return platform_expectation("TESSERA_E4M3_K1", platform, pairs)


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


def _parsed_experts(blobs, declared_group, target, device):
    """One parsing/role-validation seam for resident and research owners."""
    role_declarations = expert_role_declarations(declared_group)
    for expert, expert_blobs in enumerate(blobs):
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

    Layout compatibility is checked by ``PreparedWindow.stack``. Per-expert
    bodies/scales/alphabets may differ; heterogeneous gather layouts refuse.
    TP2 validates each original full container before the existing dense shard
    planner/slicer derives rank-local roles. Global expert IDs are unchanged.
    Only a transient per-role FP8 reference is materialized during preparation.
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
        modules = [prepare_tessera_fp8_module(shard_parsed_roles(roles, plan), device=device)
                   for roles in _parsed_experts(blobs[group], declaration,
                                                f"{target} {group}", device)]
        prepared[group] = PreparedTesseraFp8Module.stack(modules)
        del modules
    return PreparedTesseraPackedMoeExperts(prepared['w13'], prepared['w2'])


class _RankLocalPackedIntake:
    """TP2 loader ownership: one validated original becomes one local packed role."""

    def __init__(self, declared, target, device, tp_rank, tp_size):
        self.declared, self.target, self.device = declared, target, device
        self._has_loaded = False
        self.plans = {g: _packed_group_shard_plan(declared, g, target, tp_rank, tp_size)
                      for g in MOE_GROUPS}
        self.roles = {g: expert_role_declarations(declared['groups'][g]) for g in MOE_GROUPS}
        self.prepared = {g: [[None] * len(self.roles[g]) for _ in range(declared['experts'])]
                         for g in MOE_GROUPS}

    def load(self, group, index, expert, wire, *, device):
        from .fp8_route import prepare_tessera_fp8_module
        from .sharding import shard_parsed_roles

        device = torch.device(device)
        if self._has_loaded and device != self.device:
            raise ValueError(f'{self.target}: packed intake cannot change device after loading starts')
        self.device = device
        # Parse the FULL incoming container before slicing; the common parser
        # validates its digest, geometry, role and recipe against the sidecar.
        blob = wire.detach().cpu().contiguous().numpy().tobytes()
        parsed = parse_tessera_expert_blob(blob, self.roles[group][index],
            f'{self.target} {group} expert {expert}', device=self.device)
        local = shard_parsed_roles(parsed, self.plans[group])
        prepared = prepare_tessera_fp8_module(local, device=self.device)
        self.prepared[group][expert][index] = prepared
        self._has_loaded = True

    def finish(self, w13_lengths, w2_lengths):
        from .fp8_route import PreparedTesseraFp8Module

        validate_moe_wire_lengths(w13_lengths, w2_lengths,
            experts=self.declared['experts'],
            stride13=self.declared['groups']['w13']['wire_stride'],
            stride2=self.declared['groups']['w2']['wire_stride'])
        groups = {}
        for group in MOE_GROUPS:
            modules = [PreparedTesseraFp8Module.concatenate(roles)
                       for roles in self.prepared[group]]
            groups[group] = PreparedTesseraFp8Module.stack(modules)
            # Release this group's per-expert owners before stacking the next.
            self.prepared[group] = None
            del modules
        return PreparedTesseraPackedMoeExperts(groups['w13'], groups['w2'])


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


def build_tessera_moe_method(scheme: Mapping, prefix: str, mode: str, layer, *,
                             research_selected: ResearchSelectedMoeConfig | None = None):
    """Construct the vLLM fused-MoE method serving a Tessera expert stack.

    ``layer`` is the ``RoutedExperts`` being built: its ``moe_config`` is what
    the runtime's backend oracle is asked about, so it is a constructor
    argument rather than something read back later.
    """
    if mode not in MODES:
        raise ValueError(f"unknown residency mode {mode!r}")
    if research_selected is not None and not isinstance(research_selected, ResearchSelectedMoeConfig):
        raise ValueError("research_selected requires an explicit ResearchSelectedMoeConfig")
    declared = validate_tessera_moe_scheme(scheme, prefix)
    family = declared["family"]
    from .scheme import refuse_a_family_with_no_expert_route
    refuse_a_family_with_no_expert_route(family, prefix)
    # THE PLATFORM GATE FOR THE EXPERT ROUTE (#457), asked here rather than in
    # ``config.get_quant_method`` so that both builders -- dense and expert --
    # are gated at their own front door and neither can be reached past it.
    # It sits AFTER the no-expert-route refusal because that one is the
    # narrower and more useful message (a 16-bit expert stack has no builder
    # on ANY platform), and BEFORE the vLLM fused-MoE imports below, which is
    # what "before any HIP kernel is touched" means on this path.
    from .backend import require_platform_backs
    from .contract import PAYLOAD_FAMILY_BY_ROUTE

    require_platform_backs(PAYLOAD_FAMILY_BY_ROUTE[family], f"tessera target {prefix!r}")
    if mode != MODE_RESIDENT:
        raise ValueError(
            f"tessera target {prefix!r}: the expert route serves {MODE_RESIDENT!r} only. A "
            f"streamed expert stack would decode E x 2 containers inside every forward, which "
            "is a different kernel story than the dense streamed route's and carries no "
            "measurement; refusing is what keeps 'streamed' meaning one thing.")

    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        convert_to_fp8_moe_kernel_format, make_fp8_moe_kernel, make_fp8_moe_quant_config,
        select_fp8_moe_backend)
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym, kFp8StaticChannelSym)
    from vllm.model_executor.utils import replace_parameter, set_weight_attrs

    groups = declared["groups"]

    class TesseraMoEMethod(FusedMoEMethodBase):
        """Per-channel FP8 W8A8 routed experts, decoded from Tessera wires."""

        def __init__(self, moe) -> None:
            super().__init__(moe)
            self._mode = mode if research_selected is None else 'research_selected'
            self._research_phase = 'new'
            self._packed = None
            self._rank_local_intake = None
            if research_selected is not None:
                from vllm.config import get_current_vllm_config
                if not get_current_vllm_config().model_config.enforce_eager:
                    raise ValueError(f"{prefix}: research selected experts require enforce_eager")
                self._require_research_parallel_contract()
            self._tp_size = (1 if research_selected is None
                             else research_selected.expected_tensor_parallel_size)
            self._tp_rank = (0 if research_selected is None else moe.moe_parallel_config.tp_rank)
            if not moe.is_act_and_mul:
                raise ValueError(
                    f"tessera target {prefix!r}: this MoE is not gated (is_act_and_mul is "
                    "False), so its w13 is one shard rather than the gate/up pair the "
                    "sidecar's groups describe. Refusing rather than loading a pair into a "
                    "single-shard tile.")
            # The runtime picks the backend, from the runtime's own predicate,
            # for the keys this route's tile actually is.
            self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
                config=self.moe, weight_key=kFp8StaticChannelSym,
                activation_key=kFp8DynamicTokenSym, allow_vllm_cutlass=True)
            if research_selected is not None:
                from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
                if self.fp8_backend != Fp8MoeBackend.TRITON or self.experts_cls.is_monolithic():
                    raise ValueError(f"{prefix}: research selected expert mapping covers stock TRITON FP8 only")

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
                raise ValueError(f"{prefix}: research selected experts require explicit "
                                 f"TP{expected_tp}/EP1/DP1/PCP1/SP1, a valid rank, and no EP/EPLB")
            if getattr(self.moe, 'defer_moe_finalize', False):
                raise ValueError(f"{prefix}: research selected experts refuse deferred finalize")
            if expected_tp > 1 and getattr(self.moe, 'skip_final_all_reduce', False):
                raise ValueError(f"{prefix}: research selected TP2 requires stock final all-reduce")

        @property
        def supports_eplb(self) -> bool:
            # EPLB adds redundant physical experts, so the parameter holds more
            # rows than the sidecar declares and the stride invariant has no
            # expert to check.  Say no rather than half-serve.
            return False

        # -- load -------------------------------------------------------
        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype, **extra):
            if research_selected is not None and self._research_phase != 'new':
                raise RuntimeError(f"{prefix}: research owner is already constructed")
            experts = int(declared["experts"])
            global_experts = int(extra.get("global_num_experts", num_experts))
            if int(num_experts) != experts or global_experts != experts:
                raise ValueError(
                    f"tessera target {prefix!r}: this rank holds {num_experts} of "
                    f"{global_experts} experts and the sidecar declares {experts}. The wire "
                    "stride is the maximum over EVERY expert's blob, so a rank holding a "
                    "subset cannot check it -- expert parallelism is refused here rather "
                    "than served on an unverifiable stride.")
            if int(hidden_size) != int(declared["hidden_size"]):
                raise ValueError(
                    f"tessera target {prefix!r}: vLLM builds hidden_size="
                    f"{hidden_size}, the sidecar declares {declared['hidden_size']}")
            full_intermediate = int(declared["intermediate_size"])
            if (full_intermediate % self._tp_size
                    or int(intermediate_size_per_partition) != full_intermediate // self._tp_size):
                raise ValueError(
                    f"tessera target {prefix!r}: this rank's intermediate size is "
                    f"{intermediate_size_per_partition} and the sidecar declares "
                    f"{declared['intermediate_size']} at the explicit TP{self._tp_size}. "
                    "Runtime padding or a cut outside that research contract is refused.")
            n_cols = full_intermediate // self._tp_size
            n_rows, k = 2 * n_cols, int(declared["hidden_size"])

            # The wires: one padded row per (expert, projection), the group's
            # declared stride wide, each with its own loader.
            incremental = research_selected is not None and self._tp_size == 2
            if incremental:
                # Stock constructs every owner before loading any weight. Keep
                # loader names/device anchors, without full-checkpoint staging.
                w13_wire = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
                w2_wire = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
                self._rank_local_intake = _RankLocalPackedIntake(
                    declared, prefix, w13_wire.device, self._tp_rank, self._tp_size)
            else:
                w13_wire = torch.nn.Parameter(
                    torch.zeros(experts, W13_PROJECTIONS, int(groups["w13"]["wire_stride"]),
                                dtype=torch.uint8), requires_grad=False)
                w2_wire = torch.nn.Parameter(
                    torch.zeros(experts, int(groups["w2"]["wire_stride"]), dtype=torch.uint8),
                    requires_grad=False)
            layer.register_parameter("w13_wire", w13_wire)
            layer.register_parameter("w2_wire", w2_wire)
            set_weight_attrs(w13_wire, {"weight_loader": self._load_wire})
            set_weight_attrs(w2_wire, {"weight_loader": self._load_wire})
            # NOT parameters: no checkpoint tensor fills them.  They are what
            # the loader learned from the blobs it was handed, and
            # ``unpack_moe_wires`` checks them against the declared stride.
            layer.tessera_w13_wire_len = torch.zeros(experts, W13_PROJECTIONS, dtype=torch.long)
            layer.tessera_w2_wire_len = torch.zeros(experts, dtype=torch.long)
            # The loader is handed a parameter, not a layer, so it reaches the
            # length companions through the method that registered them.
            self._w13_len = layer.tessera_w13_wire_len
            self._w2_len = layer.tessera_w2_wire_len
            self._wire_ids = {'w13': id(w13_wire), 'w2': id(w2_wire)}

            # The tile, allocated at create time exactly as the stock
            # per-channel method allocates it, so what the kernel sees is the
            # runtime's own parameter set and ``replace_parameter`` has
            # something to replace.
            if research_selected is None:
                for name, shape in (("w13_weight", (experts, n_rows, k)),
                                    ("w2_weight", (experts, k, n_cols))):
                    layer.register_parameter(name, torch.nn.Parameter(
                        torch.empty(*shape, dtype=torch.float8_e4m3fn), requires_grad=False))
                for name, shape in (("w13_weight_scale", (experts, n_rows, 1)),
                                    ("w2_weight_scale", (experts, k, 1))):
                    layer.register_parameter(name, torch.nn.Parameter(
                        torch.ones(*shape, dtype=torch.float32), requires_grad=False))
            layer.w13_input_scale = None
            layer.w2_input_scale = None
            layer.tessera_mode = self._mode
            layer.tessera_family = family
            layer.tessera_structure = declared["structure"]
            layer.tessera_activation_contract = ACTIVATION_CONTRACT
            layer.tessera_rows = n_rows
            layer.tessera_columns = k
            self._research_phase = 'loading'

        def _load_wire(self, param, loaded_weight, weight_name, shard_id, expert_id,
                       return_success: bool = False):
            """Validate and retain one projection, slicing TP2 during intake.

            ``RoutedExperts.load_weights`` calls ``param.weight_loader`` with
            these keywords; the stack's own ``weight_loader`` would return
            ``False`` here on the substring test and write nothing, which is
            why the parameter carries this one.
            """
            group, index = SHARD_TO_GROUP.get(str(shard_id), (None, None))
            if group is None:
                raise ValueError(
                    f"tessera target {prefix!r}: shard_id {shard_id!r} is not one of the "
                    f"shards the expert groups hold ({sorted(SHARD_TO_GROUP)})")
            if research_selected is not None:
                if self._research_phase != 'loading':
                    raise RuntimeError(f"{prefix}: research owner is not loading ({self._research_phase})")
                if type(expert_id) is not int or not 0 <= expert_id < int(declared['experts']):
                    raise ValueError(f"{prefix}: invalid global expert ID {expert_id!r}")
                if id(param) != self._wire_ids[group]:
                    raise ValueError(f"{prefix}: wire parameter does not belong to group {group}")
                previous = self._w13_len[expert_id, index] if group == 'w13' else self._w2_len[expert_id]
                if int(previous) != 0:
                    raise ValueError(f"{prefix}: expert {expert_id} shard {shard_id} already loaded")
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
            if self._rank_local_intake is not None:
                try:
                    self._require_research_parallel_contract()
                    # Resolve from the live loader anchor, not construction:
                    # stock can use a different explicit load device. Preserve
                    # the ordinary finalizer's CUDA promotion before packing.
                    device = param.device
                    if device.type != 'cuda' and torch.cuda.is_available():
                        device = torch.device('cuda', torch.cuda.current_device())
                    self._rank_local_intake.load(group, index, expert_id, blob, device=device)
                except Exception:
                    self._research_phase = 'failed'
                    self._rank_local_intake = None
                    raise
            elif group == "w13":
                param.data[int(expert_id), index, :length] = blob
            else:
                param.data[int(expert_id), :length] = blob
            if group == "w13":
                self._w13_len[int(expert_id), index] = length
            else:
                self._w2_len[int(expert_id)] = length
            return True if return_success else None

        def process_weights_after_loading(self, layer) -> None:
            if research_selected is not None:
                if self._research_phase != 'loading':
                    raise RuntimeError(f"{prefix}: research owner is not loading ({self._research_phase})")
                self._research_phase = 'failed'  # any incomplete/invalid load stays unusable
            if self._rank_local_intake is not None:
                intake, self._rank_local_intake = self._rank_local_intake, None
                prepared = intake.finish(self._w13_len, self._w2_len)
            else:
                packed = MoePacked(
                    w13_wire=layer.w13_wire.data.cpu(), w13_wire_len=layer.tessera_w13_wire_len,
                    w2_wire=layer.w2_wire.data.cpu(), w2_wire_len=layer.tessera_w2_wire_len)
                # Every refusal of ``moe_layout`` fires here on real bytes -- in
                # particular a declared stride that is not the maximum the loaded
                # lengths imply, which is the sidecar-vs-bytes disagreement no
                # other check sees.
                w13_blobs, w2_blobs = unpack_moe_wires(packed)
                device = layer.w13_wire.device
                if device.type != "cuda" and torch.cuda.is_available():
                    device = torch.device("cuda")
                prepare = (prepare_tessera_moe_experts if research_selected is None
                           else prepare_tessera_packed_moe_experts)
                prepared = prepare(
                    {"w13": w13_blobs, "w2": [[blob] for blob in w2_blobs]},
                    declared, prefix, device=device,
                    **({} if research_selected is None else {'tp_rank': self._tp_rank, 'tp_size': self._tp_size}))
            del layer.w13_wire, layer.w2_wire
            layer.tessera_w13_wire_len = None
            layer.tessera_w2_wire_len = None
            if research_selected is not None:
                self._w13_len = self._w2_len = self._wire_ids = None
                self._packed = prepared
                self._research_phase = 'ready'
                layer.tessera_decoder = f'research_selected_{research_selected.decode_backend}_window'
                layer.tessera_backend = str(getattr(self.fp8_backend, 'value', self.fp8_backend))
                return

            w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
                fp8_backend=self.fp8_backend, layer=layer,
                w13=prepared.w13_weight, w2=prepared.w2_weight,
                w13_scale=prepared.w13_weight_scale, w2_scale=prepared.w2_weight_scale,
                w13_input_scale=None, w2_input_scale=None)
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            replace_parameter(layer, "w13_weight_scale", w13_scale)
            replace_parameter(layer, "w2_weight_scale", w2_scale)

            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            assert self.experts_cls is not None
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config, moe_config=self.moe,
                fp8_backend=self.fp8_backend, experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables())
            layer.tessera_decoder = DECODER_TORCH_STOCK
            layer.tessera_backend = str(getattr(self.fp8_backend, "value", self.fp8_backend))

        def get_fused_moe_quant_config(self, layer):
            # The compact scales/config and kernel are per invocation. The
            # stock runner permits None here and keeps shared experts under
            # its normal non-MK-overlap execution path.
            if research_selected is not None:
                return None
            return make_fp8_moe_quant_config(
                fp8_backend=self.fp8_backend,
                w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
                a1_scale=None, a2_scale=None,
                per_act_token_quant=True, per_out_ch_quant=True, block_shape=None,
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                layer=layer)

        # -- forward ----------------------------------------------------
        def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
                  shared_experts_input):
            if research_selected is not None:
                return self._apply_selected(layer, x, topk_weights, topk_ids,
                                            shared_experts, shared_experts_input)
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
            if research_selected is not None:
                raise ValueError(f"{prefix}: research selected experts require external top-k routing")
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

        def research_resident_bytes(self) -> int:
            if research_selected is None or self._research_phase != 'ready':
                raise RuntimeError(f"{prefix}: research packed owner is not ready")
            return self._packed.resident_bytes()

        def _apply_selected(self, layer, x, weights, ids, shared_experts, shared_experts_input):
            self._require_research_parallel_contract()
            if self.moe.moe_parallel_config.tp_rank != self._tp_rank:
                raise ValueError(f"{prefix}: research selected rank changed after construction")
            if layer.expert_map is not None or int(layer.global_num_experts) != int(declared['experts']):
                raise ValueError(f"{prefix}: research selected experts require unchanged global expert IDs")
            if self._research_phase != 'ready':
                raise RuntimeError(f"{prefix}: research packed owner is not ready")
            if torch.compiler.is_compiling() or (x.is_cuda and torch.cuda.is_current_stream_capturing()):
                raise ValueError(f"{prefix}: research selected experts require eager execution without capture")
            if x.ndim != 2 or x.shape[1] != int(declared['hidden_size']):
                raise ValueError(f"{prefix}: research input must be [tokens, hidden_size]")
            expected_shape = (x.shape[0], self.moe.experts_per_token)
            if tuple(ids.shape) != expected_shape or tuple(weights.shape) != expected_shape:
                raise ValueError(f"{prefix}: research routing must be [tokens, top_k]")
            if ids.dtype not in (torch.int32, torch.int64) or not weights.is_floating_point():
                raise ValueError(f"{prefix}: research routing needs integer IDs and floating weights")
            if any(t.device != self._packed.device for t in (x, ids, weights)):
                raise ValueError(f"{prefix}: inputs and packed experts must share one device")
            if x.shape[0] == 0:
                # The stock runner owns any shared-expert output independently.
                return x.new_empty((0, int(declared['hidden_size'])))
            with torch.profiler.record_function('tessera_research_select_experts'):
                selected_ids = torch.unique(ids)
                if bool((selected_ids < 0).any()) or bool((selected_ids >= self._packed.experts).any()):
                    raise ValueError(f"{prefix}: research routing contains an invalid global expert ID")
                expert_map = torch.full((self._packed.experts,), -1, dtype=torch.int32, device=x.device)
                expert_map.scatter_(0, selected_ids.long(),
                                    torch.arange(selected_ids.numel(), dtype=torch.int32, device=x.device))
            with torch.profiler.record_function('tessera_research_decode_selected_experts'):
                selected = self._packed.decode(selected_ids,
                    max_experts_per_chunk=research_selected.max_experts_per_chunk,
                    backend=research_selected.decode_backend)
            quant = make_fp8_moe_quant_config(
                fp8_backend=self.fp8_backend,
                w1_scale=selected.w13_weight_scale, w2_scale=selected.w2_weight_scale,
                a1_scale=None, a2_scale=None, per_act_token_quant=True,
                per_out_ch_quant=True, block_shape=None,
                gemm1_alpha=getattr(layer,'swiglu_alpha',None),
                gemm1_beta=getattr(layer,'swiglu_beta',None),
                swiglu_limit=getattr(layer,'swiglu_limit',None), layer=layer)
            kernel = make_fp8_moe_kernel(moe_quant_config=quant, moe_config=self.moe,
                fp8_backend=self.fp8_backend, experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables())
            with torch.profiler.record_function('tessera_research_apply_selected_experts'):
                return kernel.apply(x, selected.w13_weight, selected.w2_weight, weights, ids,
                    activation=layer.activation, global_num_experts=layer.global_num_experts,
                    expert_map=expert_map, apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    shared_experts=shared_experts, shared_experts_input=shared_experts_input)

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

    return TesseraMoEMethod(layer.moe_config)
