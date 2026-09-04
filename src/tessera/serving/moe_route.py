"""The Tessera routed-MoE expert route: per-expert E4M3 wires as the stock FP8 stack.

WHAT IT SERVES.  One ``tessera.fused`` container per expert per GROUP -- ``w13``
(gate then up, the row order ``RoutedExperts._load_w13`` narrows to) and ``w2``
-- decoded at load into exactly the parameters vLLM's own fused-MoE kernels
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

THE ROWS ARE PADDED AND THE LENGTHS RIDE BESIDE THEM.  A checkpoint stores one
tensor per expert projection at that blob's exact length; the PARAMETER is
rectangular, so ``create_weights`` allocates the group's declared
``wire_stride`` and the loader records each blob's true length.  What comes
back out is ``tessera.moe_layout.unpack_moe_wires``, whose refusals are the
integrity gate: a length past its row, a length tensor that disagrees with the
expert count, and -- the one that catches a mis-declared sidecar -- a stride
that is not the maximum its lengths imply.

WHAT IT REFUSES, AND WHY EACH IS A REFUSAL RATHER THAN A FALLBACK.  Expert
parallelism and tensor parallelism inside an expert (the stride invariant
needs every expert's blob, and no expert slicer has been run); a residency
mode other than ``resident`` (a per-forward expert decode is a different
kernel story with no measurement); a family with no expert route
(``scheme.MOE_BUILDERS`` says which, and why the other two are absent); an
expert count, hidden size or intermediate size that disagrees with the
sidecar; and a non-gated MoE, whose ``w13`` is one shard rather than the pair
this route's groups describe.

WHAT IS NOT ATTESTED.  There is no ``routed_moe`` cell in this package's
``runtime_contract.json`` and there will not be one until a served census and
KL cover it on a real artifact.  This module is what the loader DOES, on the
``loader_axes`` precedent: attempted, measured on the load-and-execute
contract, and not claimed as served.  That measurement
(``docs/measurements/tessera-moe-route-load-2026-09-04.md``) was taken twice --
once on the pin, once on the build that registers ``Glm5Next`` -- and every
recorded field, backend selection and error number is identical, so the route
does not depend on which of the two loads it.  What it still does not cover:
the model-level ``load_weights`` hop above ``RoutedExperts``, the compiled
forward, and any expert count past four.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ..moe_layout import W13_PROJECTIONS, MoePacked, unpack_moe_wires
from .lane import MODE_RESIDENT, MODES
from .scheme import (MOE_GROUP_SHARDS, MOE_GROUPS, ROUTES, TESSERA_FP8,
                     expert_role_declarations, parse_tessera_expert_blob,
                     validate_tessera_moe_scheme)
from .telemetry import DECODER_TORCH_STOCK, emit_route, route_shape

__all__ = [
    "ACTIVATION_CONTRACT",
    "GEMM_SYMBOL",
    "SHARD_TO_GROUP",
    "PreparedTesseraMoeExperts",
    "census_expected",
    "census_symbol_base",
    "prepare_tessera_moe_experts",
    "build_tessera_moe_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_FP8]["activation_contract"]
GEMM_SYMBOL = "vllm.fused_moe.modular_kernel"


def census_expected(*, compiled: bool = False) -> dict:
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
    kernel roster written in our own prose, which is what principle 14 forbids;
    pinning one would refuse a box whose runtime picked another.

    NOT PUBLISHED, DELIBERATELY.  Unlike the dense routes' sets this one is not
    derived from ``scheme.ROUTE_LAUNCHES`` and no ``lane_eligibility`` cell
    carries it: ``runtime_contract.json`` is at v14 with ``structures:
    ["dense"]``, so a served expert stack is ``unattested`` in a census's
    cell-agreement block and honestly so.  This value is what the CENSUS
    compares a record against -- code against machine.  Publishing it (a
    structure axis in ``ROUTE_LAUNCHES`` and a ``routed_moe`` cell at the next
    contract version) is the document half, with its own consumers, and it is
    not made true by this function existing.
    """
    del compiled  # documented above: one launch has nothing to combine
    pair = (GEMM_SYMBOL, DECODER_TORCH_STOCK)
    return {"decode": {pair}, "batch": {pair}}


def census_symbol_base(symbol: str) -> str:
    """The entry point in a record's ``symbol``, without the runtime's backend pick.

    ``"vllm.fused_moe.modular_kernel:TRITON" -> "vllm.fused_moe.modular_kernel"``.
    A symbol with no suffix comes back unchanged, so a record that named some
    other entry point still fails the comparison it is fed to.
    """
    return str(symbol).split(":", 1)[0]

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
    role_declarations = expert_role_declarations(declared_group)
    weights, scales = [], []
    for expert, expert_blobs in enumerate(blobs):
        if len(expert_blobs) != len(role_declarations):
            raise ValueError(
                f"{target} expert {expert}: {len(expert_blobs)} container(s) for "
                f"{len(role_declarations)} declared projection(s) "
                f"{[r['roles'][0][0] for r in role_declarations]}")
        role_w, role_s = [], []
        for blob, role in zip(expert_blobs, role_declarations):
            (_name, parsed), = parse_tessera_expert_blob(
                blob, role, f"{target} expert {expert}", device=device)
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
    experts = int(declared["experts"])
    for group in MOE_GROUPS:
        if len(blobs[group]) != experts:
            raise ValueError(
                f"{target}: group {group!r} carries {len(blobs[group])} wire(s) for "
                f"{experts} experts; every expert of a stack has one container per group")
    w13, w13_scale = _decode_group(blobs["w13"], declared["groups"]["w13"], f"{target} w13", device)
    w2, w2_scale = _decode_group(blobs["w2"], declared["groups"]["w2"], f"{target} w2", device)
    return PreparedTesseraMoeExperts(
        w13_weight=w13.view(torch.float8_e4m3fn), w2_weight=w2.view(torch.float8_e4m3fn),
        w13_weight_scale=w13_scale, w2_weight_scale=w2_scale)


def build_tessera_moe_method(scheme: Mapping, prefix: str, mode: str, layer):
    """Construct the vLLM fused-MoE method serving a Tessera expert stack.

    ``layer`` is the ``RoutedExperts`` being built: its ``moe_config`` is what
    the runtime's backend oracle is asked about, so it is a constructor
    argument rather than something read back later.
    """
    if mode not in MODES:
        raise ValueError(f"unknown residency mode {mode!r}")
    declared = validate_tessera_moe_scheme(scheme, prefix)
    family = declared["family"]
    from .scheme import MOE_BUILDERS
    if family not in MOE_BUILDERS:
        raise ValueError(
            f"tessera target {prefix!r}: {family} has no expert route in this build "
            f"(scheme.MOE_BUILDERS names {sorted(MOE_BUILDERS)}). The absences are measured, "
            "not preferred: on this build the fused-MoE oracle resolves an NVFP4 expert arm "
            "only under a swiglu_limit clamp that changes the arithmetic the experts execute "
            "(docs/measurements/nvfp4-moe-oracle-2026-09-02.md), and a 16-bit expert stack is "
            "the passthrough quantization_config.ignore already gives. An expert stack is "
            "refused rather than decoded through another family's tile.")
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
            self._mode = mode
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

        @property
        def supports_eplb(self) -> bool:
            # EPLB adds redundant physical experts, so the parameter holds more
            # rows than the sidecar declares and the stride invariant has no
            # expert to check.  Say no rather than half-serve.
            return False

        # -- load -------------------------------------------------------
        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype, **extra):
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
            if int(intermediate_size_per_partition) != int(declared["intermediate_size"]):
                raise ValueError(
                    f"tessera target {prefix!r}: this rank's intermediate size is "
                    f"{intermediate_size_per_partition} and the sidecar declares "
                    f"{declared['intermediate_size']}. A cut intermediate size is tensor "
                    "parallelism inside an expert; the expert route has no unit slicer wired "
                    "and refuses rather than decoding whole rows into a rank's slice.")
            n_rows, k = int(groups["w13"]["rows"]), int(declared["hidden_size"])
            n_cols = int(declared["intermediate_size"])

            # The wires: one padded row per (expert, projection), the group's
            # declared stride wide, each with its own loader.
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

            # The tile, allocated at create time exactly as the stock
            # per-channel method allocates it, so what the kernel sees is the
            # runtime's own parameter set and ``replace_parameter`` has
            # something to replace.
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

        def _load_wire(self, param, loaded_weight, weight_name, shard_id, expert_id,
                       return_success: bool = False):
            """Copy one expert projection's blob into its padded row.

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
            blob = loaded_weight.reshape(-1)
            if blob.dtype != torch.uint8:
                raise ValueError(
                    f"tessera target {prefix!r} expert {expert_id} {shard_id}: a Tessera wire "
                    f"is uint8 bytes, the checkpoint holds {blob.dtype}")
            length = int(blob.numel())
            stride = int(param.data.shape[-1])
            if length == 0 or length > stride:
                raise ValueError(
                    f"tessera target {prefix!r} expert {expert_id} {shard_id}: a {length}-byte "
                    f"wire does not fit the group's declared wire_stride={stride}; the sidecar "
                    "and the bytes disagree about the row width")
            if group == "w13":
                param.data[int(expert_id), index, :length] = blob
                self._w13_len[int(expert_id), index] = length
            else:
                param.data[int(expert_id), :length] = blob
                self._w2_len[int(expert_id)] = length
            return True if return_success else None

        def process_weights_after_loading(self, layer) -> None:
            packed = MoePacked(
                w13_wire=layer.w13_wire.data.cpu(), w13_wire_len=layer.tessera_w13_wire_len,
                w2_wire=layer.w2_wire.data.cpu(), w2_wire_len=layer.tessera_w2_wire_len)
            # Every refusal of ``moe_layout`` fires here on real bytes -- in
            # particular a declared stride that is not the maximum the loaded
            # lengths imply, which is the sidecar-vs-bytes disagreement no
            # other check sees.
            w13_blobs, w2_blobs = unpack_moe_wires(packed)
            device = layer.w13_weight.device
            if device.type != "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
            prepared = prepare_tessera_moe_experts(
                {"w13": w13_blobs, "w2": [[blob] for blob in w2_blobs]},
                declared, prefix, device=device)
            del layer.w13_wire, layer.w2_wire
            layer.tessera_w13_wire_len = None
            layer.tessera_w2_wire_len = None

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

    return TesseraMoEMethod(layer.moe_config)
