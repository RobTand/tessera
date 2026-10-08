"""The checkpoint vocabulary that routes a Tessera wire to one of its routes.

A Tessera unit (``tessera.unit_artifact``) is a self-describing blob: its
manifest binds the grid, the body kind, the scale-plane kind, the rate schedule
and the convolutional code into an ``encoder_profile_id`` and a payload digest,
and Tessera's reader verifies all of it from bytes alone.  The serving plugin
therefore parses nothing of its own: it hands the blob to that reader and hands
the verified planes to its decoder.  What the plugin adds is the serving half:
the sidecar scheme a gate reads without parsing the blob, the refusal when blob
and scheme disagree, and the tile the tensor core runs.

FAMILY = ROUTE.  A Tessera family names what the decoded tile *is* on the
hardware, not the body inside the blob.  Three families, one plugin:

* ``TESSERA_NVFP4`` reads E2M1x2 WINDOW L14 over LUT16. Native block-scaled
  FP4 instructions execute W4A4. Each role keeps its own weight global.
* ``TESSERA_FP8`` is the scalar E4M3 grid over the CHANNEL scale plane (schema
  minor 3: one fp16 word per output row times a global), decoded to the stock
  per-channel FP8 pair (E4M3 bytes + one fp32 scale per row) and served through
  ``torch._scaled_mm`` W8A8.
* ``TESSERA_BF16`` is the scalar BF16 grid over the same CHANNEL plane and the
  same window body, decoded to a plain bf16 tile and served through the stock
  BF16 GEMM, W16A16.  It exists because the E4M3 *alphabet* floors the body at
  ~0.022 out-space from R = 6 upward while the identical trellis over bf16
  keeps halving; above ~6 bpp an 8-bit tile has nothing left to buy.  Its row
  scale is applied on the GEMM output and never folded into the tile.

The body (span-2 coset trellis, or the window body) is the decoder's business
and the manifest's fact; the scheme carries it only so a receipt can be scoped
to it.  A checkpoint may carry both families, module by module, and one process
serves them at one declared residency.

STRUCTURE.  ``structure`` names what kind of vLLM layer the target is:
``dense`` (a ``LinearBase``, one blob per module) or ``routed_moe`` (a
``RoutedExperts`` stack, one blob per expert per projection).  A dense
scheme declares one geometry and one exact ``wire_bytes``; a routed-MoE scheme
declares an expert count and the two GROUPS vLLM's fused-MoE kernel reads --
``w13`` (gate then up, one matrix) and ``w2`` -- each with its own geometry,
roles, rungs, and a ``wire_stride`` rather than an exact length, because an
expert's blob is as long as its own manifest made it.  Both shapes go through
one ``_validate_group``: a group and a dense module are the same object here,
and two copies of that check could drift.

``STRUCTURES`` is what this build DISPATCHES, which is a narrower question
than what this module can PARSE: a value enters it when a route exists to
serve it, and ``validate_tessera_moe_scheme`` is callable in its own right
before that, because a parser is not a promise to serve.

FUSED MODULES.  vLLM builds one method per module, so a fused module's roles
share a family -- and with it the grid, body, scale plane and input width the
route decodes to one tile.  They do NOT share a rate: every decoder here reads
each role from that role's own manifest, so ``q256`` is per member and the
sidecar spells it as a list when the members differ.  ``FUSED_MODULE_FIELDS``
is that rule as a value, and ``runtime_contract.json``'s ``fused_module`` block
is checked against it, so a producer's group allocator reads the constraint off
the runtime instead of guessing at it (#37).

WHAT IS ATTESTED.  Only what a ``lane_eligibility`` cell in this package's
``runtime_contract.json`` names.  A cell appears only when a container receipt
covers it (principle 14); absence resolves ``unattested``.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

__all__ = [
    "NVFP4_ACTIVATION_CONTRACT",
    "e2m1_shape_reason",
    "e2m1_expert_rate_reason",
    "FUSED_WINDOW_DENSE_E2M1_SYMBOL",
    "ROUTED_FUSED_WINDOW_E2M1_SYMBOL",
    "FP8_ACTIVATION_CONTRACT",
    "BF16_ACTIVATION_CONTRACT",
    "TESSERA_NVFP4",
    "TESSERA_FP8",
    "TESSERA_BF16",
    "TESSERA_FAMILIES",
    "TESSERA_SCHEME_KEY",
    "STRUCTURE_DENSE",
    "STRUCTURE_ROUTED_MOE",
    "STRUCTURES",
    "MOE_GROUPS",
    "MOE_GROUP_SHARDS",
    "MOE_GROUP_ROLES",
    "MOE_SHARD_PROJECTIONS",
    "MOE_GROUP_PROJECTIONS",
    "MOE_BUILDERS",
    "MOE_SOURCE_UNPACKED",
    "MOE_SOURCE_OUT_FIRST_CHUNKED",
    "MOE_SOURCE_IN_FIRST_INTERLEAVED",
    "MOE_SOURCE_LAYOUTS",
    "ROUTES",
    "ROUTE_LAUNCHES",
    "LAUNCH_FIELDS",
    "WINDOW_GEMV_SYMBOL",
    "WINDOW_GEMM_SYMBOL",
    "A4_DENSE_GEMM_SYMBOL",
    "A4_GROUPED_GEMM_SYMBOL",
    "WINDOW_MOE_COMPACT_SYMBOL",
    "ROUTED_FUSED_WINDOW_SYMBOL",
    "FUSED_WINDOW_DENSE_SYMBOL",
    "EXPERIMENTAL_LAUNCHES",
    "DECODE_ONCE_DENSE_SYMBOL",
    "EAGER_ONLY_LAUNCHES",
    "experimental_launch_pairs",
    "parse_compact_blob_for_scheme",
    "parse_compact_tessera_expert_blob",
    "MOE_GEMM_SYMBOL",
    "moe_census_symbol_base",
    "eager_regime_problem",
    "parse_eager_shape",
    "regime_of_m",
    "route_launches",
    "launch_pairs",
    "GROUP_SIZE",
    "FUSED_MODULE_FIELDS",
    "FUSED_MODULE_SCHEMA",
    "FUSED_Q256_SPELLING",
    "FUSED_CONTAINER",
    "is_tessera_scheme",
    "route_for_grid",
    "attested_cells",
    "decide_lane_requirements",
    "lane_wire_report",
    "wire_facts_of_parsed",
    "refuse_a_family_with_no_expert_route",
    "refuse_unserveable_wire",
    "refuse_unreachable_lane",
    "lane_rate_report",
    "validate_tessera_scheme",
    "validate_tessera_moe_scheme",
    "parse_tessera_blob_for_scheme",
    "expert_role_declarations",
    "expert_group_q256",
    "expert_rungs_mixed",
    "stack_effective_rungs",
    "parse_tessera_expert_blob",
]

#: The A-side contract each route executes, in the vocabulary the packaged
#: runtime contract publishes.  Defined here rather than beside the telemetry
#: because a producer reads them from the contract on a machine with no torch:
#: this module must stay importable without it.
NVFP4_ACTIVATION_CONTRACT = "e2m1_group16_ue4m3_static"
FP8_ACTIVATION_CONTRACT = "fp8_per_token_dynamic"
#: W16A16.  There is no A-side quantiser to name -- the route hands ``x``
#: to the stock bf16 GEMM as it arrives -- and the honest spelling of that is
#: a contract that says so, not the absence of one.  A gate that reads this
#: field must be able to tell "unquantised, by design" from "nobody filled it
#: in", and only a value can carry that.
BF16_ACTIVATION_CONTRACT = "bf16_unquantized"

TESSERA_NVFP4 = "TESSERA_NVFP4"
TESSERA_FP8 = "TESSERA_FP8"
TESSERA_BF16 = "TESSERA_BF16"

#: A scheme with ``family`` in ``TESSERA_FAMILIES`` is ours.
TESSERA_SCHEME_KEY = "family"

#: What kind of vLLM layer a scheme's target is.  ``dense`` is one blob per
#: ``LinearBase``; ``routed_moe`` is one blob per expert PROJECTION on a
#: ``RoutedExperts`` stack.  ``STRUCTURES`` is what this build DISPATCHES, and
#: a structure outside it is refused by name rather than served through a
#: method that would read the wrong tensor rank.
#: The names live in the core ``tessera.structure`` module, which ``export`` and
#: ``cached_unit`` read without importing this plugin layer.
from ..structure import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE, STRUCTURES  # noqa: E402,F401

#: THE TWO EXPERT GROUPS, AND WHY THERE ARE EXACTLY TWO.  vLLM's
#: ``RoutedExperts`` holds an expert's gate and up in ONE ``w13`` matrix
#: (gate at rows ``[0:N]``, up at ``[N:2N]`` -- ``RoutedExperts._load_w13``
#: narrows on ``shard_id``) and its down alone in ``w2``.  A Tessera MoE
#: checkpoint therefore writes one fused container per projection per expert.
#: The group's containers stack into the tile the kernel reads, with the same
#: members in the same order.  ``MOE_GROUP_SHARDS`` is the runtime's own
#: shard vocabulary for each group, in the row order the group stacks; a
#: producer reads the order off this table instead of restating it.
MOE_GROUPS = ("w13", "w2")
MOE_GROUP_SHARDS: dict[str, tuple[str, ...]] = {"w13": ("w1", "w3"), "w2": ("w2",)}
#: How many projection containers each group holds. DERIVED from the shard table,
#: never a second literal: the members of a group are exactly the shards the
#: runtime loads into it.
MOE_GROUP_ROLES: dict[str, int] = {g: len(s) for g, s in MOE_GROUP_SHARDS.items()}
#: The canonical wire role for each runtime shard. Source checkpoints may
#: spell their tensors with either vocabulary, but wire roles are descriptive.
#: The exporter and sidecar reader share this table so a self-consistent pair
#: of sidecar and blobs cannot reinterpret the runtime's gate/up row order.
MOE_SHARD_PROJECTIONS = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
MOE_GROUP_PROJECTIONS = {
    group: tuple(MOE_SHARD_PROJECTIONS[shard] for shard in shards)
    for group, shards in MOE_GROUP_SHARDS.items()
}

#: The checkpoint layouts the exporter can prove it interpreted.  This is
#: provenance rather than a runtime layout: all three are normalised to the
#: same canonical per-expert gate/up/down wires before vLLM sees them.  Old
#: schemes predate the field and can only have come from the original
#: per-expert writer, so their closed-world default is ``unpacked_per_expert``.
MOE_SOURCE_UNPACKED = "unpacked_per_expert"
MOE_SOURCE_OUT_FIRST_CHUNKED = "out_first_chunked"
MOE_SOURCE_IN_FIRST_INTERLEAVED = "in_first_interleaved"
MOE_SOURCE_LAYOUTS = (
    MOE_SOURCE_UNPACKED,
    MOE_SOURCE_OUT_FIRST_CHUNKED,
    MOE_SOURCE_IN_FIRST_INTERLEAVED,
)

#: WHICH FAMILIES HAVE AN EXPERT ROUTE, and the one home for that rule.
#: FAMILY = ROUTE holds on the expert stack exactly as it does on a Linear,
#: so this is ``ROUTES``' shape again -- a builder per family -- and a family
#: absent from it is refused by name rather than served through another
#: family's decode.
#:
#: ``TESSERA_FP8``, ``TESSERA_NVFP4`` and ``TESSERA_BF16`` are here. The FP8
#: builder serves E4M3 window wires on the compact native window lane where
#: the shared compact reader is published, and otherwise decodes them into
#: vLLM's per-channel FP8 fused-MoE parameter set; the
#: The NVFP4 builder reads paired WINDOW L14 and LUT16 scales.
#: FusedRoutedE2M1MoE executes native W4A4. No stock substitute exists.
#: A builder is a dispatch fact, not a served qualification: which
#: ``(family, structure)`` cells the packaged contract attests is
#: ``lane_eligibility``'s to say, and ``attested_cells`` reads it.
#: ``TESSERA_BF16`` (tessera#609) is the compressed BF16-alphabet wire with a
#: per-row scale, served by the same builder as FP8 on the compact native
#: window lane (``native_window_moe``), which decodes the packed wire in
#: registers and materialises no expert tile.  The row scale applies on
#: the fp32 accumulator epilogue, and the launch stamps its own family
#: decoder (``DECODER_NATIVE_WINDOW_MOE_COMPACT_BF16``).  There is no
#: materialising fallback: without the compact reader a BF16 stack refuses.
#: Plain source BF16 passthrough is a different thing and uses ``ignore``.
MOE_BUILDERS: dict[str, tuple[str, str]] = {
    TESSERA_FP8: ("tessera.serving.moe_route", "build_tessera_moe_method"),
    TESSERA_NVFP4: ("tessera.serving.nvfp4_moe_route", "build_tessera_nvfp4_moe_method"),
    TESSERA_BF16: ("tessera.serving.moe_route", "build_tessera_moe_method"),
}

#: The route table states the byte reader and activation contract.
#: NVFP4 reads only paired E2M1 WINDOW with the LUT16 scale plane.
#: FP8 reads scalar E4M3 with the CHANNEL scale plane.
#: per-row fp32 scale comes from.  ``tile`` is the stock tensor the route
#: decodes to, ``columns_multiple`` the K quantum its mainloop needs,
#: ``activation_contract`` what it executes on the A side, and ``gemm_symbol``
#: the callable it actually invokes -- the route module stamps that field on
#: every route record and the census compares against the same field, so
#: "which GEMM ran" has one spelling and adding a route that calls something
#: else cannot silently read as a refusal.
#:
#: ``body``/``span`` name the trellis body the route's decoder reads, and they
#: are the same fact its own module already refuses by name at load
#: (``nvfp4_route`` for NVFP4, ``fp8_route`` for FP8,
#: ``bf16_route`` for BF16).  They are written here because the PRODUCER needs
#: them too: ``refuse_unserveable_wire`` is the export-time half of the same
#: rule, and the exporter used to carry its own copy in an if/elif -- a third
#: statement about one decoder.
ROUTES: dict[str, dict] = {
    TESSERA_NVFP4: {
        "grids": ("E2M1x2",), "plane": "LUT",
        "body": "WINDOW", "span": 1, "window_bits": 14,
        "short": "NVFP4",
        "grid_kind": "the paired E2M1",
        "builder": ("tessera.serving.nvfp4_route", "build_tessera_nvfp4_method"),
        "tile": "native E2M1x2 window words, LUT16 group-16 scales and per-role global",
        "columns_multiple": 64,
        "activation_contract": NVFP4_ACTIVATION_CONTRACT,
        "gemm_symbol": "tessera.routed_fused_e2m1.dense_forward_quantized",
    },
    TESSERA_FP8: {
        "grids": ("E4M3",), "plane": "CHANNEL",
        "body": "WINDOW", "span": 1,
        "short": "FP8",
        "grid_kind": "the scalar E4M3",
        "builder": ("tessera.serving.fp8_route", "build_tessera_fp8_method"),
        "tile": "fp8 per-channel (E4M3 bytes, one fp32 scale per row)",
        "columns_multiple": 16,
        "activation_contract": FP8_ACTIVATION_CONTRACT,
        "gemm_symbol": "torch._scaled_mm",
    },
    TESSERA_BF16: {
        "grids": ("BF16",), "plane": "CHANNEL",
        # The SAME body and span the E4M3 route reads -- the 16-bit family is
        # the window recipe with a wider alphabet, not a different trellis --
        # and ``bf16_route._parse_unit`` refuses anything else by name at load.
        # What differs between the two routes is the alphabet the window table
        # snaps to and the tile the decode lands in, and neither is a body
        # fact -- which is why this row is FP8's row here and in
        # ``sharding.ROUTE_TP_AXES``, for the same reason in both places.
        "body": "WINDOW", "span": 1,
        "short": "BF16",
        "grid_kind": "the scalar BF16",
        "builder": ("tessera.serving.bf16_route", "build_tessera_bf16_method"),
        "tile": "bf16 (the raw table values, one fp32 scale per row applied on the GEMM output)",
        # No K quantum.  The other two routes decode to a PACKED tile whose
        # mainloop reads groups -- a nibble pair, a group-16 block scale -- and
        # 16 is that group.  A bf16 tile is one word per weight, so the GEMM
        # takes any K, and asserting a quantum here would refuse a geometry
        # this route serves.
        "columns_multiple": 1,
        "activation_contract": BF16_ACTIVATION_CONTRACT,
        # NOT ``torch._scaled_mm``: there is no scale to hand a scaled GEMM.
        # The row scale is an fp32 epilogue, so this route calls the stock
        # matmul and says so.
        "gemm_symbol": "torch.mm",
    },
}


def e2m1_shape_reason(rows: int, columns: int, *, structure: str = STRUCTURE_DENSE,
                       projection: "str | None" = None) -> "str | None":
    """Metadata-only native FP4 shape rule, shared by writer and reader."""
    rows, columns = int(rows), int(columns)
    column_multiple = ROUTES[TESSERA_NVFP4]["columns_multiple"]
    if columns < 256 or columns % column_multiple:
        return (f"{columns} columns; the native E2M1 launch needs a multiple of "
                f"{column_multiple} and at least 256")
    if structure == STRUCTURE_DENSE:
        multiple = 32
    elif structure == STRUCTURE_ROUTED_MOE and projection in ("gate_proj", "up_proj", "down_proj"):
        multiple = 256 if projection == "down_proj" else 128
    else:
        return f"no native E2M1 geometry for structure {structure!r}, projection {projection!r}"
    if rows <= 0 or rows % multiple:
        return f"{rows} rows; the native E2M1 {projection or structure} launch needs a positive multiple of {multiple}"
    return None


def e2m1_expert_rate_reason(rungs) -> "str | None":
    """The native stride rule for a normalized [experts][roles] rate matrix.

    The caller validates dimensions and positive rungs first.
    A row holds down, gate/up, or gate/up/down in that order.
    Column permutations remain each projection and expert's own data.
    """
    if any(row != rungs[0] for row in rungs[1:]):
        return "experts disagree on their run tables; the lane reads one schedule per stack"
    if len(rungs[0]) >= MOE_GROUP_ROLES["w13"] and rungs[0][0] != rungs[0][1]:
        return "the gate/up launch reads one tile stride for both; gate and up need the same q256"
    return None

#: DERIVED from ``ROUTES``, never written twice.  FAMILY = ROUTE is the rule,
#: so the set of families a build serves is exactly the set of routes it has:
#: a third family (say a WINDOW/CHANNEL body whose alphabet is snapped to bf16
#: and decoded to a bf16 tile for the stock GEMM) is one route module, one
#: ``ROUTES`` entry naming its builder, and its contract rows -- and nothing
#: here or in ``lane`` has to be edited to admit it.  A hand-written tuple was
#: a fourth place to remember.
TESSERA_FAMILIES = tuple(ROUTES)

#: THE LAUNCHES EACH ROUTE MAKES, and the conditions under which it makes one.
#:
#: ``ROUTES[...]["gemm_symbol"]`` answers "which GEMM does this route call",
#: which was the whole answer while a route had exactly one launch. The dense
#: FP8 and BF16 routes have three (issue #111): the materialised tile under the
#: stock GEMM, the same GEMM over a tile the window-GEMV lane's kernel decoded,
#: and -- where the lane prepared -- the lane's own ``gemv`` op.  Which one runs
#: is a function of the token count M and of the RESIDENCY (``fp8_route`` and
#: ``bf16_route`` both set ``layer.tessera_gemv = None`` in ``resident`` mode,
#: so the lane exists in ``streamed`` alone).  The residency is an axis a
#: ``lane_eligibility`` cell carries directly; M reaches a cell through its
#: REGIME, and the two words for a regime are the whole of what this table has
#: to get right (see :func:`regime_of_m`).  So the table is here, torch-free,
#: and it is read by three sides that must not disagree about one runtime:
#:
#: * the ROUTES themselves -- the dense and expert ``census_expected`` sets
#:   are derived from it, and ``GEMV_SYMBOL`` is
#:   read from it rather than spelled a third time beside the two dispatches;
#: * the CONTRACT -- ``contract.validate_serving_contract`` refuses a
#:   ``lane_eligibility`` cell whose ``executes`` list is not exactly the
#:   launches this table admits for that cell's regime and residency, which is
#:   what makes the published value derived rather than asserted (principle 14);
#: * the CENSUS -- ``census.cell_launch_agreement`` joins a served record to
#:   the cell that covers it.
#:
#: ``structures`` separates a dense Linear from a routed-expert stack even
#: when both serve the same payload family. The expert FP8 route materialises
#: once at load and calls the runtime's modular fused-MoE kernel in resident
#: mode; it never takes a dense GEMM or a window-GEMV lane. This table states
#: dispatch capability, not attestation: it does not create a served cell.
#:
#: ``lane`` names the ``ext.NATIVE_EXTENSIONS`` entry a launch needs, or is
#: ``None`` for a launch the route makes with no extension at all.  A lane
#: launch is reachable only at a rung the lane reads -- the predicate the
#: extension publishes at ``lane.requires`` -- and the contract validator ties
#: the cell's rungs to it, so a GEMV cell cannot outlive the kernel's constants.
LAUNCH_FIELDS = ("symbol", "decoder", "regimes", "modes", "structures", "lane",
                 "when_lane_absent")


#: TWO VOCABULARIES SAY "DECODE", AND THIS IS THE ONE THE TABLE ABOVE SPEAKS.
#:
#: The KERNEL's decode is ``M <= kernel_window_gemv.GEMV_MAX_M`` -- what
#: ``fp8_gemv.decode_is_gemv`` decides, and it spans eight token counts.  The
#: CONTRACT's decode is the ONE-ROW forward: ``contract.CENSUS_PHASE_REGIMES``
#: maps the census's one-row phase to ``decode`` and its many-row phase to
#: ``batch``, and says so in its own words -- "a regime is a *problem shape*
#: and the batch cell covers every M > 1 forward, not only a first prefill".
#:
#: A ``lane_eligibility`` cell is keyed by the CONTRACT's regime, and so is
#: every census record (the tool stamps ``CENSUS_PHASE_REGIMES[phase]``), so
#: that is the word this table is written in.  Reading the kernel's word into
#: a cell is how the first version of this table came to say that the batch
#: regime never launches the GEMV: true of the census's 64-row prefill, false
#: of the 2-to-8-row forwards the same regime covers, on which the lane serves
#: the GEMV exactly as it does at one row.  That is the defect #111 was filed
#: about, one regime over.
def regime_of_m(m: int) -> str:
    """The contract's regime for a forward of ``m`` tokens."""
    if int(m) < 1:
        raise ValueError(f"M={m} is not a forward; a regime is a shape a route was called on")
    return "decode" if int(m) == 1 else "batch"


#: ``telemetry.route_shape``'s canonical concrete spelling.  Eager only: a
#: compiled forward is shape-polymorphic and stamps ``M*`` on purpose, so a
#: parser that accepted it would be reading a graph as a forward.
_EAGER_SHAPE = re.compile(r"M([1-9][0-9]*):N([1-9][0-9]*):K([1-9][0-9]*)")


def parse_eager_shape(value: Any) -> "tuple[int, int, int]":
    """Read ``telemetry.route_shape``'s canonical concrete ``M:N:K`` spelling."""
    match = _EAGER_SHAPE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"eager shape must be canonical M<n>:N<n>:K<n>, got {value!r}")
    return tuple(int(dimension) for dimension in match.groups())


def eager_regime_problem(shape: Any, regime: Any) -> "str | None":
    """Why an eager record does not exercise ``regime`` -- or ``None`` if it does.

    THE PHASE LABEL IS THE REQUEST; ``M`` IS WHAT RAN.  A census names its
    phases and a cell is keyed by the regime that phase maps to, so a gate
    reading the label alone attests the forward it asked for rather than the
    one the machine took: an eight-row observation filed under the decode
    phase was counted as a covered decode launch, and resident FP8 publishes
    the same pair in both regimes, so nothing downstream could see it (#207).

    One home for the rule (``regime_of_m`` above owns the M -> regime map),
    called by the shared cell matcher (``census.cell_launch_agreement``) and by
    the census's own shape check, so the two cannot drift.  Absent or
    unparseable evidence is a problem, not a pass: an attestation with no shape
    behind it is the thing this check exists to refuse.
    """
    try:
        m, _, _ = parse_eager_shape(shape)
    except ValueError as exc:
        return str(exc)
    if regime is None:
        return f"shape M{m} names no regime to be checked against"
    observed = regime_of_m(m)
    if observed != regime:
        return (f"shape M{m} is a {observed}-regime forward and does not exercise the "
                f"declared regime {regime}")
    return None

#: The op the window-GEMV lane dispatches through, in the spelling
#: ``kernel_window_gemv`` registers it under.  It lives HERE, torch-free,
#: because a producer reading the contract has to be able to resolve a cell's
#: ``executes`` entry without importing the kernel -- the same reason
#: ``gemm_symbol`` is a ``ROUTES`` field and not a literal in the route module.
WINDOW_GEMV_SYMBOL = "tessera_window_gemv::gemv"
#: The dense native window GEMM: the functional custom op the compact
#: loader's ``WindowGemvUnit`` is served through (``tessera.window_gemm``
#: behind ``serving.native_window``).  Same spelling rule as the GEMV's:
#: the census compares the route record against this table.
WINDOW_GEMM_SYMBOL = "tessera::window_gemm_dense"

#: The native lanes' own spellings, in the module that executes them, so a
#: route owner reports the same string the code does.  The window MoE adapter
#: left ``EXPERIMENTAL_LAUNCHES`` at contract v38 and the two A4 pairs below at
#: contract v39 (tessera#604), each when a served census earned it cells.
#: A4 (E2M1/span-2) dense: ``tessera.kernel_a4.a4_span2_gemm``.
A4_DENSE_GEMM_SYMBOL = "tessera.kernel_a4.a4_span2_gemm"
#: A4 routed experts: ``tessera.kernel_a4.a4_span2_grouped_gemm``.
A4_GROUPED_GEMM_SYMBOL = "tessera.kernel_a4.a4_span2_grouped_gemm"
#: Window routed experts: ``tessera.native_window_moe``'s adapter call, the
#: compact MoE lane for both window families.  Both families run the
#: epilogue arithmetic and stamp their own family decoder, so one
#: symbol never stands for two families.
WINDOW_MOE_COMPACT_SYMBOL = "tessera.native_window_moe.NativeWindowMoE.__call__"
#: The fused warp-specialised routed window MoE (``tessera.routed_fused``,
#: tessera#640): gate/up + SwiGLU in one persistent kernel, the down
#: projection in a second launch of the same kernel with a fixed-order
#: per-token reduction.  A NEW identity, not the compact adapter under a new
#: name: it decodes each weight once per tile and reuses it across routes,
#: schedules by route count on the device, and its down reduction is
#: deterministic where the compact adapter's is an fp32 atomic.  One
#: epilogue arithmetic, one decoder per family, its own symbol set.
ROUTED_FUSED_WINDOW_SYMBOL = "tessera.routed_fused.FusedRoutedWindowMoE.__call__"
#: Native paired WINDOW W4A4 identities; no stock or TCQ serving substitute.
FUSED_WINDOW_DENSE_E2M1_SYMBOL = "tessera.routed_fused_e2m1.dense_forward_quantized"
ROUTED_FUSED_WINDOW_E2M1_SYMBOL = "tessera.routed_fused_e2m1.FusedRoutedE2M1MoE.__call__"
#: The same kernel's DENSE identity (contract v43): the functional custom op
#: ``serving.native_window`` registers, which launches ``routed_fused_kernel``'s
#: E = 1 case once per role of a dense Linear into the role's column slice of
#: one output, splitting K at decode shapes behind a fixed-order reduce.  A
#: different launch than ``WINDOW_GEMM_SYMBOL`` over the same function of the
#: wire (its MMA accumulation order differs), so its own symbol and decoders.
FUSED_WINDOW_DENSE_SYMBOL = "tessera::fused_window_dense"
#: The E4M3 family's decode-once dense prefill lane (tessera#931): the
#: module's weights decoded once at load (``serving.e4m3_prefill``) and
#: served by ``torch._scaled_mm`` row-wise for M at or above
#: ``e4m3_prefill.MIN_M``.  Below that M the module's window lane runs.
DECODE_ONCE_DENSE_SYMBOL = "tessera.serving.e4m3_prefill.prefill_apply"
#: The entry point the expert route calls. Its recorded backend suffix is
#: selected by vLLM at runtime and remains in the census receipt.
MOE_GEMM_SYMBOL = "vllm.fused_moe.modular_kernel"

#: The decoder each launch stamps.  Strings rather than an import of
#: ``telemetry``, which imports torch; ``tests/test_serving_contract.py`` ties
#: every one of them to ``telemetry.DECODERS`` where torch is installed.
_DECODER_NATIVE_SPAN2 = "native_span2"
_DECODER_TORCH_STOCK = "torch_materialize_stock"
_DECODER_NATIVE_WINDOW_GEMM = "native_window_gemm"
#: The same dense GEMM on the BF16 family: raw bf16 table values with
#: the fp32 row scale on the accumulator epilogue.  Its own string
#: because it serves a distinct family of the same wire.
_DECODER_NATIVE_WINDOW_GEMM_BF16 = "native_window_gemm_bf16"
#: The native A4 lanes: the span-2 GEMM decodes the packed planes in-kernel
#: (dense) and the grouped form does it per selected expert.  Distinct from
#: ``native_span2`` (the load-time span-2 DECODE) and from ``torch_window``.
_DECODER_NATIVE_SPAN2_GEMM = "native_span2_gemm"
_DECODER_NATIVE_SPAN2_GROUPED = "native_span2_grouped"
#: The compact window MoE adapter: routed experts served from the loader's
#: packed ``WindowGemvUnit``s with no decoded tile.  The FP8 family keeps the
#: per-token native A quant and the row scale on the fp32 accumulator; the
#: BF16 family keeps raw bf16 values with the row scale on the same
#: epilogue and stamps its own decoder, because a cell must name
#: which family it attests.
_DECODER_NATIVE_WINDOW_MOE_COMPACT = "native_window_moe_compact"
_DECODER_NATIVE_WINDOW_MOE_COMPACT_BF16 = "native_window_moe_compact_bf16"
#: The fused routed window MoE lane (tessera#640), one epilogue arithmetic
#: and one decoder per family.  Its own strings for the reason the
#: compact pair has two: a census must be able to say which kernel
#: and which family served a stack, and the fused lane's deterministic
#: reduction is a different numerical function of the same wire than
#: the compact adapter's atomic one.
_DECODER_NATIVE_ROUTED_FUSED_WINDOW = "native_routed_fused_window"
_DECODER_NATIVE_ROUTED_FUSED_WINDOW_BF16 = "native_routed_fused_window_bf16"
#: The fused kernel's dense identity (contract v43), one decoder per family.
_DECODER_NATIVE_FUSED_WINDOW_DENSE = "native_fused_window_dense"
_DECODER_NATIVE_FUSED_WINDOW_DENSE_BF16 = "native_fused_window_dense_bf16"
#: The E4M3 family's fused identities on its own instruction
#: (``tessera_routed_fused_mma_e4m3``, ``mma.sync.m16n8k32.e4m3``): the same
#: exact products as the two E4M3 pairs above in another fp32 accumulation
#: order, so their own strings.
_DECODER_NATIVE_ROUTED_FUSED_WINDOW_E4M3MMA = "native_routed_fused_window_e4m3mma"
_DECODER_NATIVE_FUSED_WINDOW_DENSE_E4M3MMA = "native_fused_window_dense_e4m3mma"
_DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1 = "native_fused_window_dense_e2m1"
_DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1 = "native_routed_fused_window_e2m1"
_DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3 = "native_window_decode_once_e4m3"

_ALL_REGIMES = ("batch", "decode")
_ALL_MODES = ("resident", "streamed")
#: No launch in this table names a lane since #538: the only launch that did
#: was the dense window-GEMV lane's, and the dispatch that made it was retired
#: by ``1b767a207``.  ``lane`` stays a ``LAUNCH_FIELDS`` field -- the narrowing
#: in :func:`route_launches` is about the shape of a launch, not about which
#: launches exist today -- and ``ext.WINDOW_GEMV_MODULE_NAME`` remains the home
#: of the extension's name for the load path that still builds it.


def _dense_native_window_launch(decoder: str, fused_decoder: str, lane: str) -> tuple[dict, ...]:
    """The compact loader's native window GEMM, the dense half of both routes.

    ``serving.native_window`` prepares each dense role from the verified wire
    (``tessera.compact_prep.prepare_window_compact``) and runs a packed
    bitstream GEMM through one functional custom op; it serves every M in both
    residencies.  An E4M3 unit is the fp8 family; a BF16 unit is the
    value family.  Both serve the epilogue arithmetic.  The decoder
    names the family, so a cell attesting one cannot be read as
    attesting the other.

    TWO LAUNCHES since contract v43.  The Triton GEMM (``WINDOW_GEMM_SYMBOL``)
    needs no extension lane, carries no ``lane`` and is not a
    ``when_lane_absent`` fallback: it still runs beside the fused identity,
    for every module ``routed_fused.fused_dense_window_supported`` refuses
    (more than two rates, rows not a multiple of 4, a permuted column order), for a
    box whose toolchain cannot build the library, and for
    ``TESSERA_DENSE_FUSED=0``.  The fused window kernel's dense identity
    (``FUSED_WINDOW_DENSE_SYMBOL``) is the dispatch for every module the
    predicate admits -- the q256 1024 GLM dense MLPs and shared experts -- and
    names its extension ``lane``, so a cell derives it only at a rung the
    extension's own ``lane.requires`` admits (``contract._lanes_a_rung_reaches``).
    ``native_window.prepare_dense_native_module`` decides the lane once per
    module and ``apply`` stamps that module's pair.
    """
    return (
        {"symbol": WINDOW_GEMM_SYMBOL, "decoder": decoder,
         "regimes": _ALL_REGIMES, "modes": _ALL_MODES, "lane": None,
         "structures": (STRUCTURE_DENSE,),
         "when_lane_absent": False},
        {"symbol": FUSED_WINDOW_DENSE_SYMBOL, "decoder": fused_decoder,
         "regimes": _ALL_REGIMES, "modes": _ALL_MODES, "lane": lane,
         "structures": (STRUCTURE_DENSE,),
         "when_lane_absent": False},
    )


ROUTE_LAUNCHES: dict[str, tuple[dict, ...]] = {
    # Launch support is not qualification. Historical TCQ cells are withdrawn;
    # a real serving receipt must earn new WINDOW lane_eligibility cells.
    TESSERA_NVFP4: (
        {"symbol": FUSED_WINDOW_DENSE_E2M1_SYMBOL,
         "decoder": _DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1,
         "regimes": _ALL_REGIMES, "modes": _ALL_MODES,
         "lane": "tessera_routed_fused_e2m1",
         "structures": (STRUCTURE_DENSE,), "when_lane_absent": False},
        {"symbol": ROUTED_FUSED_WINDOW_E2M1_SYMBOL,
         "decoder": _DECODER_NATIVE_ROUTED_FUSED_WINDOW_E2M1,
         "regimes": _ALL_REGIMES, "modes": ("resident",),
         "lane": "tessera_routed_fused_e2m1",
         "structures": (STRUCTURE_ROUTED_MOE,), "when_lane_absent": False},
    ),
    # The dense half is TWO launches since contract v43: ``fp8_route.apply``
    # runs the prepared module's packed native window GEMM -- the Triton op,
    # or the fused window kernel's dense identity where the module's roles
    # admit it -- for every M and both residencies and raises rather than fall
    # back, so the route's own ``DENSE_LAUNCHES`` is this set and
    # ``tests/test_serving_contract.py`` asserts the equality.  The window-GEMV
    # lane's three launches stood here until #538 and were retired from the
    # dispatch by ``1b767a207``; a table that outlived its dispatch is what let
    # the published ``lane_eligibility`` cells go on naming them.
    TESSERA_FP8: _dense_native_window_launch(
        _DECODER_NATIVE_WINDOW_GEMM, _DECODER_NATIVE_FUSED_WINDOW_DENSE,
        "tessera_routed_fused_e4m3") + (
        # The dense identity on the E4M3 instruction (``TESSERA_FUSED_E4M3_
        # MMA=e4m3``): the same kernel, launch symbol and admission predicate
        # as the row above, built as ``tessera_routed_fused_mma_e4m3``.
        # Attested since contract v47 (see ``EXPERIMENTAL_LAUNCHES``).
        {"symbol": FUSED_WINDOW_DENSE_SYMBOL, "decoder": _DECODER_NATIVE_FUSED_WINDOW_DENSE_E4M3MMA,
         "regimes": _ALL_REGIMES, "modes": _ALL_MODES, "lane": "tessera_routed_fused_mma_e4m3",
         "structures": (STRUCTURE_DENSE,), "when_lane_absent": False},
        # The decode-once prefill lane (tessera#931, contract v56): default-off
        # (``TESSERA_E4M3_DECODE_ONCE=1``), resident only -- the route attaches
        # the decoded copy only to a resident module -- and taken for M at or
        # above ``e4m3_prefill.MIN_M`` in whichever phase M occurs; eager-only
        # (the route refuses the flag at load under a compiled forward).  No
        # extension lane: the decode runs the module's own Triton decoder and
        # the GEMM is torch's.  Experimental until a served census earns it a
        # cell (``EXPERIMENTAL_LAUNCHES``).
        {"symbol": DECODE_ONCE_DENSE_SYMBOL, "decoder": _DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3,
         "regimes": _ALL_REGIMES, "modes": ("resident",), "lane": None,
         "structures": (STRUCTURE_DENSE,), "when_lane_absent": False},
    ) + (
        # The compact window MoE adapter: routed experts served from the
        # loader's packed units, no decoded tile, on the epilogue arithmetic.
        # It is the expert half's ONLY launch.  The materialising
        # ``(MOE_GEMM_SYMBOL, _DECODER_TORCH_STOCK)`` entry that stood before
        # it left this table at contract v38 (tessera#604): it ran only on a
        # build that publishes no compact reader, and ``moe_route.
        # compact_window_lane`` answers True for this family whenever
        # ``parse_compact_tessera_expert_blob`` is defined, which it is in
        # this module.  A table row for a launch the build cannot make is the
        # defect v31 named for the dense routes.
        {"symbol": WINDOW_MOE_COMPACT_SYMBOL, "decoder": _DECODER_NATIVE_WINDOW_MOE_COMPACT,
         "regimes": _ALL_REGIMES, "modes": ("resident",), "lane": None,
         "structures": (STRUCTURE_ROUTED_MOE,), "when_lane_absent": False},
        # The fused warp-specialised lane (tessera#640) on the same epilogue
        # arithmetic.  ``PackedWindowMoeBundles.adapter`` takes it for every
        # stack ``routed_fused.fused_routed_window_supported`` admits (the
        # wire's one- or two-rate run table at rates 1..8 in the packer's
        # column order, window 14 -- every GLM q256 rung since contract v45,
        # tessera#694; v42-v44 read rate 4 alone) unless
        # ``TESSERA_ROUTED_FUSED=0``; the compact pair above stays the
        # dispatch for the rest (a box whose toolchain cannot build the
        # library, the opt-out).  The
        # FIRST lane-bearing row since #538: ``lane`` names the extension
        # ``native_extensions`` publishes for it, so a cell derives this pair
        # only at a rung the extension's own ``lane.requires`` admits
        # (``contract._lanes_a_rung_reaches``), and the compact row keeps
        # ``when_lane_absent`` False because it still runs beside the lane --
        # for the stacks the lane's runtime geometry check refuses and for
        # ``when_unavailable``.  A served route census of a rate-4 GLM stub
        # (contract v42) is what let the four window routed cells name it.
        {"symbol": ROUTED_FUSED_WINDOW_SYMBOL, "decoder": _DECODER_NATIVE_ROUTED_FUSED_WINDOW,
         "regimes": _ALL_REGIMES, "modes": ("resident",), "lane": "tessera_routed_fused_e4m3",
         "structures": (STRUCTURE_ROUTED_MOE,), "when_lane_absent": False},
        # The fused lane on the E4M3 instruction (``TESSERA_FUSED_E4M3_MMA=
        # e4m3``); see the dense row of the same library.  Attested since v47.
        {"symbol": ROUTED_FUSED_WINDOW_SYMBOL, "decoder": _DECODER_NATIVE_ROUTED_FUSED_WINDOW_E4M3MMA,
         "regimes": _ALL_REGIMES, "modes": ("resident",), "lane": "tessera_routed_fused_mma_e4m3",
         "structures": (STRUCTURE_ROUTED_MOE,), "when_lane_absent": False},
    ),
    # The dense half: same shape as the FP8 dense half above, and for the same
    # reason, with the row scale on the fp32 accumulator epilogue and
    # therefore its own family decoder.  The expert half (tessera#609) is
    # the compact window MoE adapter with the same epilogue, resident
    # like every expert stack.  It has no stock-kernel launch at all:
    # there is no materialising BF16 expert path to fall back to.  Both
    # halves of the route serve one arithmetic: raw bf16 values with
    # the fp32 row scale applied after the dot.
    TESSERA_BF16: _dense_native_window_launch(
        _DECODER_NATIVE_WINDOW_GEMM_BF16, _DECODER_NATIVE_FUSED_WINDOW_DENSE_BF16,
        "tessera_routed_fused_value") + (
        {"symbol": WINDOW_MOE_COMPACT_SYMBOL,
         "decoder": _DECODER_NATIVE_WINDOW_MOE_COMPACT_BF16,
         "regimes": _ALL_REGIMES, "modes": ("resident",), "lane": None,
         "structures": (STRUCTURE_ROUTED_MOE,), "when_lane_absent": False},
        # The fused lane's BF16 form (tessera#640); see the FP8 row.
        {"symbol": ROUTED_FUSED_WINDOW_SYMBOL,
         "decoder": _DECODER_NATIVE_ROUTED_FUSED_WINDOW_BF16,
         "regimes": _ALL_REGIMES, "modes": ("resident",), "lane": "tessera_routed_fused_value",
         "structures": (STRUCTURE_ROUTED_MOE,), "when_lane_absent": False},
    ),
}


#: Launches the DISPATCH can make that the packaged runtime contract does not
#: attest.  A launch here serves, and the routes' census expectation must know
#: it, but no ``lane_eligibility`` cell names it and no contract version was
#: promoted for it.  ``route_launches`` therefore leaves these out by default
#: -- the cell validator and every contract reader see exactly the attested
#: dispatch -- and the routes' ``census_expected`` opts in with
#: ``include_experimental=True`` so a served record is compared against what
#: the build can really launch.  A pair leaves this set when a receipt earns it
#: a cell.
#:
#: ``(WINDOW_GEMM_SYMBOL, _DECODER_NATIVE_WINDOW_GEMM)`` LEFT at contract v34
#: (#545): four censuses on the platform's own serve image recorded 112 of 112
#: dense modules on that pair in both regimes, both residencies, for
#: ``TESSERA_E4M3_K1`` at q256=1024 and ``TESSERA_BF16_K1`` at q256=1792, and
#: the four ``tessera_{e4m3_k1,bf16_k1}_dense_sm121_{decode,batch}`` cells name
#: it.  The removal and the cells are one change: ``contract.
#: _validate_cell_executes`` derives a cell's ``executes`` from
#: ``route_launches`` with ``include_experimental=False``, so a cell naming an
#: experimental pair is refused and a pair removed without its cells would put
#: an unattested launch in front of every contract reader.
#:
#: What stayed at v34, and why.  The two A4 pairs are deferred with the NVFP4
#: lane (#575) -- no dense or routed A4 census exists -- and the compact window
#: MoE adapter had no served receipt at all, in either arithmetic, until v38.
#:
#: ``(WINDOW_GEMM_SYMBOL, _DECODER_NATIVE_WINDOW_GEMM_FOLDED)`` ENTERED at
#: contract v37 (tessera#614).  The dense BF16 route moved from the epilogue
#: arithmetic to the folded one, so ``bf16_route.apply`` now makes a launch no
#: receipt covers: the v34 BF16 dense cells census'd the epilogue kernel.
#: Those two cells are withdrawn in the same change, for the reason the v34
#: note above gives in reverse -- ``_validate_cell_executes`` would otherwise
#: refuse them.  The epilogue pair stays attested for the E4M3 family, whose
#: arithmetic did not move.  A served census of the folded dense GEMM is what
#: earns the BF16 dense scope its cells back.
#:
#: THREE PAIRS LEFT at contract v38 (tessera#604): ``(WINDOW_GEMM_SYMBOL,
#: _DECODER_NATIVE_WINDOW_GEMM_FOLDED)``, ``(WINDOW_MOE_COMPACT_SYMBOL,
#: _DECODER_NATIVE_WINDOW_MOE_COMPACT)`` and ``(WINDOW_MOE_COMPACT_SYMBOL,
#: _DECODER_NATIVE_WINDOW_MOE_COMPACT_FOLDED)``.  One served route census of a
#: GLM-5.3-Flash stub (one dense and three MoE layers) on the GLM serving
#: image recorded all nine declared Tessera modules on their family's pair in
#: both regimes, eager, resident, ``problems: []``
#: (docs/measurements/tessera-glm-x-census-2026-09-26.md): dense E4M3 at
#: q256 832/1024/1088 on the epilogue GEMM, dense BF16 at 832/1024/1088 on the
#: folded GEMM, routed E4M3 at 896 on the compact adapter and routed BF16 at
#: 1024 on its folded form.  The eight ``*_sm121_{decode,batch}_resident``
#: cells on that image name them, in the same change, for the v34 reason.
#:
#: BOTH A4 PAIRS LEFT at contract v39 (tessera#604, second half):
#: ``(A4_DENSE_GEMM_SYMBOL, _DECODER_NATIVE_SPAN2_GEMM)`` and
#: ``(A4_GROUPED_GEMM_SYMBOL, _DECODER_NATIVE_SPAN2_GROUPED)``.  A served route
#: census of an eight-layer GLM-5.3-Flash stub with every Linear the plan
#: encodes on ``TESSERA_E2M1_K2`` at q256 896, on the GLM serving image, eager,
#: resident, recorded all sixteen dense modules (three dense MLPs and five
#: shared-expert blocks) on the span-2 GEMM and all five routed stacks on the
#: grouped form, in both regimes, ``problems: []``
#: (docs/measurements/tessera-glm-u1-census-2026-09-26.md).  The four
#: ``tessera_e2m1_k2_{dense,routed_moe}_sm121_{decode,batch}_resident`` cells
#: on that image name them, in the same change, for the v34 reason.  Nothing
#: was experimental after v39; the set stayed so the next unattested launch
#: would have a place to stand.
#:
#: TWO PAIRS PASSED THROUGH at contract v42 (tessera#640): ``(ROUTED_FUSED_
#: WINDOW_SYMBOL, _DECODER_NATIVE_ROUTED_FUSED_WINDOW)`` and ``(ROUTED_FUSED_
#: WINDOW_SYMBOL, _DECODER_NATIVE_ROUTED_FUSED_WINDOW_FOLDED)``.  The fused
#: warp-specialised routed kernel is a new launch identity the dispatch makes
#: by default for every rate-4 window-14 expert stack, so the routes'
#: ``census_expected`` had to admit it before any cell could name it, and the
#: two pairs stood here while the lane was built.  A served route census of
#: the rate-4 u1 stub B on the GLM serving image recorded both -- the E4M3
#: pair on its q256=1024 stack and the folded pair on its BF16 stack, both
#: regimes, ``problems: []`` (docs/measurements/2026-09-28-routed-fused-640.md)
#: -- and the four ``tessera_{e4m3,bf16}_k1_routed_moe_sm121_{decode,batch}_
#: resident`` cells name them in the same change, for the v34 reason.  The
#: compact pairs stay attested beside them: they serve every stack the lane
#: refuses and the ``TESSERA_ROUTED_FUSED=0`` opt-out.  The set is empty
#: again, kept so the next unattested launch has a place to stand.
#:
#: TWO MORE PAIRS PASSED THROUGH at contract v43 (the dense follow-up to
#: tessera#640): ``(FUSED_WINDOW_DENSE_SYMBOL, _DECODER_NATIVE_FUSED_WINDOW_
#: DENSE)`` and ``(FUSED_WINDOW_DENSE_SYMBOL, _DECODER_NATIVE_FUSED_WINDOW_
#: DENSE_FOLDED)``.  The fused window kernel's dense identity is the dispatch's
#: default for every rate-4 window-14 dense module whose rows are a multiple
#: of 128, so the routes' ``census_expected`` had to admit it before a cell
#: could name it.  A served route census of the rate-4 u1 stub B on the GLM
#: serving image recorded both -- the E4M3 pair on its q256=1024 shared-expert
#: modules and the folded pair on its q256=1024 BF16 shared gate/up module,
#: both regimes, ``problems: []``
#: (docs/measurements/2026-09-28-dense-fused-window.md) -- and the four
#: ``tessera_{e4m3,bf16}_k1_dense_sm121_{decode,batch}_resident`` cells name
#: them in the same change, for the v34 reason; the Triton pair stays attested
#: beside them (it serves every dense module the lane refuses and the
#: ``TESSERA_DENSE_FUSED=0`` opt-out).  The two ``tessera_e4m3_k1_dense_sm121_
#: {decode,batch}`` cells on the platform's pinned serve image, whose only
#: rung (1024) reaches the lane, name it on their OWN receipts: the v34 census
#: of ``qwen3-0.6b-uniform-R1024`` (112 modules, every one q256 1024) was run
#: again on that image with the lane as the dispatch, once per residency the
#: cells attest, and every module recorded the E4M3 pair in both regimes
#: (``tests/test_dense_fused_census_cells.py``).  Withdrawing them instead
#: would have moved ``versions.default_serve_image`` onto a build no registry
#: serves, for a lane that was never measured to be missing there.
#:
#: TWO PAIRS PASSED THROUGH at contract v47: the E4M3 family's fused
#: identities on its own tensor-core instruction, ``(ROUTED_FUSED_WINDOW_
#: SYMBOL, _DECODER_NATIVE_ROUTED_FUSED_WINDOW_E4M3MMA)`` and ``(FUSED_WINDOW_
#: DENSE_SYMBOL, _DECODER_NATIVE_FUSED_WINDOW_DENSE_E4M3MMA)`` (library
#: ``tessera_routed_fused_mma_e4m3``, ``mma.sync.m16n8k32.e4m3.e4m3.f32``),
#: which entered here at v46 and became the E4M3 family's default dispatch
#: (``TESSERA_FUSED_E4M3_MMA`` unset or ``e4m3``).  Leaving this set is per
#: pair and ``contract._validate_cell_executes`` has no image axis, so the
#: library's lane (``column_rates`` and ``column_rates_routed_moe`` 1..8)
#: puts both pairs in all six E4M3 cells at once, and each cell's image had
#: to be censused on the instruction: the v43 census of
#: ``qwen3-0.6b-uniform-R1024`` again on the platform's pinned serve image,
#: once per residency, every module on the E4M3-instruction dense pair in
#: both regimes, for the two ``tessera_e4m3_k1_dense_sm121_{decode,batch}``
#: cells (``tests/test_dense_fused_census_cells.py``); and the v45 census of
#: the u1 stub B again on the GLM serving image, every E4M3 module on its
#: family's E4M3-instruction pair in both regimes, for the four
#: ``tessera_e4m3_k1_{dense,routed_moe}_sm121_{decode,batch}_resident`` cells
#: (``tests/test_glm_u1_census_cells.py``).  The 16-bit library's pairs stay
#: attested beside them: ``TESSERA_FUSED_E4M3_MMA=f16`` still selects it.  The
#: set was empty again, kept so the next unattested launch has a place to stand.
#:
#: ONE PAIR ENTERED at contract v56: the E4M3 family's decode-once dense
#: prefill lane, ``(DECODE_ONCE_DENSE_SYMBOL, _DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3)``
#: (tessera#931), default-off.  It leaves when a served census of a T-8
#: projection artifact with ``TESSERA_E4M3_DECODE_ONCE=1`` records it.
#: Contract v59 replaces the folded BF16 arithmetic. All four new BF16
#: pairs remain experimental until a served census qualifies each pair.
EXPERIMENTAL_LAUNCHES: frozenset = frozenset({
    (DECODE_ONCE_DENSE_SYMBOL, _DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3),
    (WINDOW_GEMM_SYMBOL, _DECODER_NATIVE_WINDOW_GEMM_BF16),
    (FUSED_WINDOW_DENSE_SYMBOL, _DECODER_NATIVE_FUSED_WINDOW_DENSE_BF16),
    (WINDOW_MOE_COMPACT_SYMBOL, _DECODER_NATIVE_WINDOW_MOE_COMPACT_BF16),
    (ROUTED_FUSED_WINDOW_SYMBOL, _DECODER_NATIVE_ROUTED_FUSED_WINDOW_BF16),
})

#: Launches a compiled (``torch.compile``) forward cannot make: their owner
#: refuses them at load when vLLM's compilation mode is not NONE.  A census of a compiled serve therefore does
#: not expect them (``fp8_gemv.census_expected(compiled=True)``).  Since
#: contract v56: the decode-once dense prefill lane, whose M branch is host
#: Python (``native_window.PreparedDenseNativeModule.apply``).
EAGER_ONLY_LAUNCHES: frozenset = frozenset({
    (DECODE_ONCE_DENSE_SYMBOL, _DECODER_NATIVE_WINDOW_DECODE_ONCE_E4M3),
})


def route_launches(route: str, *, structure: str = STRUCTURE_DENSE,
                   regime: str | None = None, mode: str | None = None,
                   lanes: "tuple[str, ...] | None" = None,
                   include_experimental: bool = False) -> tuple[dict, ...]:
    """The launches ``route`` makes for a structure, narrowed by its conditions.

    ``structure`` defaults to dense for existing Linear callers. Other axes
    are optional and ``None`` means "not narrowed on this axis", so a call
    specifying only structure returns all launches that structure can make --
    the admissible set a census compares a record against. Narrowing all three
    is what a ``lane_eligibility`` cell does, and it is what makes the cell's
    ``executes`` a value rather than a disjunction.

    ``lanes`` is the set of extension lanes PREPARED.  ``()`` is a box with no
    extension at all -- the honest reading of ``when_unavailable`` -- and a
    non-empty set drops the ``when_lane_absent`` launch, exactly as the routes'
    own ``elif ... tessera_gemv is None`` branch does.

    There is deliberately no RATE axis. A rate decides whether a prepared
    lane can read a rung -- ``refuse_unreachable_lane``, represented here by
    ``lanes``. Dispatch alternatives come from ``ROUTE_LAUNCHES``, not from a
    universal GEMV/GEMM split: native dense preparation may serve either
    regime, and a regime describes problem shape rather than one kernel.
    Keep any within-regime M/rate decision in the owning lane. Adding a rate
    filter here would incorrectly erase alternatives from a regime's census.
    """
    if route not in ROUTE_LAUNCHES:
        raise ValueError(
            f"{route!r} is not a route this package serves ({sorted(ROUTE_LAUNCHES)}); a "
            "launch set for an unknown route would read as 'this route launches nothing'")
    if structure not in STRUCTURES:
        raise ValueError(
            f"{structure!r} is not a structure this package serves ({list(STRUCTURES)})")
    kept = []
    for launch in ROUTE_LAUNCHES[route]:
        if (not include_experimental
                and (launch["symbol"], launch["decoder"]) in EXPERIMENTAL_LAUNCHES):
            continue
        if structure not in launch["structures"]:
            continue
        if regime is not None and regime not in launch["regimes"]:
            continue
        if mode is not None and mode not in launch["modes"]:
            continue
        if lanes is not None and launch["lane"] is not None and launch["lane"] not in lanes:
            continue
        kept.append(launch)
    # The fallback is a fallback: where the caller SAID which lanes are
    # prepared and a lane launch survives every filter above, the launch that
    # runs only in its absence does not.  ``lanes=None`` is "not narrowed", so
    # it keeps both -- which is the admissive set a census compares against,
    # where a rate-1 unit and a box with no toolchain both legitimately fall
    # back inside a regime the lane otherwise owns.
    if lanes is not None and any(l["lane"] is not None for l in kept):
        kept = [l for l in kept if not l["when_lane_absent"]]
    return tuple(kept)


def launch_pairs(route: str, **narrow) -> set:
    """``{(symbol, decoder)}`` for :func:`route_launches` -- the census's shape."""
    return {(l["symbol"], l["decoder"]) for l in route_launches(route, **narrow)}


def experimental_launch_pairs(route: str, **narrow) -> set:
    """The pairs a route can report that no contract cell attests.

    How a native-lane owner reports its actual candidate: add these to the
    route's ``census_expected`` (or call ``launch_pairs(...,
    include_experimental=True)`` there) and a census accepts the candidate
    pair, while ``launch_pairs``' default view keeps the cell validator on the
    attested dispatch.  Empty for a route with no experimental lane.
    """
    return (launch_pairs(route, include_experimental=True, **narrow)
            - launch_pairs(route, **narrow))


def moe_census_symbol_base(symbol: str) -> str:
    """A routed launch entry point without the runtime-selected backend suffix.

    Keep exact symbols in receipts. Only comparison removes the suffix; its
    dependency-free home lets receipt replay run without importing torch or
    the runtime route implementation.
    """
    return str(symbol).split(":", 1)[0]


_BODIES = ("TCQ", "WINDOW")
_REQUIRED = ("family", "grid", "body", "plane", "q256", "rows", "columns", "wire_bytes", "roles")
GROUP_SIZE = 16

#: WHAT A FUSED MODULE'S MEMBERS MUST SHARE, AND WHAT IS FREE PER MEMBER (#37).
#:
#: vLLM merges q/k/v and gate/up into one Linear and builds ONE quant method per
#: module, so everything that selects a method or a tile -- the family, and with
#: it the grid, body and scale plane the route decodes, plus the input width the
#: mainloop reads -- is a module fact.  The RATE is not one of them.  Every
#: decoder here is fed from each member's OWN parsed manifest: the FP8 and BF16
#: routes call ``prepare_window(unit.body_bits, unit.rates, unit.window_bits,
#: unit.window_codes, ...)`` per role and concatenate, and the NVFP4 route packs
#: each role's own ``rate``/``arity``/``memory``/``half`` scalars and decodes
#: into that role's row slice.  A module of three roles at three rungs decodes
#: element-for-element to what the three roles decode alone
#: (``experiments/fused_member_rung_identity.py`` on a real Qwen3-0.6B q/k/v,
#: ``tests/test_fused_member_rungs.py``).
#:
#: This dict is the value ``runtime_contract.json``'s ``fused_module.fields``
#: block is checked against, exactly as ``sharding.ROUTE_TP_AXES`` is what
#: ``tensor_parallel.units[].loader_axes`` is checked against: a producer's
#: group allocator learns the constraint from the table this runtime publishes
#: instead of carrying a local ban, and the table cannot drift from the code.
#:
#: ``grid`` is listed shared and not per-member on purpose.  A route may hold
#: more than one grid (``TESSERA_NVFP4`` holds ``E2M1`` and ``E2M1x2``), and
#: nothing has ever decoded a module that mixed them; the sidecar carries one
#: grid per module and ``parse_tessera_blob_for_scheme`` refuses a member that
#: disagrees with it.  Unattested is not "probably fine".
FUSED_MODULE_FIELDS: dict[str, str] = {
    "family": "shared",
    "structure": "shared",
    "grid": "shared",
    "body": "shared",
    "plane": "shared",
    "columns": "shared",
    "q256": "per_member",
    "rows": "per_member",
}
FUSED_MODULE_SCHEMA = "tessera.fused-module.v1"
#: How a mixed-rung module is SPELLED in the sidecar.  ``q256`` is the module's
#: rung when every role carries it -- the spelling of every checkpoint written
#: before #37, unchanged -- or a list of one rung per role in ``roles`` order
#: when they differ.  One field, one fact: a second field beside it could
#: disagree with it, and a reader would then have two answers about one module.
#: A plugin build older than this one refuses the list form on sight
#: (``q256 must be an integer``), which is the fail-closed direction.
FUSED_Q256_SPELLING = "int_or_per_role_list"
#: The container framing the module's blob is (``tessera.fused``).
FUSED_CONTAINER = "TSRFUSE1"


def is_tessera_scheme(scheme: Any) -> bool:
    return isinstance(scheme, Mapping) and scheme.get(TESSERA_SCHEME_KEY) in TESSERA_FAMILIES


def _as_int(scheme: Mapping, field: str, target: str) -> int:
    return _positive_int(scheme.get(field), field, target)


def _positive_int(value, field: str, target: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"tessera target {target!r}: {field} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"tessera target {target!r}: {field} must be positive, got {value}")
    return value


def _rungs_for_roles(scheme: Mapping, roles, target: str) -> "list[int]":
    """The rung of every role, read from the scheme's one ``q256`` field.

    An int is the module's rung, carried by every role -- the spelling of every
    checkpoint written before #37 and the one the exporter still writes for a
    uniform group.  A list is one rung per role in ``roles`` order, which is
    what a fused group whose members took different rates looks like: the rate
    is a per-member fact here because the DECODE is (see
    ``FUSED_MODULE_FIELDS``).  Length must equal the role count exactly -- a
    list that does not line up with the roles it indexes is a wrong tensor
    waiting to happen, not something to pad or truncate.
    """
    if isinstance(scheme.get("q256"), (list, tuple)):
        declared = list(scheme["q256"])
        if len(declared) != len(roles):
            raise ValueError(
                f"tessera target {target!r}: q256 is a per-role list of {len(declared)} rungs but "
                f"the scheme declares {len(roles)} role(s) {[r[0] for r in roles]}; a per-role "
                "rate list is read positionally against roles and must be the same length")
        return [_positive_int(v, f"q256[{i}]", target) for i, v in enumerate(declared)]
    return [_as_int(scheme, "q256", target)] * len(roles)


def _expert_role_rungs(group: Mapping, experts: int, roles, family: str,
                       target: str) -> "list[list[int]]":
    """The rung of every (expert, role) of a routed group, as ``[E][roles]``.

    The group's one ``q256`` field reads three ways.  An int is the whole
    stack's rung -- every checkpoint through #966.  A flat list is one rung
    per role, uniform across the expert axis -- the pre-#967 mixed-RUNG
    spelling, unchanged.  A list of lists is the per-unit spelling (#967): one
    row per expert in expert order, each a rung per role in the group's row
    order. The FP8 and BF16 owners can store different rates across experts.
    The native E2M1 owner requires one run pair per projection.
    A rate matrix must cover every expert and role without a gap.
    """
    declared = group.get("q256")
    arity = len(roles)
    if not isinstance(declared, (list, tuple)):
        rungs = [_as_int(group, "q256", target)] * arity
        return [list(rungs) for _ in range(experts)]
    if declared and all(isinstance(row, (list, tuple)) for row in declared):
        if len(declared) != experts or any(len(row) != arity for row in declared):
            raise ValueError(
                f"tessera target {target!r}: q256 is an expert-major matrix of "
                f"{[len(row) for row in declared]} rung(s) per expert, the scheme declares "
                f"{experts} expert(s) x {arity} role(s) {[r[0] for r in roles]}; an "
                "expert-role matrix is read as [experts][roles] and must cover every "
                "expert and role exactly")
        matrix = [[_positive_int(v, f"q256[{e}][{j}]", target)
                   for j, v in enumerate(row)] for e, row in enumerate(declared)]
        if family == TESSERA_NVFP4:
            reason = e2m1_expert_rate_reason(matrix)
            if reason is not None:
                raise ValueError(f"tessera target {target!r}: {reason}")
        return matrix
    if len(declared) != arity:
        raise ValueError(
            f"tessera target {target!r}: q256 is a flat list of {len(declared)} rung(s) for a "
            f"group of {arity} role(s) {[r[0] for r in roles]}; a routed group's q256 is an "
            f"integer, a per-role list of {arity}, or an expert-major [{experts}][{arity}] "
            "matrix of one rung per expert and role")
    rungs = [_positive_int(v, f"q256[{j}]", target) for j, v in enumerate(declared)]
    return [list(rungs) for _ in range(experts)]


def _refuse_an_unreadable_rung(route: str, grid: str, q256: int, target: str) -> None:
    """Refuse a rung outside this route and grid's declared reader range.

    Reader support does not attest a D41 cell or a served checkpoint.
    The native E2M1 reader accepts paired WINDOW rates 1 through 8.
    It does not accept the old TCQ serving body.
    """
    from .contract import reader_accepts, reader_rate_grid

    found = reader_rate_grid(route, grid)
    if found is None:
        raise ValueError(
            f"tessera target {target!r}: this build publishes no decodable rate range for "
            f"{route} on grid {grid!r}, so there is no rung it can promise to read. "
            "runtime_contract.json describes the (route, grid) pairs this plugin serves; a pair "
            "it does not describe is refused rather than served on another pair's numbers.")
    family, low, high, step = found
    if not reader_accepts(q256, low, high, step):
        span = f"[{low}, {high}]" + (f" step {step}" if step != 1 else " (every integer)")
        raise ValueError(
            f"tessera target {target!r}: q256={q256} is outside the rungs this build's decoder "
            f"reads for {family} -- {span}. Serving it would decode bytes the reader does not "
            "promise to understand, which is a wrong tensor rather than a refusal. Re-export at "
            "a rung inside the range, or publish a measured range that contains this one in "
            "runtime_contract.json.")


def route_for_grid(grid: str, family: "str | None" = None) -> "str | None":
    """The route in THIS build that holds ``grid``, or ``None``.

    Derived from ``ROUTES`` rather than written a second time: a grid reaches a
    route because that route lists it, and a build that gains a route gains its
    grids here with no edit.  ``BF16`` maps to ``TESSERA_BF16`` since issue #9
    (contract v5: the ``TESSERA_BF16_K1`` row, reader range [256, 4096],
    attested rung 1792); the export gate (``refuse_unserveable_wire``) resolves
    through this mapping.  ``None`` is the honest answer for a grid Tessera
    can ENCODE and this plugin has no decoder for -- a grid no ``ROUTES``
    entry lists.

    A grid held by more than one route is REFUSED, naming every holder: with
    two answers the grid-only question is ambiguous, and returning the first
    holder in dict order would let table order silently own whichever gate
    resolved through it (#51).  A caller that knows which route it is writing
    passes ``family`` alongside the grid -- the route is then checked to hold
    the grid and returned -- so the ambiguity is answered by the question, not
    by the table's order.
    """
    if family is not None:
        route = ROUTES.get(family)
        if route is None:
            raise ValueError(
                f"unknown family {family!r}; this build serves {TESSERA_FAMILIES}")
        if grid not in route["grids"]:
            raise ValueError(
                f"{family} holds {route['grid_kind']} grid {route['grids']}, got {grid!r}; "
                "a wire is gated against the route that will decode it, not another "
                "route that happens to read the same alphabet")
        return family
    holders = [fam for fam, route in ROUTES.items() if grid in route["grids"]]
    if len(holders) > 1:
        raise ValueError(
            f"grid {grid!r} is held by more than one route {holders}; the grid alone "
            "does not say which route's range gates this wire. Pass family=... naming "
            "the route being written.")
    return holders[0] if holders else None


def refuse_a_family_with_no_expert_route(route: str, target: str) -> None:
    """Refuse a routed-MoE stack on a route ``MOE_BUILDERS`` has no builder for.

    ONE HOME for a rule three call sites state: the plugin at load
    (``moe_route.build_tessera_moe_method``), the exporter at plan time
    (``plan_expert_stack``) and the export gate (``refuse_unserveable_wire``
    with ``structure="routed_moe"``), which asks it before it asks what the
    cells attest -- a route the build cannot dispatch is refused for that
    reason, not for the absence of a cell that could never exist.
    """
    if route in MOE_BUILDERS:
        return
    raise ValueError(
        f"tessera target {target!r}: {route} has no expert route in this build "
        f"(scheme.MOE_BUILDERS names {sorted(MOE_BUILDERS)}). Plain source BF16 "
        "passthrough is separate and uses quantization_config.ignore. An expert "
        "stack is "
        "refused rather than decoded through another family's tile: plan it on a family "
        "with a route, or leave it out to pass it through as BF16.")


def attested_cells(family: str, structure: str,
                   contract: "Mapping | None" = None, *,
                   device_backed: bool = True) -> "list[dict]":
    """The ``lane_eligibility`` cells that attest ``(family, structure)``.

    A cell is the unit of attestation (``contract`` module docstring): it
    names the platform, family, STRUCTURE, regime, residency and image a
    container receipt covered, and the rungs it covered them at.  A gate
    asking what this build has served for a routed-MoE stack reads the cells
    of that structure and nothing else -- the format row's
    ``reader_rate_range_q256`` is the dense route's reader, and the routed
    route runs a different consuming kernel (#135).  Returns the cells in
    contract order, ``[]`` when none attests the pair; the caller decides
    whether an empty list is a refusal.  A cell that states a ``predicates``
    narrowing is REFUSED rather than returned: this filter reads family and
    structure, which cannot resolve a predicate, and a narrowed cell read as
    unconditional is the failure the grammar exists to prevent
    (``contract.refuse_unevaluated_predicates``).

    THE CELL'S OWN QUALIFICATION IS PART OF THE SELECTION.  A cell may state the
    weaker fact -- ``qualification: compile_only`` beside ``route_status:
    unbacked``, the one combination v10 (#456) permits for a compile receipt --
    and such a cell proves a toolchain fact and never a serve.  ``device_backed``
    (True by default) returns only the cells a DEVICE backed
    (``contract.cell_is_device_backed``), which is what both callers mean by
    "attests": the export gate that admits a stack's rungs, and the manifest's
    ``attested_by`` record of the cells that admitted them.  With it False the
    cells that state the pair WITHOUT a serve are returned instead, which is what
    a refusal names so that "no cell" and "a cell that is not a receipt" do not
    read as the same sentence.  Either way a cell whose qualification or
    ``route_status`` this build cannot read is refused by name rather than
    counted on one side of a question the document did not answer.
    """
    from .contract import (cell_is_device_backed, load_serving_contract,
                           refuse_unevaluated_predicates)

    payload = load_serving_contract() if contract is None else contract
    cells = [cell for cell in payload["lane_eligibility"]["cells"]
             if cell["family"] == family and cell["structure"] == structure]
    for cell in cells:
        refuse_unevaluated_predicates(cell, f"lane_eligibility cell {cell.get('id')}")
    wanted = bool(device_backed)
    return [cell for cell in cells
            if cell_is_device_backed(cell, f"lane_eligibility cell {cell.get('id')}") == wanted]


def refuse_unserveable_wire(grid: str, q256: int, body: str, plane: str,
                            *, family: "str | None" = None,
                            span: "int | None" = None, target: str,
                            structure: str = STRUCTURE_DENSE,
                            contract: "Mapping | None" = None) -> str:
    """Refuse bytes outside the published serving reader contract.

    ``family`` identifies the consuming route. ``structure`` selects dense
    or routed ownership. Native E2M1 uses one paired WINDOW reader for both.
    Its reader range states byte support, not D41 or serving qualification.
    The FP8 and BF16 routes retain their existing structure-specific bounds.
    Research encoding remains outside this serving check.
    """
    from .contract import reader_accepts, reader_rate_grid

    if structure not in STRUCTURES:
        raise ValueError(
            f"tessera export {target!r}: structure {structure!r} is not one this build "
            f"dispatches ({STRUCTURES}); there is no route for a wire served as it.")

    q256 = int(q256)
    still_legal = (
        "The rung is still legal to ENCODE -- this refusal is about what THIS plugin build can "
        "decode, not about the wire -- so a research or measurement artifact at this rung is "
        "unaffected; it just cannot be served by this build.")
    route = route_for_grid(grid, family)
    if route is None:
        held = ", ".join(f"{f} holds {r['grids']}" for f, r in ROUTES.items())
        raise ValueError(
            f"tessera export {target!r}: no route in this plugin build holds grid {grid!r}, so "
            f"there is nothing for a q256={q256} wire on it to be served by ({held}). "
            + still_legal)
    found = reader_rate_grid(route, grid, contract)
    if found is None:
        raise ValueError(
            f"tessera export {target!r}: {route} holds grid {grid!r}, but runtime_contract.json "
            f"publishes no decodable rate range for the pair ({route}, {grid!r}) -- so this build "
            f"promises no rung it can read there, and a q256={q256} wire would be refused at "
            f"load. A range is published when a rung has been taken through the decoder and "
            f"measured, never asserted here. " + still_legal)
    family, low, high, step = found
    if structure == STRUCTURE_DENSE or route == TESSERA_NVFP4:
        # The native E2M1 library has the same byte reader for dense and routed
        # owners. Admit its declared reader range without manufacturing a
        # serving attestation; D41 allocator eligibility remains separate.
        if structure == STRUCTURE_ROUTED_MOE:
            refuse_a_family_with_no_expert_route(route, target)
        if not reader_accepts(q256, low, high, step):
            span_text = f"[{low}, {high}]" + (f" step {step}" if step != 1 else " (every integer)")
            raise ValueError(
                f"tessera export {target!r}: q256={q256} on grid {grid!r} is outside the rungs "
                f"this build's decoder reads for {family} -- runtime_contract.json publishes "
                f"{span_text}. Writing it would produce a checkpoint {route} refuses at load. "
                f"Re-export inside the range, or publish a measured range that contains this "
                f"rung. " + still_legal)
    else:
        refuse_a_family_with_no_expert_route(route, target)
        # THE FORMAT ROW IS THE DENSE READER'S RANGE.  A non-dense structure
        # is attested only where a cell of that structure exists, so its
        # bound is the union of the rungs those cells cover (their census
        # rungs and, since lane schema v11, the allowable rungs of their run
        # tables; ``contract.cell_covers_rung``) -- which the contract
        # validator already keeps inside the row's range, so this is the
        # tighter of the two and the only one that names the right kernel.
        # A cell that attests a toolchain is not one of these: see
        # ``attested_cells`` on the qualification, and the branch below, which
        # names those cells rather than reporting the pair as unattested.
        cells = attested_cells(family, structure, contract)
        if not cells:
            compiled_only = attested_cells(family, structure, contract, device_backed=False)
            if compiled_only:
                raise ValueError(
                    f"tessera export {target!r}: the {structure} cells of runtime_contract.json "
                    f"for {family} state the pair and no serve -- "
                    + "; ".join(f"{cell['id']} is {cell['qualification']}/{cell['route_status']}"
                                for cell in compiled_only)
                    + f". A cell attests a serve only when a device ran the route, so this "
                    f"build promises no rung for a {structure} stack on grid {grid!r}; the "
                    "compile receipt is real evidence about the toolchain and no evidence "
                    "that these bytes were ever served. " + still_legal)
            raise ValueError(
                f"tessera export {target!r}: no lane_eligibility cell in runtime_contract.json "
                f"attests {family} served as structure {structure!r}, so this build promises "
                f"no rung for a {structure} stack on grid {grid!r}. A cell is published when a "
                f"container receipt covers the structure on a runtime image; absence is "
                f"'unattested', which an export that declares the structure cannot ship on. "
                + still_legal)
        # Since lane schema v11 a cell covers its census rungs AND every rung
        # its family's allowable_rungs rule admits in one of its run tables
        # (tessera#750): ``contract.cell_covers_rung``, the one predicate.
        from .contract import cell_covers_rung, format_entry

        entry = format_entry(family, contract)
        if not any(cell_covers_rung(cell, q256, entry) for cell in cells):
            rungs = sorted({int(r) for cell in cells for r in cell["rungs_q256"]})
            per_cell = "; ".join(
                f"{cell['id']} attests {sorted(cell['rungs_q256'])}"
                + (f" and run tables {cell['run_tables']}" if cell.get("run_tables") else "")
                for cell in cells)
            raise ValueError(
                f"tessera export {target!r}: q256={q256} on grid {grid!r} is outside the rungs "
                f"the {structure} cells of runtime_contract.json attest for {family} "
                f"({rungs}: {per_cell}). The format row's reader range is the dense route's; "
                f"a {structure} stack is served by a different consuming kernel and is "
                f"attested only at the rungs its own cells cover: their census rungs, and the "
                f"allowable rungs of their run tables. Re-plan the stack on a covered rung, "
                f"or serve the rung and publish the cell. " + still_legal)
    expected_plane = ROUTES[route]["plane"]
    if plane != expected_plane:
        raise ValueError(
            f"tessera export {target!r}: {route} decodes the {expected_plane} scale plane to its "
            f"{ROUTES[route]['tile']} tile; this wire carries the {plane!r} plane, which has no "
            f"{ROUTES[route]['short']} tile. " + still_legal)
    expected_body, expected_span = ROUTES[route]["body"], ROUTES[route]["span"]
    if body != expected_body or (span is not None and int(span) != expected_span):
        carries = f"{body} span {span}" if span is not None else str(body)
        raise ValueError(
            f"tessera export {target!r}: {route} decodes the span-{expected_span} "
            f"{expected_body} body; grid {grid!r} at q256={q256} resolves to {carries}, which "
            f"this build has no in-forward decoder for. " + still_legal)
    return route


def lane_rate_report(lane: str, rates, contract: "Mapping | None" = None) -> dict:
    """Which of ``rates`` this lane can read, as a value a gate can read.

    ``{"lane", "supported", "rates", "offending", "reachable"}``.  Used on
    both sides of the seam: at plan time over the rate SET a rung implies
    (:func:`refuse_unreachable_lane`), and after the fact over the rates a
    parsed unit actually carries (``tools/tessera_lane_preflight.py``), so a
    producer and an auditor answer the question with one function.
    """
    from .contract import lane_requirements

    supported = tuple(int(r) for r in lane_requirements(lane, contract).get("column_rates", ()))
    seen = tuple(sorted({int(r) for r in rates}))
    offending = tuple(r for r in seen if supported and r not in supported)
    return {"lane": lane, "supported": list(supported), "rates": list(seen),
            "offending": list(offending), "reachable": not offending}


#: How each PUBLISHED lane requirement is decided against ONE parsed unit's
#: facts: the requirement's own name -> (the fact it reads, how it is read).
#: A requirement this table has not learned is REFUSED rather than skipped,
#: which is the whole of #206 -- an offline checker that quietly evaluated the
#: subset it knew published ``READABLE`` for wire its own loader rejects, and
#: was therefore weaker than the plan-time gate it exists to double-check.
#:
#: This table and :func:`decide_lane_requirements` below are ONE HOME (#264):
#: the byte-time report (:func:`lane_wire_report`), the plan-time gate
#: (:func:`refuse_unreachable_lane`), the loader
#: (``kernel_window_gemv.prepare_from_parsed``) and the route-side gate
#: (``bf16_route.gemv_refusal_for_unit``) all decide a unit through the same
#: function over the same published requirements.  Before that, the contract
#: published four conditions while the loader refused nine, so a TP row shard
#: -- ``layout.slice_unit`` stamps ``initial_state`` -- read READABLE at
#: preflight and fell back module by module at load.
_LANE_WIRE_CHECKS = {
    "column_rates": ("rates", "every_in"),
    # Structure-scoped (tessera#694): the rates the lane's ROUTED-EXPERT launch
    # reaches on the target, decided over ``facts["rates"]`` only when
    # ``facts["structure"]`` is the routed-MoE structure; a caller that did
    # not state the structure is refused by name, and any other structure is
    # decided as not binding.  The fused window lane reads every rate of the
    # wire (``column_rates``) on its one-table dense launch, but its two-table
    # gate/up launch holds fewer word-stage words in sm_121's opt-in shared
    # memory, and the value library's dense launch reaches rates (9..14,
    # tessera#750 item 4) its routed launches are not built for, so a stack
    # above this set keeps the compact adapter.
    "column_rates_routed_moe": ("rates", "every_in_routed_moe"),
    "window_bits": ("window_bits", "one_of"),
    "body": ("body", "wire_spelling"),
    "plane": ("plane", "wire_spelling"),
    "release_overrides": ("release_overrides", "carries"),
    "diagonals": ("diagonals", "carries"),
    "start_state": ("start_state", "carries"),
    "rotation": ("rotation", "state_in"),
    "grid_arities": ("grid_arity", "one_of"),
}

#: What a ``carries``-mode refusal says, per requirement.  ``{n}`` is the
#: fact's count where the fact carries one (RELEASE overrides), else elided.
_CARRIES_REFUSALS = {
    "release_overrides": (
        "the unit carries {n}RELEASE override(s), and the lane decodes straight "
        "off the wire with no override read"),
    "diagonals": (
        "the unit carries segment-2a diagonals, and the lane applies none"),
    "start_state": (
        "the unit carries a shard start state (INITIAL_STATE, layout.slice_unit), "
        "and the lane supplies state_{{-1}} = 0 itself -- the L-bit pad that opens "
        "a wire column is not stored, so a shard packed against the pinned start "
        "would decode its first rows to plausible wrong weights; a lane that "
        "threads the state (tessera.kernel_window) or the materialised path "
        "serves this shard"),
}


def decide_lane_requirements(lane: str, requires: Mapping[str, Any],
                             facts: Mapping[str, Any]) -> "list[str]":
    """Refusals, by name, for ONE unit's facts against a lane's requirements.

    THE HOME of the lane-readability decision (#264).  ``requires`` is the
    published predicate (``native_extensions[].lane.requires``, or the
    build's own copy ``ext.WINDOW_GEMV_LANE["requires"]`` -- the contract
    validator pins the two equal); ``facts`` is one unit's wire facts in
    :func:`wire_facts_of_parsed`'s vocabulary.  Every caller that answers
    "can this lane read this unit" calls this: the byte-time and plan-time
    gates, the loader and the bf16 route's gate, so the published predicate
    and the loader agree by construction rather than by three restatements.

    A requirement this function has not learned RAISES rather than being
    skipped (#206), and a fact the caller did not supply is a refusal, never
    a pass -- absent evidence is not evidence of absence.
    """
    from .contract import route_wire_spelling

    unknown = sorted(set(requires) - set(_LANE_WIRE_CHECKS))
    if unknown:
        raise ValueError(
            f"lane {lane!r} publishes requirement(s) {unknown} that this reader cannot "
            f"decide (it decides {sorted(_LANE_WIRE_CHECKS)}). A checker that skipped a "
            "published condition would call a unit readable that the loader refuses, so "
            "it refuses to answer instead. Teach scheme._LANE_WIRE_CHECKS to read the "
            "fact, and give it a case in tests/test_lane_reachability.py.")

    refusals: "list[str]" = []
    for name in sorted(requires):
        fact_name, how = _LANE_WIRE_CHECKS[name]
        if how == "every_in_routed_moe":
            structure = facts.get("structure")
            if structure is None:
                refusals.append(
                    f"the unit's structure was not read, so the lane's {name} requirement "
                    f"({requires[name]!r}) cannot be decided; absent evidence is not a pass")
                continue
            if structure not in STRUCTURES:
                refusals.append(
                    f"the unit's structure {structure!r} is not one this build serves "
                    f"({list(STRUCTURES)}), so the lane's {name} requirement cannot be decided")
                continue
            if structure != STRUCTURE_ROUTED_MOE:
                continue                     # decided: the requirement binds expert stacks only
            how = "every_in"
        value = facts.get(fact_name)
        if value is None or (how == "every_in" and not tuple(value)):
            refusals.append(
                f"the unit's {fact_name} was not read, so the lane's {name} requirement "
                f"({requires[name]!r}) cannot be decided; absent evidence is not a pass")
            continue
        if how == "every_in":
            supported = [int(r) for r in requires[name]]
            offending = sorted({int(r) for r in value} - set(supported))
            if offending and name == "column_rates_routed_moe":
                refusals.append(
                    f"{name} {offending} are outside the rates this lane's routed-expert "
                    f"launches reach ({supported}); the lane reads the wire at these rates "
                    "but its routed launches do not reach them on the target (the gate/up "
                    "launch's two tables and word stages exceed the target's shared memory "
                    "above that set), so the stack keeps the compact adapter")
            elif offending:
                refusals.append(
                    f"{name} {offending} are outside the rates this lane reads "
                    f"({supported}); the lane repacks each column at its own rate, "
                    "so one column out of range refuses the whole unit")
        elif how == "one_of":
            allowed = [int(v) for v in requires[name]]
            if int(value) not in allowed:
                refusals.append(
                    f"{name} {int(value)} is outside the {allowed} this lane reads")
        elif how == "carries":
            if bool(value) != bool(requires[name]):
                if bool(requires[name]):
                    refusals.append(
                        f"{name}: the lane reads only units that carry this, "
                        "and the unit does not")
                else:
                    count = "" if isinstance(value, bool) else f"{int(value)} "
                    refusals.append(
                        f"{name}: " + _CARRIES_REFUSALS[name].format(n=count))
        elif how == "state_in":
            allowed = [route_wire_spelling("rotation", v) for v in requires[name]]
            if str(value) not in allowed:
                refusals.append(
                    f"{name} {value} is not among the {allowed} states this lane reads; "
                    "the kernel decodes the wire without applying a rotation")
        else:
            wanted = route_wire_spelling(name, requires[name])
            if str(value) != wanted:
                refusals.append(
                    f"{name} {value} is not the {wanted} {name} this lane reads")
    return refusals


def wire_facts_of_parsed(parsed) -> dict:
    """The byte-side facts a lane predicate is decided against, off a parsed unit.

    Read from ``unit_artifact.parse_unit_artifact``'s own unit -- the object
    the load path gates on -- and named the way :func:`lane_wire_report` and
    :func:`refuse_unreachable_lane` name them, so an auditor holding bytes and
    a producer holding a plan hand the same four facts to the same rules.

    Duck-typed rather than imported: this module is torch-free on purpose
    (``tests/test_census_no_torch.py``) and the manifest enums are not. A fact
    the unit does not carry comes back ``None``, which the report refuses by
    name rather than defaulting -- absent evidence is not a passing check.

    Three facts read a MISSING attribute as its whole-unit value rather than
    as absent evidence, because that is exactly how the loader reads it and
    the loader is what this predicate must agree with (#264): a plain
    ``EncodedUnit`` has no ``initial_state`` at all (only ``layout.SlicedUnit``
    does), ``diagonals`` is ``None`` on every CHANNEL-plane unit, and a unit
    written before the rotation field decodes as ``RotationState.NONE`` --
    ``prepare_from_parsed`` spells all three with the same ``getattr``
    defaults.  ``grid_arity`` has no such default: it lives on the PARSED
    wrapper, not the unit, so a caller holding only a bare unit gets ``None``
    and the report refuses to decide it.
    """
    unit = getattr(parsed, "unit", parsed)
    body = getattr(unit, "body", None)
    plane = getattr(unit, "scale_plane", None)
    release = getattr(unit, "release_index", None)
    rotation = getattr(unit, "rotation", None)
    grid = getattr(parsed, "grid", None)
    arity = int(getattr(grid, "arity", 0)) if grid is not None else 0
    return {
        "rates": tuple(int(r) for r in (getattr(unit, "rates", ()) or ())),
        "window_bits": getattr(unit, "window_bits", None),
        "body": getattr(body, "name", body),
        "plane": getattr(plane, "name", plane),
        # The COUNT, not a flag: a refusal that says how many overrides the
        # unit carries is a value; ``None`` when the plane was never read.
        "release_overrides": None if release is None else int(release.numel()),
        "diagonals": getattr(unit, "diagonals", None) is not None,
        "start_state": getattr(unit, "initial_state", None) is not None,
        # The state's NAME (``RotationState.name``); an unrecognisable value
        # -- neither an enum nor absent -- reads ``None`` and is refused.
        "rotation": "NONE" if rotation is None else getattr(rotation, "name", None),
        "grid_arity": arity if arity > 0 else None,
    }


def lane_wire_report(lane: str, facts: Mapping[str, Any],
                     contract: "Mapping | None" = None) -> dict:
    """Whether ``lane`` can read ONE unit, against EVERY requirement it publishes.

    The byte-side twin of :func:`refuse_unreachable_lane`, and the reason the
    two are next to each other: the plan-time gate reads a rung, this reads
    the bytes that rung produced, and a wire checker that decided fewer
    conditions than the plan checker cannot double-check it. ``tools/
    tessera_lane_preflight.py`` reported ``READABLE`` for a unit at rate 4
    whose window was 12 and whose plane was LUT, because it kept the rates off
    the parse and dropped everything else (#206).

    ``{"lane", "requirements", "facts", "refusals", "readable"}``. The
    requirement roster is the contract's, not a list here, and the decision
    is :func:`decide_lane_requirements` -- the one home every gate shares
    (#264): an unknown requirement RAISES, so a lane predicate that grows a
    condition refuses every unit until the core learns to decide it.
    """
    from .contract import lane_requirements

    requires = lane_requirements(lane, contract)
    refusals = decide_lane_requirements(lane, requires, facts)
    return {"lane": lane, "requirements": dict(requires), "facts": dict(facts),
            "refusals": refusals, "readable": not refusals}


def refuse_unreachable_lane(lane: str, *, grid: str, q256: int, rate_cap: int,
                            body: str, plane: str, window_bits: int,
                            target: str, release_overrides: int = 0,
                            diagonals: bool = False, rotation: str = "NONE",
                            start_state: bool = False,
                            grid_arity: int = 1,
                            structure: "str | None" = None) -> tuple[int, ...]:
    """Refuse, AT PLAN TIME, a rung whose columns ``lane`` could never read.

    THE SEAM MOVES ONE STAGE EARLIER, and that is the whole point of #104.
    :func:`refuse_unserveable_wire` asks whether the ROUTE publishes a decode
    for these bytes; this asks whether a named LANE INSIDE that route can read
    them.  The two are different questions and the second one had no producer
    side at all: ``kernel_window_gemv.repack_window_body`` raises per unit at
    LOAD, the streamed FP8 route catches it and serves the same bytes through
    the torch window decode, and the module reports itself served.  So a
    checkpoint built to exercise the GEMV lane exercised nothing, 112 modules
    at a time, and the census that measured it recorded one route and an empty
    problem list -- an experiment that compared a thing against itself and
    reported agreement.

    A rung is a ROOT rate: ``grammar.bresenham_rate_schedule`` realises it by
    mixing the two rates bracketing it, so the rung determines the rate SET
    without the column count (``grammar.rate_set``) and this gate can run
    before a single shape is read -- before hours of encoding, which is the
    difference between a refusal and a bill.

    The rate set here is derived PER WEIGHT (``root_from_q256``), while
    ``contract._validate_cell_executes`` derives a cell's set PER CODE (rung x
    arity / 256) -- the encoder's own spelling (``export.encode_linear_planes``
    writes ``q256 * grid.arity``).  The two agree on every arity-1 grid, which
    is every grid a lane exists for today, so the disagreement is latent; the
    decision core is still one home (``decide_lane_requirements``), and if a
    lane ever grows on an arity-2 route the derivation to change is this one.

    Every bound comes from the packaged contract (``lane_requirements``),
    never from a constant here: the day the kernel grows a 6-byte lane, the
    contract changes and this follows it.  The DECISION is
    :func:`decide_lane_requirements`, the same core the byte-time report and
    the loader run (#264), so a requirement the contract grows and this gate
    has not learned refuses the plan rather than being skipped -- the plan
    side of #206's rule.  Returns the accepted rate set.

    The keyword facts past ``target`` state what the PLAN will write, and
    their defaults are the exporter's own pinned plan: whole units (no shard
    ``start_state`` -- a slice is a later act, ``layout.slice_unit``, which
    the byte-time gate reads off the wire), unrotated, no diagonals, no
    RELEASE overrides, a scalar (arity-1) grid.  A caller planning otherwise
    says so and is refused by name.  ``structure`` is what the plan serves the
    unit AS (``tessera.structure.STRUCTURES``); it has NO passing default: a
    lane that publishes a structure-scoped requirement (the fused window
    lanes' ``column_rates_routed_moe``, tessera#694) refuses a plan that did
    not state it, and a lane that publishes none does not read it.
    """
    from fractions import Fraction

    from ..grammar import rate_set, root_from_q256
    from .contract import lane_requirements

    requires = lane_requirements(lane)
    q256 = int(q256)
    still_legal = (
        f"The rung is still legal to ENCODE and to SERVE -- the {lane} lane is one launch "
        "inside a route, and a unit it cannot read is served by that route's other path "
        "(for the window GEMV: the torch window decode plus _scaled_mm, same bytes, slower). "
        "What is refused here is only the CLAIM that this artifact exercises the lane.")

    # Per-WEIGHT root (see the docstring): agrees with the per-code spelling in
    # contract._validate_cell_executes on arity 1, which is every lane's grid.
    rates = rate_set(root_from_q256(q256), cap=int(rate_cap))
    facts = {
        "rates": rates, "window_bits": int(window_bits), "body": str(body),
        "plane": str(plane), "release_overrides": int(release_overrides),
        "diagonals": bool(diagonals), "rotation": str(rotation),
        "start_state": bool(start_state), "grid_arity": int(grid_arity),
        "structure": None if structure is None else str(structure),
    }
    refusals = decide_lane_requirements(lane, requires, facts)
    if not refusals:
        return rates
    notes = ""
    if any(refusal.startswith("column_rates_routed_moe") for refusal in refusals):
        notes = (
            f" q256={q256} realises column rates {list(rates)}; the lane reads them on the "
            f"wire (runtime_contract.json native_extensions[{lane}].lane.requires."
            "column_rates) but its routed-expert launch reaches only column_rates_routed_moe "
            "on the target, so an expert stack at this rung is served by the compact adapter "
            "and an artifact built to measure the lane on it would measure that adapter. "
            "Re-plan the stack on a rung whose rate set is inside column_rates_routed_moe, "
            "or plan it as a dense structure, which the lane's one-table dense launch reads "
            "at every rate of column_rates.")
    elif any(refusal.startswith("column_rates") for refusal in refusals):
        root = Fraction(q256, 256)
        notes = (
            f" q256={q256} is root rate {float(root):.4f}, which bresenham_rate_schedule "
            f"realises as column rates {list(rates)} (runtime_contract.json "
            f"native_extensions[{lane}].lane.requires.column_rates). EVERY unit of a "
            "checkpoint at this rung would refuse the lane at load and be served by the "
            "fallback, so an artifact built to measure the lane would measure the "
            "fallback. Re-plan on a rung whose rate set is inside the published set: an "
            "integral root lands every column on one rate (q256 = 256*R), and a "
            "fractional root mixes only the two rates bracketing it.")
    raise ValueError(
        f"tessera lane {lane!r} for {target}: grid {grid!r} at q256={q256} is refused -- "
        + "; ".join(refusals) + "." + notes + " " + still_legal)


def _validate_group(group: Mapping, family: str, target: str, *, byte_field: str,
                    experts: "int | None" = None) -> dict:
    """The geometry half of a scheme, for a dense module or one expert group.

    A dense module and a routed-MoE expert group are the SAME object at this
    level -- a stack of roles over one input width, at one rung per role,
    inside one ``tessera.fused`` container -- and the checks are therefore one
    function rather than two that could drift.  What differs is only the byte
    field's meaning, which is why the caller names it: a dense module declares
    ``wire_bytes``, the exact length of its one blob; an expert group declares
    ``wire_stride``, the width of the parameter row its E blobs are copied
    into, because their lengths differ per expert (the manifest's exact-ratio
    ``global_scale`` makes the blob length follow the data -- see
    ``tessera.moe_layout``) and no single number is every expert's length.
    """
    route = ROUTES[family]
    grid = group.get("grid")
    if grid not in route["grids"]:
        # The description comes off the route, not off an if-chain here: an
        # if-chain is a second place to remember, and the one that describes
        # every family it has not heard of as the family it was written for.
        raise ValueError(
            f"tessera target {target!r}: {family} holds {route['grid_kind']} grid "
            f"{route['grids']}, got {grid!r}")
    if group.get("plane") != route["plane"]:
        raise ValueError(
            f"tessera target {target!r}: {family} decodes the {route['plane']} scale plane to its "
            f"{route['tile']} tile; plane {group.get('plane')!r} has no {route['short']} tile")
    body = group.get("body")
    if body not in _BODIES:
        raise ValueError(f"tessera target {target!r}: body must be one of {_BODIES}, got {body!r}")
    if body != route["body"]:
        raise ValueError(
            f"tessera target {target!r}: {family} serves {route['body']} bodies; body {body!r} "
            f"has no {route['short']} tile")
    rows = _as_int(group, "rows", target)
    columns = _as_int(group, "columns", target)
    if family != TESSERA_NVFP4 and columns % route["columns_multiple"]:
        raise ValueError(
            f"tessera target {target!r}: the {family} mainloop needs "
            f"K % {route['columns_multiple']} == 0, got {columns}")
    wire = _as_int(group, byte_field, target)
    roles = group.get("roles")
    if (not isinstance(roles, (list, tuple)) or not roles
            or any(not (isinstance(r, (list, tuple)) and len(r) == 2 and isinstance(r[0], str)
                        and isinstance(r[1], int) and not isinstance(r[1], bool) and r[1] > 0)
                   for r in roles)):
        raise ValueError(
            f"tessera target {target!r}: roles must be a non-empty list of [name, rows] pairs, "
            f"got {roles!r}")
    if sum(r[1] for r in roles) != rows:
        raise ValueError(
            f"tessera target {target!r}: roles stack to {sum(r[1] for r in roles)} rows, the "
            f"scheme declares {rows}")
    if family == TESSERA_NVFP4:
        structure = STRUCTURE_ROUTED_MOE if experts is not None else STRUCTURE_DENSE
        for name, role_rows in roles:
            reason = e2m1_shape_reason(role_rows, columns, structure=structure, projection=name)
            if reason is not None:
                raise ValueError(f"tessera target {target!r} role {name!r}: {reason}")
    # AFTER the roles, because the rate is a per-ROLE fact and the roles are
    # what indexes it.  Every rung is put through the same gate the module-level
    # one used to be: a group is legal only when EVERY member's rate is one this
    # build's decoder publishes a read for.
    if experts is not None:
        matrix = _expert_role_rungs(group, experts, roles, family, target)
        if family == TESSERA_NVFP4:
            reason = e2m1_expert_rate_reason(matrix)
            if reason is not None:
                raise ValueError(f"tessera target {target!r}: {reason}")
        for e, row in enumerate(matrix):
            for (name, _role_rows), rung in zip(roles, row):
                _refuse_an_unreadable_rung(family, grid, rung,
                                           f"{target} expert {e} role {name!r}")
        rows_uniform = all(row == matrix[0] for row in matrix[1:])
        out = {
            "family": family, "grid": grid, "body": body, "plane": route["plane"],
            # The declared shape, normalised: an int when the WHOLE stack is
            # one rung (as every checkpoint before #967 is), else the FIRST
            # expert's row as a per-role list.  ``role_q256`` is that row: the
            # monomorphic one the dense-cut readers and every pre-#967
            # consumer read.  A genuinely across-expert mixed stack carries
            # the full matrix beside them, and nothing else moves.
            "q256": matrix[0][0] if len({r for row in matrix for r in row}) == 1
            else list(matrix[0]),
            "role_q256": [int(r) for r in matrix[0]],
            "rows": rows, "columns": columns,
            byte_field: wire, "roles": [(str(n), int(r)) for n, r in roles],
        }
        if not rows_uniform:
            out["expert_role_q256"] = [[int(r) for r in row] for row in matrix]
        return out
    rungs = _rungs_for_roles(group, roles, target)
    for (name, _role_rows), rung in zip(roles, rungs):
        _refuse_an_unreadable_rung(family, grid, rung, f"{target} role {name!r}")
    uniform = len(set(rungs)) == 1
    return {
        "family": family, "grid": grid, "body": body, "plane": route["plane"],
        # The declared shape, normalised: an int when the module is uniform (as
        # every checkpoint before #37 is), the per-role list when it is not.
        # ``role_q256`` is the monomorphic one a consumer should read.
        "q256": rungs[0] if uniform else list(rungs),
        "role_q256": [int(r) for r in rungs],
        "rows": rows, "columns": columns,
        byte_field: wire, "roles": [(str(n), int(r)) for n, r in roles],
    }


def validate_tessera_scheme(scheme: Mapping, target: str) -> dict:
    """Resolve a declared Tessera scheme without parsing the blob.

    Returns the normalised scheme; raises ``ValueError`` on anything no route
    can serve, at sidecar-parse time -- before a parameter exists.
    """
    family = scheme.get("family")
    if family not in TESSERA_FAMILIES:
        raise ValueError(
            f"tessera target {target!r}: family must be one of {TESSERA_FAMILIES}, got {family!r}")
    # A checkpoint written before the field existed is dense by construction:
    # every wire the plugin has ever served is one blob per vLLM Linear.
    structure = scheme.get("structure", STRUCTURE_DENSE)
    if structure not in STRUCTURES:
        raise ValueError(
            f"tessera target {target!r}: structure {structure!r} is not served; this plugin "
            f"serves {STRUCTURES} today. No method is registered for this structure.")
    if structure == STRUCTURE_ROUTED_MOE:
        return validate_tessera_moe_scheme(scheme, target)
    missing = [f for f in _REQUIRED if f not in scheme]
    if missing:
        raise ValueError(
            f"tessera target {target!r}: scheme is missing {missing}; a Tessera scheme "
            "declares its route, grid, body, plane, rate, geometry, byte count and roles")
    declared = _validate_group(scheme, family, target, byte_field="wire_bytes")
    declared["structure"] = structure
    return declared


def validate_tessera_moe_scheme(scheme: Mapping, target: str) -> dict:
    """Resolve a routed-MoE scheme: E experts, two groups, one route.

    THE SHAPE, AND WHY IT IS NOT THE DENSE ONE.  A dense scheme describes one
    module: one blob, one exact byte count. An expert stack carries per-expert
    gate, up and down containers: ``w13`` stacks gate then up in the row order
    ``RoutedExperts._load_w13`` narrows to, and ``w2`` holds down. Their lengths
    differ by projection and expert. So the sidecar declares the two GROUPS
    and the expert count, and per group a ``wire_stride`` (the parameter row
    width every expert's blob is copied into) rather than a ``wire_bytes``.
    The true length of a blob is the blob's own, carried beside it
    (``tessera.moe_layout``) and re-checked by ``fused.parse_fused``, which
    refuses trailing bytes -- so a wrong length is a refusal, not a short read.

    ``family``, ``grid``, ``body`` and ``plane`` are MODULE facts and are
    declared once at the top; vLLM builds one quant method per expert stack, so
    the two groups cannot take different routes any more than a fused Linear's
    members can (``FUSED_MODULE_FIELDS``).  ``rows``, ``columns``, ``roles``
    and ``q256`` are per group, because the two groups have genuinely different
    geometry: ``w13`` is ``[2N, K]``, ``w2`` is ``[K, N]``.
    """
    family = scheme.get("family")
    if family not in TESSERA_FAMILIES:
        raise ValueError(
            f"tessera target {target!r}: family must be one of {TESSERA_FAMILIES}, got {family!r}")
    source_layout = scheme.get("source_layout", MOE_SOURCE_UNPACKED)
    if source_layout not in MOE_SOURCE_LAYOUTS:
        raise ValueError(
            f"tessera target {target!r}: source_layout must be one of "
            f"{MOE_SOURCE_LAYOUTS}, got {source_layout!r}. The source convention "
            "decides how packed expert weights are sliced before encoding; an "
            "unknown value cannot be reconstructed safely from the emitted wires.")
    experts = _as_int(scheme, "experts", target)
    groups = scheme.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError(
            f"tessera target {target!r}: a routed_moe scheme declares its two expert groups "
            f"under 'groups' ({list(MOE_GROUPS)}), got {type(groups).__name__}")
    if tuple(sorted(groups)) != tuple(sorted(MOE_GROUPS)):
        raise ValueError(
            f"tessera target {target!r}: a routed_moe scheme declares exactly the groups "
            f"{sorted(MOE_GROUPS)}, got {sorted(groups)}; the groups are the tiles vLLM's "
            "fused-MoE kernel reads, so a third group names a tile no kernel takes and a "
            "missing one names a tile with no bytes")
    shared = {k: scheme.get(k) for k in ("family", "grid", "body", "plane")}
    declared_groups: dict[str, dict] = {}
    for name in MOE_GROUPS:
        group = groups[name]
        if not isinstance(group, Mapping):
            raise ValueError(
                f"tessera target {target!r}: group {name!r} must be a mapping, got "
                f"{type(group).__name__}")
        for field, value in shared.items():
            if field in group and group[field] != value:
                raise ValueError(
                    f"tessera target {target!r} group {name!r}: {field}={group[field]!r} "
                    f"disagrees with the module's {field}={value!r}. vLLM builds ONE quant "
                    f"method per expert stack, so {field} is a module fact "
                    "(scheme.FUSED_MODULE_FIELDS), not a per-group one")
        merged = dict(shared)
        merged.update(group)
        declared = _validate_group(merged, family, f"{target} group {name!r}",
                                   byte_field="wire_stride", experts=experts)
        if len(declared["roles"]) != MOE_GROUP_ROLES[name]:
            raise ValueError(
                f"tessera target {target!r} group {name!r}: {len(declared['roles'])} role(s) "
                f"{[r[0] for r in declared['roles']]}, expected {MOE_GROUP_ROLES[name]} -- the "
                f"group's members are exactly the shards the runtime loads into it "
                f"({MOE_GROUP_SHARDS[name]}, scheme.MOE_GROUP_SHARDS), in that row order")
        role_names = tuple(role for role, _ in declared["roles"])
        if role_names != MOE_GROUP_PROJECTIONS[name]:
            raise ValueError(
                f"tessera target {target!r} group {name!r}: roles {role_names} must be "
                f"{MOE_GROUP_PROJECTIONS[name]} in the runtime's row order")
        if name == "w13" and len({rows for _, rows in declared["roles"]}) != 1:
            raise ValueError(
                f"tessera target {target!r} group 'w13': role rows "
                f"{[rows for _, rows in declared['roles']]} must be equal halves; "
                "the runtime splits gate and up at N in its [2N, K] tile")
        declared_groups[name] = declared
    if declared_groups["w13"]["rows"] != 2 * declared_groups["w2"]["columns"]:
        raise ValueError(
            f"tessera target {target!r}: w13 stacks {declared_groups['w13']['rows']} rows and w2 "
            f"takes {declared_groups['w2']['columns']} input columns; w13 is [2N, K] and w2 is "
            "[K, N] over the same expert, so w13's rows are twice w2's columns")
    if declared_groups["w13"]["columns"] != declared_groups["w2"]["rows"]:
        raise ValueError(
            f"tessera target {target!r}: w13 takes {declared_groups['w13']['columns']} input "
            f"columns and w2 stacks {declared_groups['w2']['rows']} rows; both are the model's "
            "hidden size, so they are one number")
    return {
        "family": family, "structure": STRUCTURE_ROUTED_MOE, "experts": experts,
        "source_layout": source_layout,
        "grid": declared_groups["w13"]["grid"], "body": declared_groups["w13"]["body"],
        "plane": declared_groups["w13"]["plane"],
        "hidden_size": declared_groups["w13"]["columns"],
        "intermediate_size": declared_groups["w2"]["columns"],
        "groups": declared_groups,
    }


def _declared_members(blob: bytes, declared: Mapping, target: str,
                      expect_bytes: "int | None" = None):
    """The container's members against the declared role list, framing checked.

    Shared by the materialising and compact readers: the exact-length
    ``wire_bytes`` check (dense side), the fused framing, and the role list
    comparison are one question asked before either reader touches a unit.
    """
    from tessera import fused

    if expect_bytes is not None and len(blob) != expect_bytes:
        raise ValueError(
            f"tessera target {target!r}: scheme declares wire_bytes={expect_bytes} but "
            f"the loaded blob is {len(blob)} bytes")
    members = fused.parse_fused(bytes(blob))
    if [(m.name, m.rows) for m in members] != declared["roles"]:
        raise ValueError(
            f"tessera target {target!r}: the container holds roles "
            f"{[(m.name, m.rows) for m in members]} but the scheme declares {declared['roles']}")
    return members


def _per_weight_q256(root: int, arity: int, target: str, name: str) -> int:
    """The sidecar rung, from the manifest's per-CODE root rate.

    The manifest's root rate is per CODE and a code covers ``arity`` weights
    (``export.encode_linear_planes`` writes ``q256 * grid.arity``); the scheme
    speaks per weight, as the exporter's CLI and ``wire_recipe`` do.  ONE home
    for the divisibility refusal the materialising reader and the compact one
    both apply.
    """
    if root % int(arity):
        raise ValueError(
            f"tessera target {target!r} role {name!r}: root_q256={root} is not a whole "
            f"per-weight rate over an arity-{int(arity)} grid")
    return root // int(arity)


def _role_expected(declared: Mapping, member, member_q256: int) -> dict:
    """What the sidecar promises THIS role -- its own rung, not the module's."""
    expected = {"grid": declared["grid"], "body": declared["body"], "plane": declared["plane"],
                "q256": int(member_q256), "rows": member.rows,
                "columns": declared["columns"],
                "span": ROUTES[declared["family"]]["span"]}
    if declared["family"] == TESSERA_NVFP4:
        expected.update(window_bits=ROUTES[TESSERA_NVFP4]["window_bits"], half=GROUP_SIZE)
    return expected


def _require_role_facts(actual: Mapping, expected: Mapping, target: str, name: str) -> None:
    """Refuse a wire whose byte facts are not the sidecar's declaration.

    ONE home for the comparison: the materialising reader builds ``actual``
    off a ``ParsedUnit`` and the compact reader off ``ParsedMetadata``
    (``compact_prep.CompactWire.role_facts``), and both come through here, so
    the two can never describe one wire differently.  The sidecar carries no
    span field, so there is nothing to compare the wire's span against except
    the span the route itself reads: a span mismatch was refused nowhere until
    this comparison named it (neither ``validate_tessera_scheme`` nor the
    prepare gates checked it).
    """
    if actual != expected:
        raise ValueError(
            f"tessera target {target!r} role {name!r}: the wire is {actual} but the "
            f"sidecar scheme declares {expected}; refusing rather than serving bytes no "
            "receipt describes")


def _parse_container(blob: bytes, declared: Mapping, target: str, device="cpu",
                     expect_bytes: "int | None" = None) -> list:
    """One ``tessera.fused`` container against one normalised group.

    Shared by the dense route (whose container is the module) and the expert
    route (whose container is one expert's group), because it is one question
    in both places: are these bytes the roles, geometry, rungs and body the
    sidecar promised?  ``expect_bytes`` is the dense side's exact-length check;
    an expert's length is the blob's own and is bounded by the group's stride
    at the caller, so it passes ``None`` rather than a number it would have to
    invent.
    """
    from tessera import unit_artifact

    members = _declared_members(blob, declared, target, expect_bytes)
    parsed = []
    for member, member_q256 in zip(members, declared["role_q256"]):
        unit = unit_artifact.parse_unit_artifact(member.blob, device=device)
        geometry = unit.manifest.geometry
        actual = {
            "grid": unit.grid.name, "body": unit.body.name,
            "plane": unit.manifest.scale_plane.kind.name,
            "q256": _per_weight_q256(int(unit.manifest.branch.root_q256),
                                     unit.grid.arity, target, member.name),
            "rows": geometry.rows, "columns": geometry.columns,
            "span": int(unit.manifest.span),
        }
        if declared["family"] == TESSERA_NVFP4:
            actual.update(window_bits=int(unit.manifest.window_bits),
                          half=int(geometry.half_weights))
        _require_role_facts(actual, _role_expected(declared, member, member_q256),
                            target, member.name)
        parsed.append((member.name, unit))
    return parsed


def parse_tessera_blob_for_scheme(blob: bytes, scheme: Mapping, target: str, device="cpu") -> list:
    """Parse a module's fused container and refuse it unless it IS what the
    scheme declared.  Returns ``[(role, ParsedUnit)]`` in stacking order."""
    declared = validate_tessera_scheme(scheme, target)
    return _parse_container(blob, declared, target, device, expect_bytes=declared["wire_bytes"])


def _parse_compact_container(blob: bytes, declared: Mapping, target: str,
                             device="cuda", expect_bytes: "int | None" = None,
                             memo: "dict | None" = None) -> list:
    """The compact reader's half of ``_parse_container``: same questions, no
    expanded plane.

    Same container framing and role list (``_declared_members``), same
    per-member rung/geometry/body/grid comparison built through
    ``_per_weight_q256`` / ``_role_expected`` / ``_require_role_facts``, and
    the same refusals -- through ``unit_artifact.parse_unit_metadata``, so no
    weight plane is expanded and no reference decode runs.  Returns
    ``[(role, CompactWire)]`` in stacking order.  One home for both the dense
    and the expert compact adapters, exactly as ``_parse_container`` is for
    the materialising ones.
    """
    from tessera.compact_prep import parse_compact_wire

    members = _declared_members(blob, declared, target, expect_bytes)
    out = []
    for member, member_q256 in zip(members, declared["role_q256"]):
        wire = parse_compact_wire(member.blob, device=device, name=member.name,
                                  memo=memo)
        meta = wire.metadata
        actual = {
            "grid": meta.grid.name, "body": meta.body.name,
            "plane": meta.manifest.scale_plane.kind.name,
            "q256": _per_weight_q256(int(meta.manifest.branch.root_q256),
                                     meta.grid.arity, target, member.name),
            "rows": meta.rows, "columns": meta.columns,
            "span": int(meta.span),
        }
        if declared["family"] == TESSERA_NVFP4:
            actual.update(window_bits=int(meta.manifest.window_bits),
                          half=int(meta.manifest.geometry.half_weights))
        _require_role_facts(actual, _role_expected(declared, member, member_q256),
                            target, member.name)
        out.append((member.name, wire))
    return out


def parse_compact_blob_for_scheme(blob: bytes, scheme: Mapping, target: str,
                                  device="cuda") -> list:
    """The compact twin of ``parse_tessera_blob_for_scheme``: a module's fused
    container, validated against the scheme it declares, with no weight plane
    expanded and no reference decode.  ``[(role, CompactWire)]``.

    (It imports ``compact_prep``, which imports torch, so it lives here as a
    function and the module stays torch-free at import.)
    """
    declared = validate_tessera_scheme(scheme, target)
    return _parse_compact_container(blob, declared, target, device,
                                    expect_bytes=declared["wire_bytes"])


def parse_compact_tessera_expert_blob(blob: bytes, declared_role: Mapping, target: str,
                                      device="cpu", memo: "dict | None" = None) -> list:
    """The compact twin of ``parse_tessera_expert_blob``, signature for signature.

    ``declared_role`` is one entry of :func:`expert_role_declarations`.  The
    length check is the group's ``wire_stride`` (an expert's blob is as long as
    its own manifest made it) and ``fused.parse_fused`` is what refuses a blob
    that does not END where the caller said it does -- the same two checks, in
    the same order, with the same words as the materialising reader.  Returns
    ``[(role, CompactWire)]``.
    """
    stride = int(declared_role["wire_stride"])
    if len(blob) > stride:
        raise ValueError(
            f"tessera target {target!r}: the expert blob is {len(blob)} bytes, longer than the "
            f"group's declared wire_stride={stride} -- the parameter row it was copied into "
            "ends before the blob does, so this is truncated data rather than a shorter read")
    return _parse_compact_container(blob, declared_role, target, device,
                                    expect_bytes=None, memo=memo)


def expert_role_declarations(declared_group: Mapping, *,
                             expert: int = 0) -> "list[dict]":
    """One single-member declaration per projection, in the group's row order,
    resolved for ONE expert.

    A routed-MoE checkpoint stores ONE container per expert PROJECTION.  That
    is the granularity of the checkpoint's tensors, of ``RoutedExperts``' shard
    ids (``w1``/``w3``/``w2``, one call each) and of ``tessera.moe_layout``'s
    cells; the GROUP is how those containers stack into the tile the fused-MoE
    kernel reads, not a container of its own.  So the group's role list indexes
    containers, and each is checked as the single-member container it is --
    same ``_parse_container``, same refusals, one role at a time.

    ``expert`` resolves the rungs for that expert BEFORE any byte is validated
    and before any TP cut: a mixed per-unit stack (#967) carries its rungs in
    the group's ``expert_role_q256`` matrix, and the loader asks for expert e's
    row here so each container is checked against its own unit's declaration.
    The default (``expert=0``) keeps the legacy one-argument call -- the first
    expert's row, which is what ``role_q256`` alone ever described -- for
    uniform stacks and pre-#967 callers.
    """
    matrix = declared_group.get("expert_role_q256")
    if matrix is not None:
        if not 0 <= int(expert) < len(matrix):
            raise ValueError(
                f"expert {expert} is outside the stack's {len(matrix)} declared expert(s); "
                "the declarations resolve one expert at a time, in expert order")
        row = matrix[int(expert)]
    else:
        # No matrix: the group's rungs are expert-invariant, so any expert
        # index resolves to the same row -- there is no per-expert fact to
        # bound here, and the caller's expert loop stays the authority.
        row = declared_group["role_q256"]
    out = []
    for (name, rows), rung in zip(declared_group["roles"], row):
        out.append({
            "family": declared_group["family"], "grid": declared_group["grid"],
            "body": declared_group["body"], "plane": declared_group["plane"],
            "columns": declared_group["columns"], "rows": int(rows),
            "roles": [(str(name), int(rows))], "role_q256": [int(rung)],
            "q256": int(rung), "wire_stride": declared_group["wire_stride"],
        })
    return out


def expert_group_q256(rungs) -> "int | list":
    """The sidecar spelling for a group's per-expert rung matrix.

    One writer-side answer to the three spellings the group gate reads: the
    scalar when every (expert, role) carries one rung, the shared row as a
    per-role list when every expert's row is the same, the ``[experts][roles]``
    matrix only when the rows genuinely differ.  The exporter emits through
    this and the loader reads through ``_expert_role_rungs``, so the spelling
    has one home and cannot drift into a second convention.
    """
    rows = [[int(r) for r in row] for row in rungs]
    if not rows:
        raise ValueError("a routed group's rung matrix needs at least one expert row")
    if len({r for row in rows for r in row}) == 1:
        return rows[0][0]
    if all(row == rows[0] for row in rows[1:]):
        return list(rows[0])
    return rows


def expert_rungs_mixed(declared_group: Mapping) -> bool:
    """Whether a normalised routed group's rungs differ across experts.

    The normalised shape answers directly: the matrix exists exactly when the
    expert rows differ, so the intake's size preallocation and the publication
    validator read this one predicate instead of restating the field's rule.
    """
    return declared_group.get("expert_role_q256") is not None


def stack_effective_rungs(declared: Mapping) -> "list[int]":
    """Every rung a routed stack's expert projections carry, sorted.

    The guard's question is the EFFECTIVE set, not any one spelling: an
    expert-major matrix, a per-role list, and group-level differences (w13 at
    one rung, w2 at another) all name real served schedules, and the fused
    gate/up launch needs one run table and stride for the whole stack.  One
    home for the gathering so the loader, the intake and any future gate read
    the same set.
    """
    rungs: set = set()
    experts = int(declared["experts"])
    for group in MOE_GROUPS:
        group_decl = declared["groups"][group]
        matrix = group_decl.get("expert_role_q256")
        rows = matrix if matrix is not None else [group_decl["role_q256"]] * experts
        for row in rows:
            rungs.update(int(r) for r in row)
    return sorted(rungs)


def parse_tessera_expert_blob(blob: bytes, declared_role: Mapping, target: str,
                              device="cpu") -> list:
    """One expert projection's container against the role the sidecar declared.

    ``declared_role`` is one entry of :func:`expert_role_declarations`.  The
    length check is the group's stride rather than an exact byte count -- an
    expert's blob is as long as its own manifest made it -- and
    ``fused.parse_fused`` is what refuses a blob that does not END where the
    caller said it does, which is the check that matters.
    """
    stride = int(declared_role["wire_stride"])
    if len(blob) > stride:
        raise ValueError(
            f"tessera target {target!r}: the expert blob is {len(blob)} bytes, longer than the "
            f"group's declared wire_stride={stride} -- the parameter row it was copied into "
            "ends before the blob does, so this is truncated data rather than a shorter read")
    return _parse_container(blob, declared_role, target, device, expect_bytes=None)
