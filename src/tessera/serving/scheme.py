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

* ``TESSERA_NVFP4`` is any E2M1-based grid over a LUT scale plane, decoded to
  the stock NVFP4 tile (nibble-packed E2M1 codes + group-16 ue4m3 block scales
  + one global) and served through ``torch._scaled_mm`` W4A4.
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

from typing import Any, Mapping

__all__ = [
    "NVFP4_ACTIVATION_CONTRACT",
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
    "MOE_GEMM_SYMBOL",
    "moe_census_symbol_base",
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
    "refuse_a_family_with_no_expert_route",
    "refuse_unserveable_wire",
    "refuse_unreachable_lane",
    "lane_rate_report",
    "validate_tessera_scheme",
    "validate_tessera_moe_scheme",
    "parse_tessera_blob_for_scheme",
    "expert_role_declarations",
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
STRUCTURE_DENSE = "dense"
STRUCTURE_ROUTED_MOE = "routed_moe"
STRUCTURES = (STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE)

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
#: Only ``TESSERA_FP8`` is here, and the absences are measured rather than
#: preferred.  ``TESSERA_NVFP4``: the pinned build's fused-MoE oracle resolves
#: an NVFP4 expert arm only under a ``swiglu_limit`` clamp
#: (``docs/measurements/nvfp4-moe-oracle-2026-09-02.md``), which changes the
#: arithmetic the experts execute, so there is no NVFP4 expert tile to decode
#: to that is the same object the dense NVFP4 route serves.  ``TESSERA_BF16``:
#: a 16-bit expert stack is the passthrough vLLM already serves through
#: ``ignore``, and a route that decoded a wire to it would spend the wire's
#: bytes to arrive where no wire is needed.
MOE_BUILDERS: dict[str, tuple[str, str]] = {
    TESSERA_FP8: ("tessera.serving.moe_route", "build_tessera_moe_method"),
}

#: What each route can hold, by Tessera's own names (``PayloadGrid.name``,
#: ``ScalePlaneKind.name``).  NVFP4: grids whose codes are E2M1 nibbles (arity
#: 1 or 2 over the E2M1 base) over the LUT plane the tile's ue4m3 block scales
#: come from.  FP8: the scalar E4M3 grid over the CHANNEL plane the tile's
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
#: (``ops.prepare_tessera_module`` for NVFP4, ``fp8_route`` for FP8,
#: ``bf16_route`` for BF16).  They are written here because the PRODUCER needs
#: them too: ``refuse_unserveable_wire`` is the export-time half of the same
#: rule, and the exporter used to carry its own copy in an if/elif -- a third
#: statement about one decoder.
ROUTES: dict[str, dict] = {
    TESSERA_NVFP4: {
        "grids": ("E2M1", "E2M1x2"), "plane": "LUT",
        "body": "TCQ", "span": 2,
        "short": "NVFP4",
        "grid_kind": "an E2M1-based",
        "builder": ("tessera.serving.nvfp4_route", "build_tessera_nvfp4_method"),
        "tile": "nvfp4 (packed E2M1 codes, group-16 ue4m3 block scales, one global)",
        "columns_multiple": 16,
        "activation_contract": NVFP4_ACTIVATION_CONTRACT,
        "gemm_symbol": "torch._scaled_mm",
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

#: The op the window-GEMV lane dispatches through, in the spelling
#: ``kernel_window_gemv`` registers it under.  It lives HERE, torch-free,
#: because a producer reading the contract has to be able to resolve a cell's
#: ``executes`` entry without importing the kernel -- the same reason
#: ``gemm_symbol`` is a ``ROUTES`` field and not a literal in the route module.
WINDOW_GEMV_SYMBOL = "tessera_window_gemv::gemv"
#: The entry point the expert route calls. Its recorded backend suffix is
#: selected by vLLM at runtime and remains in the census receipt.
MOE_GEMM_SYMBOL = "vllm.fused_moe.modular_kernel"

#: The decoder each launch stamps.  Strings rather than an import of
#: ``telemetry``, which imports torch; ``tests/test_serving_contract.py`` ties
#: every one of them to ``telemetry.DECODERS`` where torch is installed.
_DECODER_NATIVE_SPAN2 = "native_span2"
_DECODER_TORCH_WINDOW = "torch_window"
_DECODER_WINDOW_GEMV = "window_gemv"
_DECODER_TORCH_STOCK = "torch_materialize_stock"

_ALL_REGIMES = ("batch", "decode")
_ALL_MODES = ("resident", "streamed")
#: The extension whose ``lane`` block gates the two window launches below.
#: Spelled here rather than imported from ``ext`` to keep this module's
#: import graph flat; ``test_the_launch_tables_lane_is_the_published_extension``
#: ties it to ``ext.WINDOW_GEMV_MODULE_NAME``.
_WINDOW_GEMV_LANE = "tessera_window_gemv"


def _window_launches(gemm_symbol: str) -> tuple[dict, ...]:
    """The three launches a WINDOW-body route makes, given its GEMM.

    The FP8 and BF16 routes differ in their alphabet, their tile and their
    GEMM; they do not differ in this shape, and writing it twice is how the
    second one would quietly stop matching the first.

    The two lane launches are separated by M through :func:`regime_of_m`, and
    ``tests/test_serving_contract.py`` derives both ``regimes`` fields below
    from the routes' own ``decode_is_gemv`` rather than trusting them.
    """
    return (
        # The materialised tile.  ``when_lane_absent`` is the ``elif
        # getattr(layer, "tessera_gemv", None) is None`` branch both routes
        # take, as a value: in ``resident`` mode there is no lane to be absent
        # from and this is simply the launch, and in ``streamed`` mode it is
        # what runs on a unit the lane did not prepare.  M does not enter it.
        {"symbol": gemm_symbol, "decoder": _DECODER_TORCH_WINDOW,
         "regimes": _ALL_REGIMES, "modes": _ALL_MODES, "lane": None,
         "structures": (STRUCTURE_DENSE,),
         "when_lane_absent": True},
        # The same GEMM over a tile the LANE's kernel decoded: the branch
        # ``decode_is_gemv`` refuses.  Every forward it can run on has M > 1 --
        # M above ``GEMV_MAX_M``, or M >= 3 on a unit with a rate-1 column,
        # which has no 8-row lane -- so it is a BATCH launch and cannot occur
        # at one row.
        {"symbol": gemm_symbol, "decoder": _DECODER_WINDOW_GEMV,
         "regimes": ("batch",), "modes": ("streamed",), "lane": _WINDOW_GEMV_LANE,
         "structures": (STRUCTURE_DENSE,),
         "when_lane_absent": False},
        # The lane's own op, in BOTH regimes.  One row always takes it (the
        # rate-1 refusal starts at the 4-row tile), and so does the two-row
        # tile, on every unit the lane prepared: ``items_key(2)`` is the 1-key
        # table, which a rate-1 column has.  So the batch regime -- every
        # M > 1 forward, not only a first prefill -- launches the GEMV too,
        # and a cell that says otherwise is right about a 64-row prefill and
        # wrong about the runtime.
        {"symbol": WINDOW_GEMV_SYMBOL, "decoder": _DECODER_WINDOW_GEMV,
         "regimes": _ALL_REGIMES, "modes": ("streamed",), "lane": _WINDOW_GEMV_LANE,
         "structures": (STRUCTURE_DENSE,),
         "when_lane_absent": False},
    )


ROUTE_LAUNCHES: dict[str, tuple[dict, ...]] = {
    # One launch: the span-2 kernel decodes and the scaled GEMM runs, in both
    # regimes and both residencies.  The stock-materialise decoder is what runs
    # when the extension is ABSENT, which is published as a fallback
    # (``ext.NATIVE_EXTENSIONS[].when_unavailable``) and is not a launch an
    # attested cell may name.
    TESSERA_NVFP4: (
        {"symbol": ROUTES[TESSERA_NVFP4]["gemm_symbol"], "decoder": _DECODER_NATIVE_SPAN2,
         "regimes": _ALL_REGIMES, "modes": _ALL_MODES, "lane": None,
         "structures": (STRUCTURE_DENSE,),
         "when_lane_absent": True},
    ),
    TESSERA_FP8: _window_launches(ROUTES[TESSERA_FP8]["gemm_symbol"]) + (
        {"symbol": MOE_GEMM_SYMBOL, "decoder": _DECODER_TORCH_STOCK,
         "regimes": _ALL_REGIMES, "modes": ("resident",), "lane": None,
         "structures": (STRUCTURE_ROUTED_MOE,), "when_lane_absent": True},
    ),
    TESSERA_BF16: _window_launches(ROUTES[TESSERA_BF16]["gemm_symbol"]),
}


def route_launches(route: str, *, structure: str = STRUCTURE_DENSE,
                   regime: str | None = None, mode: str | None = None,
                   lanes: "tuple[str, ...] | None" = None) -> tuple[dict, ...]:
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

    There is deliberately no RATE axis.  A rate decides whether the lane can
    read a rung at all -- ``refuse_unreachable_lane``, and the caller passes
    the answer in ``lanes`` -- and above that it decides only which M the GEMV
    covers *within* a regime, never which launches the regime contains: the
    one-row forward is always the GEMV and the batch regime always holds both,
    rate-1 columns or not.  A rate filter here read as the second and cost the
    batch cell its GEMV launch.
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


def _refuse_an_unreadable_rung(route: str, grid: str, q256: int, target: str) -> None:
    """The rate has to be one the decoder reads, and the contract has to say so.

    Nothing checked this.  An allocated checkpoint served seven rungs
    (R749/R750/R934/R1006/R1083/R1107/R1262) that the contract's published set
    did not name, and nothing refused -- the run happened to be fine because
    every one of them IS decodable, but a rung the decoder could not read would
    have produced a wrong tensor rather than a refusal, which is the failure
    this gate exists for.

    The published range is derived from the decoder, not from what has been
    exported: each candidate rate was encoded and taken through this very load
    path, and the accepted set is what the contract now states.  Two different
    mechanisms bound it.  On ``E4M3`` the trellis grammar's shaped domain
    (code rate 1..8 over an 8-bit-native alphabet) gives a CONTINUOUS
    ``q256`` in [256, 2048] -- so the contract says continuous, not a list of
    the two rungs anyone had built.  On ``E2M1x2`` the same grammar caps the
    top at 896 (rate 7 of arity-2 native 8) and the native decoder cuts off the
    bottom: it serves the span-2 TCQ body only, and below 896 the recipe writes
    a WINDOW body that has no served decode.  One point, and it is a real one.
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
        f"(scheme.MOE_BUILDERS names {sorted(MOE_BUILDERS)}). The absences are measured, "
        "not preferred: on this build the fused-MoE oracle resolves an NVFP4 expert arm "
        "only under a swiglu_limit clamp that changes the arithmetic the experts execute "
        "(docs/measurements/nvfp4-moe-oracle-2026-09-02.md), and a 16-bit expert stack is "
        "the passthrough quantization_config.ignore already gives. An expert stack is "
        "refused rather than decoded through another family's tile: plan it on a family "
        "with a route, or leave it out to pass it through as BF16.")


def attested_cells(family: str, structure: str,
                   contract: "Mapping | None" = None) -> "list[dict]":
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
    """
    from .contract import load_serving_contract, refuse_unevaluated_predicates

    payload = load_serving_contract() if contract is None else contract
    cells = [cell for cell in payload["lane_eligibility"]["cells"]
             if cell["family"] == family and cell["structure"] == structure]
    for cell in cells:
        refuse_unevaluated_predicates(cell, f"lane_eligibility cell {cell.get('id')}")
    return cells


def refuse_unserveable_wire(grid: str, q256: int, body: str, plane: str,
                            *, family: "str | None" = None,
                            span: "int | None" = None, target: str,
                            structure: str = STRUCTURE_DENSE,
                            contract: "Mapping | None" = None) -> str:
    """Refuse, AT EXPORT, a wire this plugin build publishes no decode for.

    The producer's output range was wider than the consumer's input range, in
    the one direction nothing checked (#41).  ``export.wire_recipe`` writes the
    WINDOW body over LUT16 for every sub-cap ``E2M1x2`` unit -- the shipping
    default below q256 896 -- and the contract publishes ``E2M1x2`` as the
    single point 896, so a legal low-rate unit encoded fine and was refused at
    LOAD, hours later, on the operator rather than at export on the exporter.

    This is principle 9's one carve-out for a producer-side refusal: a MEASURED
    platform fact -- the pinned runtime has no native route for these bytes --
    and not a taste-based ban on a format.  It is therefore scoped to the
    SERVING boundary: research and measurement encode sub-cap ``E2M1x2``
    constantly (the whole rate-frontier body of work does) and are untouched.
    ``encode_linear``, ``wire_recipe`` and the rung axis stay exactly as wide as
    they were; what narrows is only what may be written into a checkpoint that
    declares ``quant_method: "tessera"``.

    Principle 14: every bound it reads comes from the packaged
    ``runtime_contract.json`` (``contract.reader_rate_grid`` /
    ``reader_accepts``) or from ``ROUTES``, the table the loader itself gates
    on.  Nothing here hardcodes 896: the day a measured range widens, the file
    changes and this gate follows it.

    Returns the route the wire would take.  Raises ``ValueError`` naming the
    unit, the ``(route, grid, q256)`` it asked for, the range this build
    publishes, and the fact that the rung is still encodable for research.

    ``family`` names the route being written.  The exporter knows it -- it is
    the family the checkpoint will declare for this wire -- so it passes it,
    and the gate reads the published range for THAT ``(route, grid)`` pair
    rather than resolving the route from the grid alone.  Resolving from the
    grid alone is ambiguous the moment two routes hold one grid, and the
    winner would be whichever entry the dict order puts first (#51); omitting
    ``family`` keeps the old behaviour for grids one route holds, and refuses
    an ambiguous grid instead of guessing.

    ``structure`` names what the wire is served AS, and it picks the BOUND
    (#135).  A dense module is read by the route's decoder, whose range is
    the format row's ``reader_rate_range_q256``; a ``routed_moe`` stack is
    read by ``moe_route`` into a fused-MoE kernel the contract attests
    separately, as cells of its own structure at their own ``rungs_q256``.
    Until this parameter existed a routed stack was held to the dense
    reader's range -- ``[256, 2048]`` on E4M3 -- while the routed cells
    attest exactly one rung, and no producer gate read them.  The bound for
    a non-dense structure is the union of its cells' rungs, so a rung the
    dense cells attest and no routed cell does is refused, and the refusal
    names the cells it read.  ``contract`` is for a caller holding a table
    other than the packaged one (tests); the packaged file is the default.
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
    if structure == STRUCTURE_DENSE:
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
        # bound is the union of those cells' rungs -- which the contract
        # validator already keeps inside the row's range, so this is the
        # tighter of the two and the only one that names the right kernel.
        cells = attested_cells(family, structure, contract)
        if not cells:
            raise ValueError(
                f"tessera export {target!r}: no lane_eligibility cell in runtime_contract.json "
                f"attests {family} served as structure {structure!r}, so this build promises "
                f"no rung for a {structure} stack on grid {grid!r}. A cell is published when a "
                f"container receipt covers the structure on a runtime image; absence is "
                f"'unattested', which an export that declares the structure cannot ship on. "
                + still_legal)
        rungs = sorted({int(r) for cell in cells for r in cell["rungs_q256"]})
        if q256 not in rungs:
            per_cell = "; ".join(f"{cell['id']} attests {sorted(cell['rungs_q256'])}"
                                 for cell in cells)
            raise ValueError(
                f"tessera export {target!r}: q256={q256} on grid {grid!r} is outside the rungs "
                f"the {structure} cells of runtime_contract.json attest for {family} "
                f"({rungs}: {per_cell}). The format row's reader range is the dense route's; "
                f"a {structure} stack is served by a different consuming kernel and is "
                f"attested only at the rungs its own cells name. Re-plan the stack on one of "
                f"{rungs}, or serve the rung and publish the cell. " + still_legal)
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


def refuse_unreachable_lane(lane: str, *, grid: str, q256: int, rate_cap: int,
                            body: str, plane: str, window_bits: int,
                            target: str) -> tuple[int, ...]:
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

    Every bound comes from the packaged contract (``lane_requirements``),
    never from a constant here: the day the kernel grows a 6-byte lane, the
    contract changes and this follows it.  Returns the accepted rate set.
    """
    from fractions import Fraction

    from ..grammar import rate_set, root_from_q256
    from .contract import lane_requirements, route_wire_spelling

    requires = lane_requirements(lane)
    q256 = int(q256)
    still_legal = (
        f"The rung is still legal to ENCODE and to SERVE -- the {lane} lane is one launch "
        "inside a route, and a unit it cannot read is served by that route's other path "
        "(for the window GEMV: the torch window decode plus _scaled_mm, same bytes, slower). "
        "What is refused here is only the CLAIM that this artifact exercises the lane.")

    wanted_body = requires.get("body")
    if wanted_body is not None and body != route_wire_spelling("body", wanted_body):
        raise ValueError(
            f"tessera lane {lane!r} for {target}: the lane reads the "
            f"{route_wire_spelling('body', wanted_body)} body; grid {grid!r} at q256={q256} "
            f"resolves to {body}. " + still_legal)
    wanted_plane = requires.get("plane")
    if wanted_plane is not None and plane != route_wire_spelling("plane", wanted_plane):
        raise ValueError(
            f"tessera lane {lane!r} for {target}: the lane reads the "
            f"{route_wire_spelling('plane', wanted_plane)} scale plane; grid {grid!r} at "
            f"q256={q256} resolves to {plane}. " + still_legal)
    windows = [int(w) for w in requires.get("window_bits", ())]
    if windows and int(window_bits) not in windows:
        raise ValueError(
            f"tessera lane {lane!r} for {target}: the lane's value table is built for window "
            f"bits {windows}; grid {grid!r} at q256={q256} resolves to L={int(window_bits)}. "
            + still_legal)

    rates = rate_set(root_from_q256(q256), cap=int(rate_cap))
    report = lane_rate_report(lane, rates)
    if not report["reachable"]:
        supported = report["supported"]
        root = Fraction(q256, 256)
        raise ValueError(
            f"tessera lane {lane!r} for {target}: q256={q256} is root rate {float(root):.4f}, "
            f"which bresenham_rate_schedule realises as column rates {report['rates']} -- and "
            f"{report['offending']} {'is' if len(report['offending']) == 1 else 'are'} outside "
            f"the rates this lane reads ({supported}, runtime_contract.json "
            f"native_extensions[{lane}].lane.requires.column_rates). EVERY unit of a checkpoint "
            f"at this rung would refuse the lane at load and be served by the fallback, so an "
            f"artifact built to measure the lane would measure the fallback. Re-plan on a rung "
            f"whose rate set is inside {supported}: an integral root lands every column on one "
            f"rate (q256 = 256*R), and a fractional root mixes only the two rates bracketing "
            f"it. " + still_legal)
    return rates


def _validate_group(group: Mapping, family: str, target: str, *, byte_field: str) -> dict:
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
    if columns % route["columns_multiple"]:
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
    # AFTER the roles, because the rate is a per-ROLE fact and the roles are
    # what indexes it.  Every rung is put through the same gate the module-level
    # one used to be: a group is legal only when EVERY member's rate is one this
    # build's decoder publishes a read for.
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
                                   byte_field="wire_stride")
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
    from tessera import fused, unit_artifact

    if expect_bytes is not None and len(blob) != expect_bytes:
        raise ValueError(
            f"tessera target {target!r}: scheme declares wire_bytes={expect_bytes} but "
            f"the loaded blob is {len(blob)} bytes")
    members = fused.parse_fused(bytes(blob))
    if [(m.name, m.rows) for m in members] != declared["roles"]:
        raise ValueError(
            f"tessera target {target!r}: the container holds roles "
            f"{[(m.name, m.rows) for m in members]} but the scheme declares {declared['roles']}")
    parsed = []
    for member, member_q256 in zip(members, declared["role_q256"]):
        unit = unit_artifact.parse_unit_artifact(member.blob, device=device)
        geometry = unit.manifest.geometry
        # The manifest's root rate is per CODE; a code covers ``arity`` weights
        # (``export.encode_linear_planes`` writes ``q256 * grid.arity``).  The
        # scheme speaks per weight, as the exporter's CLI and ``wire_recipe`` do.
        root = int(unit.manifest.branch.root_q256)
        if root % unit.grid.arity:
            raise ValueError(
                f"tessera target {target!r} role {member.name!r}: root_q256={root} is not a whole "
                f"per-weight rate over an arity-{unit.grid.arity} grid")
        actual = {
            "grid": unit.grid.name, "body": unit.body.name,
            "plane": unit.manifest.scale_plane.kind.name, "q256": root // unit.grid.arity,
            "rows": geometry.rows, "columns": geometry.columns,
            # The sidecar carries no span field, so there is nothing to compare
            # the wire's span against except the span the route itself reads:
            # a span mismatch was refused nowhere until this comparison named
            # it (neither this function nor validate_tessera_scheme checked
            # it, and the prepare_* gates below did not either).
            "span": int(unit.manifest.span),
        }
        # THIS ROLE's rung, not the module's: the sidecar carries one per role
        # (``FUSED_MODULE_FIELDS``), so the member the reader parsed is compared
        # against the rate the sidecar promised for THAT member.
        expected = {"grid": declared["grid"], "body": declared["body"], "plane": declared["plane"],
                    "q256": int(member_q256), "rows": member.rows,
                    "columns": declared["columns"],
                    "span": ROUTES[declared["family"]]["span"]}
        if actual != expected:
            raise ValueError(
                f"tessera target {target!r} role {member.name!r}: the wire is {actual} but the "
                f"sidecar scheme declares {expected}; refusing rather than serving bytes no "
                "receipt describes")
        parsed.append((member.name, unit))
    return parsed


def parse_tessera_blob_for_scheme(blob: bytes, scheme: Mapping, target: str, device="cpu") -> list:
    """Parse a module's fused container and refuse it unless it IS what the
    scheme declared.  Returns ``[(role, ParsedUnit)]`` in stacking order."""
    declared = validate_tessera_scheme(scheme, target)
    return _parse_container(blob, declared, target, device, expect_bytes=declared["wire_bytes"])


def expert_role_declarations(declared_group: Mapping) -> "list[dict]":
    """One single-member declaration per projection, in the group's row order.

    A routed-MoE checkpoint stores ONE container per expert PROJECTION.  That
    is the granularity of the checkpoint's tensors, of ``RoutedExperts``' shard
    ids (``w1``/``w3``/``w2``, one call each) and of ``tessera.moe_layout``'s
    cells; the GROUP is how those containers stack into the tile the fused-MoE
    kernel reads, not a container of its own.  So the group's role list indexes
    containers, and each is checked as the single-member container it is --
    same ``_parse_container``, same refusals, one role at a time.
    """
    out = []
    for (name, rows), rung in zip(declared_group["roles"], declared_group["role_q256"]):
        out.append({
            "family": declared_group["family"], "grid": declared_group["grid"],
            "body": declared_group["body"], "plane": declared_group["plane"],
            "columns": declared_group["columns"], "rows": int(rows),
            "roles": [(str(name), int(rows))], "role_q256": [int(rung)],
            "q256": int(rung), "wire_stride": declared_group["wire_stride"],
        })
    return out


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
