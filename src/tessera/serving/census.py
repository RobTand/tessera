"""Lane engagement: did the arm that requested a lane get any modules on it?

A route census (``tools/tessera_route_census.py``) reads the route record
every served Tessera module wrote and checks each one against the pair its
route owns.  That is a check on AGREEMENT, and agreement is exactly what a
void experiment produces: the streamed FP8 route's decode regime may report
the window-GEMV pair OR the materialised pair -- a rate-1 unit and a box with
no toolchain both legitimately fall back inside that regime -- so a serve in
which the GEMV lane prepared for *nothing* passes module by module.  Issue
#104 is what that costs.  Four censuses of the #91 reproduction logged 112 of
112 modules refusing the lane at load, every receipt recorded one route,
``decoder torch_window``, ``problems: []``, and the two arms of the experiment
were one lane state wearing two names.

So this module adds the question the per-module check cannot ask: **an arm
that requested a route and got zero units on it is a failed census.**  It is
a value, not a log line -- :func:`lane_engagement` returns a block a gate
reads (per phase, per lane, how many modules took it) plus the problems that
block implies -- and it is emitted whether or not anything was required, so a
receipt written without ``--require-lane`` still says how many modules took
each decoder instead of leaving it to be inferred from a histogram.

``all_required_engaged`` is deliberately three-valued: ``true``, ``false``, or
``null`` when the census was never told which lane the arm was built to
exercise.  A gate must be able to tell "nothing was required" from "everything
required was engaged", because the first is the state every receipt in the
#104 report was in.

Torch-free on purpose: the tool imports this inside a serving container, and
the tests exercise it on synthetic and replayed records with no GPU at all.
"""
from __future__ import annotations

import collections
from typing import Any, Mapping

__all__ = [
    "CELL_AGREEMENT_SCHEMA",
    "LANE_ENGAGEMENT_SCHEMA",
    "STRUCTURE_BY_RECORD_KIND",
    "cell_launch_agreement",
    "decoder_histogram",
    "lane_engagement",
]

#: Bumped when the block's shape changes.  A consumer keys on it.
LANE_ENGAGEMENT_SCHEMA = "tessera.lane-engagement/1"

#: A route record's ``kind`` (``telemetry.ROUTE_FIELDS``) -> the
#: ``lane_eligibility`` STRUCTURE whose cells could cover it.  Two vocabularies
#: for one thing, and the join between them belongs in exactly one place: the
#: record says what the layer IS ("dense" | "moe"), a cell says what it covers
#: ("dense" | "routed_moe"), and :func:`cell_launch_agreement` is keyed to one
#: structure per block.  A kind absent from this map is covered by nothing --
#: the conservative direction, since the alternative is a block reporting
#: agreement for a launch nobody published.
STRUCTURE_BY_RECORD_KIND = {"dense": "dense", "moe": "routed_moe"}


def decoder_histogram(records: Mapping[str, Mapping[str, Any]]) -> dict:
    """``decoder -> module count`` over one phase's route records.

    A record with no ``decoder`` is counted under ``""``: a route that stamped
    nothing is a hole in the observation, not an absence of routes, and it
    must not disappear into a total.
    """
    counts = collections.Counter(str(r.get("decoder") or "") for r in records.values())
    return dict(sorted(counts.items()))


def lane_engagement(records_by_phase: Mapping[str, Mapping[str, Mapping[str, Any]]],
                    *, required_lanes=(), lane_decoders: "Mapping[str, str] | None" = None,
                    refusals_by_phase: "Mapping[str, Mapping[str, str]] | None" = None,
                    contract: "Mapping[str, Any] | None" = None) -> "tuple[dict, list[str]]":
    """``(block, problems)`` for the lanes this arm was built to exercise.

    ``records_by_phase`` maps a census phase to that phase's Tessera route
    records, keyed by module name -- the records the census already filtered
    to Tessera policies.  ``required_lanes`` names lanes by the extension
    ``module_name_prefix`` the runtime contract publishes them under; the
    decoder each stamps is read from the contract (``lane.decoder``), never
    restated by the caller, so a lane rename fails at the contract rather than
    silently counting zero for a name nothing writes.  ``lane_decoders``
    overrides that resolution for a test with no packaged contract.

    ``refusals_by_phase`` maps a phase to ``{module: refusal}`` -- what the
    load path recorded when a lane could not prepare
    (``telemetry.read_lane_refusal``).  It is aggregated into the block so a
    receipt says WHY zero modules took the lane, which is the difference
    between "the lane is unreached" and "nobody looked".
    """
    required = tuple(dict.fromkeys(str(lane) for lane in required_lanes))
    if lane_decoders is None:
        from .contract import lane_decoder
        decoders = {lane: lane_decoder(lane, contract) for lane in required}
    else:
        missing = [lane for lane in required if lane not in lane_decoders]
        if missing:
            raise ValueError(f"no decoder given for required lane(s) {missing}")
        decoders = {lane: str(lane_decoders[lane]) for lane in required}

    problems: "list[str]" = []
    phases: "dict[str, dict]" = {}
    engaged_anywhere = {lane: 0 for lane in required}
    all_refusals: "collections.Counter[str]" = collections.Counter()
    modules_seen = 0
    for phase in sorted(records_by_phase):
        records = records_by_phase[phase]
        hist = decoder_histogram(records)
        engaged = {lane: int(hist.get(decoders[lane], 0)) for lane in required}
        for lane, n in engaged.items():
            engaged_anywhere[lane] = max(engaged_anywhere[lane], n)
        refusals = dict(sorted(collections.Counter(
            str(v) for v in (refusals_by_phase or {}).get(phase, {}).values()).items()))
        # A load-time refusal is a fact about the LOAD, not about a phase: the
        # tool hands the same {module: refusal} map to every phase, so summing
        # would report "224 of 112 modules refused" on a two-phase census.
        # Take the largest count each reason reached in any one phase.
        for reason, n in refusals.items():
            all_refusals[reason] = max(all_refusals[reason], n)
        modules_seen = max(modules_seen, len(records))
        phases[phase] = {
            "tessera_modules": len(records),
            "decoders": hist,
            "engaged_modules": engaged,
            "unengaged_required_lanes": [lane for lane, n in engaged.items() if n == 0],
            "lane_refusals": refusals,
        }

    # THE VERDICT IS OVER THE CENSUS, NOT OVER EACH PHASE.  A lane is one
    # launch inside a route and it owns a REGIME, not a serve: the window GEMV
    # decodes M <= 8 (``kernel_window_gemv.GEMV_MAX_M``), so a prefill that
    # takes the torch decode is the lane working as specified, not a hole.
    # Refusing per phase would have made every correct serve a failed census
    # and taught the next reader to ignore the field.  What cannot happen is
    # zero modules in EVERY phase: that arm measured the fallback throughout.
    for lane in required:
        if engaged_anywhere[lane]:
            continue
        reason = ""
        if all_refusals:
            worst = max(all_refusals.items(), key=lambda kv: kv[1])
            reason = (f" {worst[1]} of {modules_seen} module(s) recorded a load-time "
                      f"refusal, most commonly: {worst[0]}")
        per_phase = {phase: phases[phase]["decoders"] for phase in phases}
        problems.append(
            f"lane {lane!r} was REQUIRED and took 0 of {modules_seen} Tessera modules in "
            f"any phase (decoder {decoders[lane]!r}; observed {per_phase}). An arm that "
            "requested a route and got zero units on it measured the fallback, not the lane."
            + reason)

    block = {
        "schema": LANE_ENGAGEMENT_SCHEMA,
        "required_lanes": list(required),
        "required_decoders": dict(decoders),
        "phases": phases,
        "engaged_modules_max": dict(engaged_anywhere),
        # Three-valued on purpose: ``None`` means nobody said what to require.
        "all_required_engaged": (None if not required
                                 else all(n > 0 for n in engaged_anywhere.values())),
    }
    return block, problems


#: Bumped when the block's shape changes.  A consumer keys on it.
CELL_AGREEMENT_SCHEMA = "tessera.cell-launch-agreement/3"


def cell_launch_agreement(records_by_phase, *, cells, phase_regimes, platform,
                          rungs_by_module, families_by_route, structure="dense",
                          symbol_alias=None, runtime_image=None, execution_mode=None):
    """Every served record's launch against the CELL that covers it (#111).

    The lane-engagement block above asks whether a lane took any modules.  This
    asks the other half: does what the serve executed match what the contract
    says it executes?  Since lane_eligibility schema v4 a cell publishes
    ``executes`` -- the ``(symbol, decoder)`` launches the route makes at that
    cell's regime, residency and rungs -- so the question is answerable, and a
    receipt is where it has to be answered: the cell is DERIVED from the
    dispatch table, which proves the document agrees with the code, and only a
    serve proves the code agrees with the machine.

    The route and the residency come off the record's own ``policy`` field
    (``ROUTE:mode``) and ``families_by_route`` turns the route into the payload
    family a cell is keyed by (``contract.PAYLOAD_FAMILY_BY_ROUTE``), so the
    caller supplies the rung per module and the measured image/execution mode.
    Since lane schema v5, those runtime facts must match the cell's explicit
    scope. Missing context and cells without scope cover nothing; a global
    image pin is never borrowed. A module the table does not cover
    is counted as ``unattested`` and is NOT a problem: absence is the honest
    state a closed-world table has for a rung no receipt covered, and inventing a
    verdict from it is the failure the whole table is shaped to avoid.

    Returns ``(block, problems)``.  ``agrees`` is three-valued for the same
    reason ``all_required_engaged`` is: ``None`` when no record was covered by
    any cell, so a gate can tell "nothing to check" from "everything checked".
    """
    from .contract import cell_runtime_scope, refuse_unevaluated_predicates
    from .scheme import eager_regime_problem

    runtime = {"image": runtime_image, "execution_mode": execution_mode}
    unsupported_reason = (
        "compiled dense route records combine shape-polymorphic launches; "
        "per-cell agreement is unsupported for this observation"
        if execution_mode == "compiled" and structure != "routed_moe" else None)
    by_regime = {}
    for cell in cells:
        if cell.get("platform") != platform or cell.get("structure") != structure:
            continue
        # A cell this block would consider, checked before it is keyed: the key
        # below is (family, residency, regime, rung) and a predicate is what
        # that key cannot carry, so a narrowed cell entering it would report
        # agreement for launches it never covered.
        refuse_unevaluated_predicates(cell, f"lane_eligibility cell {cell.get('id')}")
        if runtime_image is None or execution_mode is None or "runtime" not in cell:
            continue
        image, execution_modes = cell_runtime_scope(cell)
        if image != runtime_image or execution_mode not in execution_modes:
            continue
        modes = _cell_modes(cell)
        for mode in modes:
            for rung in cell.get("rungs_q256", ()):
                key = (cell["family"], mode, cell["regime"], int(rung))
                by_regime[key] = cell

    phases = {}
    problems = []
    covered_total = 0
    for phase, records in sorted(records_by_phase.items()):
        regime = phase_regimes.get(phase)
        counts = collections.Counter()
        covered = unattested = unsupported = 0
        for name, record in sorted(records.items()):
            # THE RECORD'S OWN STRUCTURE, CHECKED BEFORE ITS RUNG.  This block
            # is keyed to one structure and a record of another is not
            # something its cells can cover.  Explicit rather than left to the
            # accident that today a routed-expert stack's module name carries
            # no rung: the record is written at ``<prefix>.routed_experts``
            # while the checkpoint declares ``<prefix>``, and the day a caller
            # resolves rungs across that join (the census's
            # ``join_records_to_declared`` does exactly that join for the
            # family lookup) a DENSE cell would begin "covering" a stack that
            # executes vLLM's fused-MoE kernel -- reporting agreement, or a
            # disagreement, about a launch no cell in this contract publishes.
            if STRUCTURE_BY_RECORD_KIND.get(str(record.get("kind"))) != structure:
                unattested += 1
                continue
            if unsupported_reason:
                unattested += 1
                unsupported += 1
                continue
            policy = str(record.get("policy", ""))
            route, _, mode = policy.partition(":")
            family = families_by_route.get(route)
            rung = rungs_by_module.get(name)
            cell = (None if rung is None or family is None
                    else by_regime.get((family, mode, regime, int(rung))))
            if cell is None:
                unattested += 1
                continue
            pair = (str(record.get("symbol")), str(record.get("decoder")))
            if execution_mode == "compiled" and (
                    len(cell["executes"]) != 1 or any("+" in item for item in pair)):
                # Only the routed single-launch observation can be joined to
                # a phase cell. A combined trace does not identify which of
                # its branches the driven phase executed.
                unattested += 1
                unsupported += 1
                continue
            if execution_mode == "eager":
                # THE PHASE LABEL IS NOT THE OBSERVATION.  The cell above was
                # selected by the regime this phase DECLARES; what the machine
                # ran is the record's own M, and the two are different facts.
                # Resident FP8 publishes one launch pair for both regimes, so
                # an eight-row forward filed under the decode phase agreed
                # with the decode cell and was counted as decode evidence
                # (#207).  A regime attestation is the shape that ran.
                why = eager_regime_problem(record.get("shape"), regime)
                if why is not None:
                    unattested += 1
                    problems.append(
                        f"{phase}: {name} is keyed to cell {cell['id']!r}, which covers regime "
                        f"{regime!r}, but {why}. The phase label is what the census asked for; "
                        "the record's shape is what the machine ran, and only the second "
                        "attests a regime.")
                    continue
            covered += 1
            allowed = {(str(e["symbol"]), str(e["decoder"])) for e in cell["executes"]}
            counts[cell["id"]] += 1
            # The route may carry an observed backend suffix while a cell
            # publishes the runtime entry point. Preserve exact matching too:
            # a cell that explicitly pins a backend must not accept another.
            alias_pair = ((str(symbol_alias(pair[0])), pair[1])
                          if symbol_alias is not None else pair)
            if pair not in allowed and alias_pair not in allowed:
                problems.append(
                    f"{phase}: {name} executed {pair!r}, which cell {cell['id']!r} does not "
                    f"publish (it executes {sorted(allowed)!r}). The cell is what a producer "
                    "prices and a gate reads; a serve that disagrees with it means one of the "
                    "two is describing a runtime nobody ran.")
        covered_total += covered
        phases[phase] = {"regime": regime, "modules": len(records),
                         "covered_by_cell": covered, "unattested": unattested,
                         "unsupported_records": unsupported,
                         "cells": dict(sorted(counts.items()))}
    block = {"schema": CELL_AGREEMENT_SCHEMA, "platform": platform, "structure": structure,
             "runtime": runtime, "phases": phases,
             # A problem is a disagreement whether or not the record it was
             # raised on survived to be counted: a record refused for
             # attesting the wrong regime is NOT covered, and a block that
             # read ``None`` there would publish "nothing to check" over an
             # observation it had just refused.
             "agrees": False if problems else (None if not covered_total else True)}
    if unsupported_reason:
        block["unsupported_reason"] = unsupported_reason
    return block, problems


def _cell_modes(cell):
    """The residencies a cell's serve flags name.

    The published contract validates the flag strictly. This observation
    reader treats a missing ``TESSERA_SERVE_MODE=a|b`` flag as covering
    nothing rather than covering every residency.
    """
    head = "TESSERA_SERVE_MODE="
    for flag in cell.get("requires_serve_flags", ()):
        if str(flag).startswith(head):
            return tuple(str(flag)[len(head):].split("|"))
    return ()
