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
    "LANE_ENGAGEMENT_SCHEMA",
    "decoder_histogram",
    "lane_engagement",
]

#: Bumped when the block's shape changes.  A consumer keys on it.
LANE_ENGAGEMENT_SCHEMA = "tessera.lane-engagement/1"


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
    for phase in sorted(records_by_phase):
        records = records_by_phase[phase]
        hist = decoder_histogram(records)
        engaged = {lane: int(hist.get(decoders[lane], 0)) for lane in required}
        for lane, n in engaged.items():
            engaged_anywhere[lane] = max(engaged_anywhere[lane], n)
        unengaged = [lane for lane, n in engaged.items() if n == 0]
        refusals = dict(sorted(collections.Counter(
            str(v) for v in (refusals_by_phase or {}).get(phase, {}).values()).items()))
        phases[phase] = {
            "tessera_modules": len(records),
            "decoders": hist,
            "engaged_modules": engaged,
            "unengaged_required_lanes": unengaged,
            "lane_refusals": refusals,
        }
        for lane in unengaged:
            reason = ""
            if refusals:
                worst = max(refusals.items(), key=lambda kv: kv[1])
                reason = (f" {worst[1]} of {len(records)} module(s) recorded a load-time "
                          f"refusal, most commonly: {worst[0]}")
            problems.append(
                f"{phase}: lane {lane!r} was REQUIRED and took 0 of {len(records)} Tessera "
                f"modules (decoder {decoders[lane]!r}; observed {hist}). An arm that requested "
                f"a route and got zero units on it measured the fallback, not the lane."
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
