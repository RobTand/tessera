#!/usr/bin/env python3
"""Census of the dispatch routes a served Tessera checkpoint executes.

The ``lane_eligibility`` cells in ``tessera/serving/runtime_contract.json``
state which route the plugin *executes*.  This script is the serve-side
observation behind such a cell: it loads a Tessera checkpoint through
``vllm.LLM`` in one process, runs a prefill-shaped forward and a decode-shaped
forward, then reads the route record every Tessera module wrote
(``tessera.serving.telemetry.read_route``) from inside the worker.  No log line
is parsed: the record is the same scalars the route tests assert on, read from
the same objects the serve dispatched through.

It exits non-zero unless every Tessera module, in both shapes, reports
``state == "served"``, a ``<family>:<mode>`` policy equal to the family the
checkpoint declares for that module, that route's activation contract, and a
``(symbol, decoder)`` pair the route owns for the driven regime (the streamed
FP8 route reports the window-GEMV pair wherever the lane prepared, and the
materialised tile under the stock GEMM in the batch regime alone -- the shapes
above the lane's max M, which no one-row forward can be -- with the streamed
BF16 route the same shape over ``torch.mm``) -- so the JSON it writes is a
receipt only when the run also passed.

AND WHAT THE CONTRACT SAYS IT EXECUTES.  Since ``lane_eligibility`` schema v4
a cell publishes ``executes`` -- the ``(symbol, decoder)`` launches the route
makes at that cell's regime, residency and rungs -- so "does the serve match
the document" is finally a question with an answer, and this is where it has
to be asked: the cell is DERIVED from the dispatch table, which proves the
document agrees with the code, and only a serve proves the code agrees with
the machine.  ``census.cell_launch_agreement`` joins every record to the cell
covering its ``(platform, family, structure, regime, residency, rung)``, the
``cell_launch_agreement`` block lands in the receipt, and a disagreement is a
refusal.  A module at a rung no cell covers is ``unattested``, which is the
only negative signal a closed-world table has and is not a failure.  The check
is EAGER-ONLY: a compiled record stamps both launches as one ``a+b`` pair
because one graph serves every M, so a compiled run writes ``agrees: null``
with a reason rather than reading a traced record as a disagreement.

AND ONE QUESTION THE PER-MODULE CHECK CANNOT ASK.  Everything above is a check
on AGREEMENT, and agreement is what a void experiment produces: every regime
of the streamed FP8 route legitimately admits the window-GEMV pair OR the
torch window decode, so a serve in which the GEMV lane prepared for *nothing*
passes module by module.  Issue #104 is what that cost -- four censuses logged
112 of 112 modules refusing the lane at load, every receipt recorded one route
and ``problems: []``, and the two arms of the experiment were one lane state
wearing two names.  So ``--require-lane`` (or the artifact's own
``requires_lanes``, stamped by ``export_tessera_serving.py --require-lane``)
names the lane the arm was BUILT to exercise, and a phase in which that lane
took zero modules is a REFUSAL.  The ``lane_engagement`` block is written
either way -- decoder counts per phase, and ``all_required_engaged`` as
``true``/``false``/``null``, the third meaning nobody said what to require --
so a gate can tell "nothing was required" from "everything required was
engaged".  ``lane_refusals`` carries what the load path recorded when a lane
could not prepare, so the receipt says WHY it took nothing.

usage::

    tessera_route_census.py <checkpoint-dir> <out.json> \
        [--expect-modules N] [--prompt-tokens 64] [--gpu-memory-utilization 0.3]

The two forwards it drives are the two regimes ``lane_eligibility`` declares.
The census calls them by the shape it drove (``prefill``, ``decode``) because
its receipt is keyed by those names and served receipts quote them; the
contract calls them ``batch`` and ``decode``.  One table maps the pair --
``tessera.serving.contract.CENSUS_PHASE_REGIMES`` -- the census resolves its
phase names through it and stamps the contract's word into every histogram
entry, and ``load_serving_contract`` refuses a contract whose declared regimes
are not exactly that table's values.  So a per-(family, regime) expectation can
join the two sides, and a rename or a third regime fails before the first model
load rather than at a per-module ``KeyError`` after two.

Run it inside the serving image with the plugin installed (the same container
the KL dumps ran in); ``TESSERA_SERVE_MODE`` selects the residency exactly as
it does for ``vllm serve``.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import subprocess
import sys
import time


#: The regimes this tool can actually drive, in the contract's vocabulary: one
#: many-row forward and one one-row forward.  It is a statement about this
#: tool's two ``llm.generate`` calls, not about the contract -- which is why a
#: regime declared there and absent here is a refusal below rather than a
#: silently unobserved cell.
DRIVEN_REGIMES = ("batch", "decode")


def census(model):
    """Runs inside the worker: every module carrying a route record."""
    from tessera.serving.telemetry import read_route
    out = {}
    for name, mod in model.named_modules():
        rec = read_route(mod)
        if rec is not None:
            out[name] = rec
    return out


def declared_in_module_space(model, targets):
    """Runs inside the worker: the checkpoint's target names, in vLLM's namespace.

    ``config_groups`` targets are written in the CHECKPOINT's namespace;
    ``named_modules()`` -- where every route record is read from -- is the
    namespace vLLM built.  For a model class that declares an
    ``hf_to_vllm_mapper`` those are different strings for the same module,
    which is why vLLM hands the quant config the mapper at load and why
    ``TesseraConfig.apply_vllm_mapper`` exists.

    Returns ``None`` when the class declares no mapper -- then checkpoint space
    IS module space, which is the case for every census taken before this
    (Qwen3-0.6B) and is why the omission never showed.  A target the mapper
    DROPS maps to ``None``: the runtime builds no module for it, and the caller
    reports that rather than passing the unmapped name through.

    The mapper is the runtime's own table, replayed here, not restated.
    """
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return None
    unstacked = mapper.get_unstacked_mapper()
    out = {}
    for target in targets:
        if "." not in target or target.startswith("re:"):
            out[target] = target      # a module class name or a regex, not a path
            continue
        mapped = unstacked.apply_list([target])
        out[target] = mapped[0] if mapped else None
    return out


def join_records_to_declared(records, declared):
    """Which declared target each route record belongs to, and what is ambiguous.

    A dense module carries its record where the checkpoint names it, so the
    join is the identity.  A ROUTED EXPERT STACK does not: vLLM builds one
    quant method for the stack's prefix and attaches it to the
    ``RoutedExperts`` child it constructs underneath, so the record is read at
    ``<declared>.routed_experts`` and an exact-name join reports the same
    served stack TWICE -- once as a route the checkpoint declares nothing for,
    once as a declaration nothing served.  That is what the first routed-MoE
    census said: eight dense modules clean, three expert stacks served, and
    ``REFUSED`` on six problems that were one name.

    The join is by containment and only for an expert record: a record whose
    ``kind`` is ``moe`` and whose module path lies under exactly one declared
    target belongs to that target.  ``kind`` is the record's own word for what
    it served, so the rule reads the runtime's statement rather than matching
    the child's name; a second Tessera module nested under a declared one would
    be AMBIGUOUS and is reported, never guessed.
    """
    owner, problems = {}, []
    for name, record in records.items():
        if name in declared:
            owner[name] = name
            continue
        if record.get("kind") != "moe":
            continue
        parents = sorted(d for d in declared if name.startswith(d + "."))
        if len(parents) == 1:
            owner[name] = parents[0]
        elif parents:
            problems.append(
                f"{name} is an expert record under {len(parents)} declared targets "
                f"{parents}; a stack belongs to one declaration or to none")
    return owner, problems


def lane_refusals(model):
    """Runs inside the worker: every module whose LANE refused at load.

    A load fact, read once, never from ``apply()``: what the route record
    cannot say is that the lane the artifact was built to exercise took
    nothing, and a stderr warning is not a value a gate reads -- 112 of them
    scrolled past under four censuses that each reported ``problems: []``
    (issue #104).
    """
    from tessera.serving.telemetry import read_lane_refusal
    out = {}
    for name, mod in model.named_modules():
        refusal = read_lane_refusal(mod)
        if refusal is not None:
            out[name] = refusal
    return out


def _git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best effort, the check is not
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--expect-modules", type=int, default=None,
                    help="number of Tessera modules the checkpoint declares")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--compiled", action="store_true",
                    help="load with enforce_eager=False (vLLM's default compiled forward + CUDA "
                         "graphs) instead of eager; the route records then carry M='*' because "
                         "the record is written from the trace, and a route that cannot be traced "
                         "fails here with its own traceback instead of an engine-start refusal")
    ap.add_argument("--allow-fallback-decoder", action="store_true",
                    help="accept a module decoded by the pure-torch fallback instead of the "
                         "native span-2 kernel; without it a fallback serve REFUSES, because a "
                         "receipt must not claim the native route for bytes another decoder made")
    ap.add_argument("--require-lane", action="append", default=None, metavar="LANE",
                    help="a lane this arm was BUILT to exercise, named by the extension "
                         "module_name_prefix runtime_contract.json publishes it under (e.g. "
                         "tessera_window_gemv). The census then REFUSES a phase in which that "
                         "lane took zero modules: an arm that requested a route and got no "
                         "units on it measured the fallback, not the lane (issue #104). "
                         "Repeatable. Without it the engagement block is still written -- with "
                         "all_required_engaged null, which is how a gate tells 'nothing was "
                         "required' from 'everything required was engaged'.")
    ap.add_argument("--no-manifest-lanes", action="store_true",
                    help="ignore requires_lanes in the checkpoint's tessera_serving_manifest.json; "
                         "by default an artifact that DECLARES which lane it was built for is "
                         "believed, so the requirement travels with the bytes rather than with a "
                         "shell history")
    ap.add_argument("--tessera-commit", default=None,
                    help="the host's `git rev-parse HEAD` for the Tessera checkout under test; "
                         "inside a container a worktree's .git pointer resolves nowhere and the "
                         "receipt would carry None")
    args = ap.parse_args()

    # The census function must run in the process that holds the model.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    import tessera
    import tessera.serving as serving
    from tessera.serving import bf16_route, fp8_gemv, fp8_route, nvfp4_route
    from tessera.serving.census import cell_launch_agreement, lane_engagement
    from tessera.serving.contract import (
        CENSUS_PHASE_REGIMES, PAYLOAD_FAMILY_BY_ROUTE, load_serving_contract)
    from tessera.serving.lane import TESSERA_MODE_ENV
    from tessera.serving.scheme import (
        ROUTES, TESSERA_BF16, TESSERA_FAMILIES, TESSERA_FP8, TESSERA_NVFP4)
    from tessera.serving.telemetry import DECODER_NATIVE_SPAN2, DECODER_TORCH_WINDOW

    # The executed A-side contract each route stamps on its layers: the value a
    # cell publishes, compared here against what the serve recorded.
    contract_for = {TESSERA_NVFP4: nvfp4_route.ACTIVATION_CONTRACT,
                    TESSERA_FP8: fp8_route.ACTIVATION_CONTRACT,
                    TESSERA_BF16: bf16_route.ACTIVATION_CONTRACT}
    # The decoder each route must have used.  The NVFP4 route's must be
    # the native span-2 kernel unless the operator explicitly accepted the
    # fallback.
    decoder_for = {TESSERA_NVFP4: DECODER_NATIVE_SPAN2, TESSERA_FP8: DECODER_TORCH_WINDOW,
                   TESSERA_BF16: DECODER_TORCH_WINDOW}
    # The GEMM each route invokes, off the route table rather than a literal
    # here: the two 4/8-bit routes call ``torch._scaled_mm`` and the 16-bit one
    # calls ``torch.mm`` (there is no scale to hand a scaled GEMM -- the row
    # scale is an epilogue), and a hardcoded symbol read that as a refusal on
    # every module of a route it had simply never been told about.
    symbol_for = {family: ROUTES[family]["gemm_symbol"] for family in TESSERA_FAMILIES}
    # The streamed FP8 route serves two launches where the lane prepared: the
    # window GEMV, which BOTH regimes may report (the one-row forward always
    # takes it, and so does the two-row tile), and the kernel-decoded tile
    # under ``_scaled_mm``, which only the batch regime can -- it is the
    # branch ``decode_is_gemv`` refuses, and every M that refuses is above one
    # row.  Where the lane did not prepare, the torch window decode, at any M.
    # The streamed BF16 route is the same shape over ``torch.mm``.  The pairs
    # each regime may report live where the dispatch lives
    # (``fp8_gemv.census_expected``, ``bf16_route.census_expected``), not in a
    # second spelling here; every other family reports one pair.
    fp8_expected = fp8_gemv.census_expected(compiled=args.compiled)
    bf16_expected = bf16_route.census_expected(compiled=args.compiled)

    def _expected(family, regime):
        if family == TESSERA_FP8:
            return fp8_expected[regime]
        if family == TESSERA_BF16:
            return bf16_expected[regime]
        return {(symbol_for[family], decoder_for[family])}
    missing = sorted(set(TESSERA_FAMILIES) - (set(contract_for) & set(decoder_for)))
    if missing:
        raise SystemExit(
            f"this census has no expectation for {missing}; a family the plugin serves and "
            "the census does not know would be counted as a mismatch on every module. Add "
            "its contract and decoder above rather than widening the comparison.")

    # ONE regime vocabulary (issue #61).  ``load_serving_contract`` refuses a
    # contract whose ``lane_eligibility.regimes`` are not exactly this table's
    # values, and the phase names below are resolved THROUGH the table rather
    # than written a second time here.  Both checks run before the first model
    # load (85-160 s each in tessera-bf16-route-served-2026-09-02.md), because
    # the one outcome a receipt tool must not have is failing at a per-module
    # lookup with two loaded models behind it.
    load_serving_contract()
    undrivable = sorted(set(CENSUS_PHASE_REGIMES.values()) - set(DRIVEN_REGIMES))
    if undrivable:
        raise SystemExit(
            f"the contract declares the regime(s) {undrivable}, which this census does not "
            f"drive (it drives {list(DRIVEN_REGIMES)}). A declared regime no forward exercises "
            "is a cell nothing observes; add the forward here rather than widening the table.")
    phase_of = {regime: phase for phase, regime in CENSUS_PHASE_REGIMES.items()}
    if len(phase_of) != len(CENSUS_PHASE_REGIMES):
        raise SystemExit(
            f"CENSUS_PHASE_REGIMES maps two phases onto one regime ({dict(CENSUS_PHASE_REGIMES)}); "
            "the join is then ambiguous in the direction this tool reads it.")
    batch_phase, decode_phase = phase_of["batch"], phase_of["decode"]

    # THE LANE THE ARM REQUESTED, resolved BEFORE the first model load: a lane
    # name this build publishes no decoder for must fail in a second, not after
    # two 85-160 s loads.  The artifact's own declaration is read first, so the
    # requirement travels with the bytes (export_tessera_serving.py --require-lane
    # stamps requires_lanes into the manifest) rather than depending on whoever
    # types the census command.
    required_lanes = list(args.require_lane or ())
    manifest_lanes = []
    manifest_path = os.path.join(args.model, "tessera_serving_manifest.json")
    if not args.no_manifest_lanes and os.path.isfile(manifest_path):
        with open(manifest_path) as fh:
            manifest_lanes = list(json.load(fh).get("requires_lanes") or ())
    for lane in manifest_lanes:
        if lane not in required_lanes:
            required_lanes.append(lane)
    lane_decoders = {}
    from tessera.serving.contract import lane_decoder as _lane_decoder
    for lane in required_lanes:
        lane_decoders[lane] = _lane_decoder(lane)      # raises on an unpublished lane
    if required_lanes:
        print(f"[census] required lane(s): "
              + ", ".join(f"{lane} -> decoder {lane_decoders[lane]!r}" for lane in required_lanes)
              + (f" (declared by the artifact: {manifest_lanes})" if manifest_lanes else ""),
              flush=True)

    with open(os.path.join(args.model, "config.json")) as fh:
        cfg = json.load(fh)
    qc = cfg.get("quantization_config", {})
    groups = qc.get("config_groups", {})
    tessera_groups = {k: g for k, g in groups.items()
                      if g.get("scheme", {}).get("family") in TESSERA_FAMILIES}
    # Which family the checkpoint declares for each module: the route the serve
    # must have taken, module by module (a mixed checkpoint has both).
    declared = {t: g["scheme"]["family"] for g in tessera_groups.values() for t in g.get("targets", [])}

    # THE RUNG PER MODULE, the one fact a route record does not carry and the
    # last key a cell is resolved by.  A fused module's ``q256`` is a per-role
    # LIST since contract v6; a group whose members disagree resolves to no
    # rung, so its modules land in the honest ``unattested`` bucket rather than
    # borrowing one member's cell.
    def _rung(scheme):
        value = scheme.get("q256")
        if isinstance(value, list):
            distinct = {int(v) for v in value}
            return distinct.pop() if len(distinct) == 1 else None
        return None if value is None else int(value)

    declared_rungs = {t: _rung(g["scheme"])
                      for g in tessera_groups.values() for t in g.get("targets", [])}

    problems = []
    t0 = time.time()
    llm = LLM(model=args.model, enforce_eager=not args.compiled, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=0)

    # CHECKPOINT NAMES ARE NOT MODULE NAMES, and this join is made in module
    # space.  ``config_groups`` targets are written in the CHECKPOINT's
    # namespace; ``named_modules()`` -- where every route record above is read
    # from -- is the namespace vLLM built.  For a model class that declares an
    # ``hf_to_vllm_mapper`` the two differ, and vLLM hands the quant config the
    # mapper for exactly this reason (``TesseraConfig.apply_vllm_mapper``).
    # This tool did not: on Qwen3-0.6B, which declares no mapper, the two
    # spaces coincide and every census so far was taken there.  On
    # ``Glm5NextForConditionalGeneration`` the mapper is
    # ``{"model.language_model." -> "language_model.model.", ...}``, so without
    # this NOTHING joins and every served module is reported as one the
    # checkpoint declares no wire for -- a refusal that says the opposite of
    # what is true.  The table is the RUNTIME's (the model class's own mapper),
    # replayed here rather than restated.
    name_map = llm.apply_model(
        lambda model: declared_in_module_space(model, list(declared)))[0]
    if name_map is not None:
        dropped = sorted(t for t, m in name_map.items() if m is None)
        if dropped:
            problems.append(
                f"the model's hf_to_vllm_mapper drops {len(dropped)} declared target(s), "
                f"e.g. {dropped[:3]}; the runtime builds no module for them")
        declared = {name_map[t] or t: f for t, f in declared.items()}
        declared_rungs = {name_map[t] or t: r for t, r in declared_rungs.items()}
    tok = llm.get_tokenizer()
    text = ("The receipt names the route the serve took, for every module, "
            "in both the prefill and the decode shape. ") * 20
    ids = tok.encode(text, add_special_tokens=False)[: args.prompt_tokens]
    prompt = {"prompt_token_ids": ids}

    phases = {}
    # One forward over M = len(ids) rows, sample one token, stop.
    outs = llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0))
    phases[batch_phase] = llm.apply_model(census)[0]
    # Decode steps follow; the last forward is one row wide.
    outs = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0))
    phases[decode_phase] = llm.apply_model(census)[0]
    generated = outs[0].outputs[0].text
    # Load facts, so once and after the forwards: which modules' lane refused
    # to prepare, and why.  Same for every phase by construction.
    refusals = llm.apply_model(lane_refusals)[0]

    mode = os.environ.get(TESSERA_MODE_ENV, "")
    prefixes = tuple(f"{family}:" for family in TESSERA_FAMILIES)
    histogram = {}
    record_owner = {}
    for phase, recs in phases.items():
        tess = {n: r for n, r in recs.items() if str(r.get("policy", "")).startswith(prefixes)}
        other = {n: r for n, r in recs.items() if n not in tess}
        h = collections.Counter(
            (r["policy"], r["symbol"], r["contract"], r["state"], r.get("kind"), r.get("decoder"))
            for r in tess.values())
        by_family = collections.Counter(str(r["policy"]).split(":")[0] for r in tess.values())
        histogram[phase] = {
            # The contract's word for the shape this phase drove, so a
            # per-(family, regime) expectation joins the receipt to a cell
            # without either side guessing the other's vocabulary.
            "regime": CENSUS_PHASE_REGIMES[phase],
            "tessera_modules": len(tess),
            "tessera_modules_by_family": dict(sorted(by_family.items())),
            "other_route_modules": len(other),
            "routes": [dict(policy=k[0], symbol=k[1], contract=k[2], state=k[3], kind=k[4],
                            decoder=k[5], modules=v)
                       for k, v in sorted(h.items(), key=lambda kv: tuple(map(str, kv[0])))],
            "shapes": sorted({str(r.get("shape")) for r in tess.values()}),
        }
        if not tess:
            problems.append(f"{phase}: no module reports a Tessera route")
        owner, join_problems = join_records_to_declared(tess, declared)
        record_owner[phase] = owner
        problems.extend(f"{phase}: {m}" for m in join_problems)
        for name, r in tess.items():
            family = declared.get(owner.get(name, name))
            if family is None:
                problems.append(
                    f"{phase}: {name} took a Tessera route but the checkpoint declares none for it")
                continue
            if r["state"] != "served":
                problems.append(f"{phase}: {name} state={r['state']!r} reason={r.get('reason')!r}")
            if r["contract"] != contract_for[family]:
                problems.append(f"{phase}: {name} contract={r['contract']!r} != {contract_for[family]!r}")
            if r["policy"] != f"{family}:{mode}":
                problems.append(f"{phase}: {name} policy={r['policy']!r} != declared {family}:{mode}")
            # The (symbol, decoder) pair, not each half alone: the streamed FP8
            # route reports the GEMV pair wherever the lane prepared and the
            # kernel-decoded tile under the stock GEMM above the lane's max M
            # (``fp8_gemv.census_expected`` owns the sets), and a half-wise
            # comparison would read either half as a refusal on every module
            # that legitimately took the other launch.
            want = _expected(family, CENSUS_PHASE_REGIMES[phase])
            if ((r["symbol"], r.get("decoder")) not in want
                    and not (args.allow_fallback_decoder and r["symbol"] == symbol_for[family])):
                problems.append(
                    f"{phase}: {name} (symbol, decoder)={(r['symbol'], r.get('decoder'))!r} "
                    f"not in {sorted(want)!r}; without --allow-fallback-decoder a serve must "
                    "report a pair its route owns")
        missing = sorted(set(declared) - set(owner.values()))
        if missing:
            problems.append(
                f"{phase}: {len(missing)} declared Tessera modules report no route, e.g. {missing[:3]}")
        if args.expect_modules is not None and len(tess) != args.expect_modules:
            problems.append(
                f"{phase}: {len(tess)} Tessera modules, the checkpoint declares {args.expect_modules}")
    if phases[batch_phase] and phases[decode_phase]:
        shapes = [(phases[batch_phase][n].get("shape", ""), phases[decode_phase][n].get("shape", ""))
                  for n in phases[batch_phase] if n in phases[decode_phase]]
        if args.compiled:
            # A compiled forward writes the record from the trace, where the
            # token dimension is symbolic: ``route_shape`` spells it ``M*`` and
            # the record cannot tell the regimes apart.  What the census can
            # attest here is dispatch (every module reached its route in both
            # steps) and the polymorphic form itself; a concrete M in a compiled
            # record would mean the route specialised the batch.
            bad = [p for p, d in shapes if not (p.startswith("M*:") and d.startswith("M*:"))]
            if bad:
                problems.append(f"compiled records must be shape-polymorphic (M*); got {bad[:3]}")
        elif all(p == d for p, d in shapes):
            problems.append(
                f"{batch_phase} and {decode_phase} records carry the same shape; only one "
                "shape was exercised")

    # LANE ENGAGEMENT.  The per-module check above is a check on AGREEMENT, and
    # the decode regime legitimately admits both the GEMV pair and the
    # materialised one -- so a serve in which the lane prepared for NOTHING
    # passes it module by module.  This asks the question that cannot: did the
    # lane this arm requested take any units at all (issue #104)?  Emitted
    # unconditionally so a receipt written without --require-lane still carries
    # the decoder counts a gate would need.
    tessera_by_phase = {
        phase: {n: r for n, r in recs.items()
                if str(r.get("policy", "")).startswith(prefixes)}
        for phase, recs in phases.items()}
    engagement, engagement_problems = lane_engagement(
        tessera_by_phase, required_lanes=required_lanes, lane_decoders=lane_decoders or None,
        refusals_by_phase={phase: refusals for phase in phases})
    engagement["declared_by_artifact"] = manifest_lanes
    problems.extend(engagement_problems)

    # WHAT THE CONTRACT SAYS THIS SERVE EXECUTES, against what it executed.
    # ``lane_eligibility`` cells publish ``executes`` since schema v4 (#111), a
    # value DERIVED from the dispatch table -- which proves the document agrees
    # with the code.  Only a serve proves the code agrees with the machine, so
    # the join is made here, per module, in both phases.  A compiled record
    # stamps the two launches as one ``a+b`` pair because one graph serves
    # every M, and no cell publishes that form: the check is eager-only and
    # says so rather than reading a traced record as a disagreement.
    if args.compiled:
        agreement, agreement_problems = (
            {"schema": "tessera.cell-launch-agreement/1", "agrees": None,
             "skipped": "a compiled record stamps one shape-polymorphic pair for both "
                        "launches; cells publish launches, so there is nothing to join"},
            [])
    else:
        agreement, agreement_problems = cell_launch_agreement(
            tessera_by_phase, cells=load_serving_contract()["lane_eligibility"]["cells"],
            phase_regimes=CENSUS_PHASE_REGIMES,
            platform=f"sm_{torch.cuda.get_device_capability(0)[0]}"
                     f"{torch.cuda.get_device_capability(0)[1]}",
            rungs_by_module=declared_rungs, families_by_route=PAYLOAD_FAMILY_BY_ROUTE)
    problems.extend(agreement_problems)

    receipt = {
        "schema": "tessera.serving.route_census/1",
        "checkpoint": os.path.abspath(args.model),
        "quant_method": qc.get("quant_method"),
        "compiled": bool(args.compiled),
        "tessera_config_groups": len(tessera_groups),
        "declared_names_mapped_to_module_space": name_map is not None,
        "declared_name_mapping": name_map,
        "prompt_tokens": len(ids),
        "generated_text": generated,
        "declared_families": dict(sorted(collections.Counter(declared.values()).items())),
        "env": {TESSERA_MODE_ENV: mode or None,
                "VLLM_DISABLED_KERNELS": os.environ.get("VLLM_DISABLED_KERNELS")},
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__,
                     "tessera": getattr(tessera, "__version__", None),
                     "tessera_serving": getattr(serving, "__version__", None),
                     "tessera_commit": args.tessera_commit or _git_head(
                         os.path.dirname(os.path.dirname(os.path.dirname(
                             os.path.abspath(tessera.__file__))))),
                     "python": platform.python_version()},
        "cell_launch_agreement": agreement,
        "device": {"name": torch.cuda.get_device_name(0),
                   "capability": list(torch.cuda.get_device_capability(0))},
        "elapsed_s": round(time.time() - t0, 1),
        "histogram": histogram,
        "lane_engagement": engagement,
        "lane_refusals": refusals,
        "records": phases,
        # Which declared target each record was joined to.  For a dense
        # module that is the identity; for an expert stack it names the
        # declaration the RoutedExperts child served, so the join is a
        # value a reader can check rather than a rule they must trust.
        "record_owner": record_owner,
        "problems": problems,
        "verdict": "served" if not problems else "REFUSED",
    }
    with open(args.out, "w") as fh:
        json.dump(receipt, fh, indent=1, sort_keys=True)
    print(json.dumps({k: receipt[k] for k in ("verdict", "histogram", "lane_engagement",
                                              "env", "device", "elapsed_s")},
                     indent=1))
    for p in problems:
        print("PROBLEM:", p)
    print(f"-> {args.out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
