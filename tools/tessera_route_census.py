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
covering its ``(platform, family, structure, regime, residency, rung)`` and
explicit runtime image/execution mode, the
``cell_launch_agreement`` block lands in the receipt, and a disagreement is a
refusal.  A module at a rung no cell covers is ``unattested``, which is the
only negative signal a closed-world table has and is not a failure. Compiled
dense agreement is unsupported: its trace combines launches as ``a+b`` for a
graph serving every M. A compiled routed-MoE record may agree only when its
cell and observation name a single launch. Unsupported observations are counted
as unattested and retain their exact records.

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

AND ONE RANK IS NOT A WORLD.  ``LLM.apply_model`` returns one result per
worker and this tool read ``[0]``, so a census could only ever describe rank 0
-- and a receipt of rank 0 alone reads identically at world size 1 and at world
size 8, which is why no Tessera artifact has a per-rank route histogram and no
contract cell can name a world size from a receipt.  The topology arguments
(``--tensor-parallel-size``, ``--distributed-executor-backend {mp,ray}``,
``--nnodes``, ``--node-rank``, ``--master-addr``, ``--master-port``) are
declared by ``tessera.serving.topology``, validated before the first model load
and passed to ``LLM(...)`` unchanged; the per-module checks then run on EVERY
rank, and ``census.join_rank_histograms`` sums the ranks into the ``histogram``
the attestation reads.  Above one rank the receipt gains two blocks -- a
``topology`` block holding what was asked for beside the world that answered,
and a ``ranks`` list of ``tessera.rank-census/1`` records naming each rank's id,
world size, node, device, platform token and runtime image.  Both are ADDITIVE
and appear only when the world has more than one rank: at one rank the joined
record IS the receipt, byte for byte what this tool wrote before the arguments
existed, and the blocks' absence is what says world size 1.

usage::

    tessera_route_census.py <checkpoint-dir> <out.json> \
        --runtime-image <repository@sha256:digest> \
        [--expect-modules N] [--prompt-tokens 64] [--gpu-memory-utilization 0.3] \
        [--tensor-parallel-size N] [--distributed-executor-backend {mp,ray}] \
        [--nnodes N] [--node-rank 0] [--master-addr HOST] [--master-port PORT]

Above one box, ``experiments/tessera_plugin_served_tp.sh`` is the driver: it
starts a ray head here and a ray worker on the other box, checks both hold the
same checkout and the same pinned image, and runs this tool at ``--tensor-
parallel-size 2 --distributed-executor-backend ray``.

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

Run it inside the serving image with the plugin installed, through
``experiments/tessera_plugin_run.sh`` (the same container the KL dumps ran in);
``TESSERA_SERVE_MODE`` selects the residency exactly as it does for ``vllm
serve``.  Through that wrapper and no other: ``--runtime-image`` is checked
against the reference the launcher resolved from docker's own ``RepoDigests``
and declared into the container, and a run the launcher declared no image for
refuses before the first model load (issue #132).  A declaration, not an
attestation: a host process can export the same pair by hand, which is why the
receipt records the mechanism beside the name.
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
CHECKPOINT_SIDECAR_NAMES = ("config.json", "tessera_serving_manifest.json")


def checkpoint_sidecar_hashes(checkpoint, *, expected=None):
    """Exact sidecar bytes observed by this process, independent of mount names.

    Config is required by every census. A generic census may legitimately have
    no serving manifest; publish that absence rather than inventing a digest.
    Campaign orchestration holds the assembled artifact unchanged through the
    serve. These hashes do not replace the assembly's tensor/wire audit.
    """
    from pathlib import Path
    from tessera.serving_parts import sha256_file
    checkpoint = Path(checkpoint)
    config, manifest = (checkpoint / name for name in CHECKPOINT_SIDECAR_NAMES)
    observed = {config.name: sha256_file(config),
                manifest.name: sha256_file(manifest) if manifest.is_file() else None}
    if expected is not None and observed != expected:
        raise ValueError("checkpoint sidecars changed during census; no served receipt is valid")
    return observed


def census(model):
    """Runs inside the worker: every module carrying a route record."""
    from tessera.serving.telemetry import read_route
    out = {}
    for name, mod in model.named_modules():
        rec = read_route(mod)
        if rec is not None:
            out[name] = rec
    return out


def rank_identity(model):
    """Runs inside the worker: which rank read the records beside this, and where.

    A route record says what a module executed; it never says which rank's
    shard of that module executed it, on which box, in which image.  At one
    rank that omission is invisible -- there is one of everything -- and above
    one rank it is the whole question, so the identity is read in the worker
    process rather than assumed by the driver.

    The rank and the world size come from vLLM's own world group, which is the
    table the engine itself dispatches on; ``torch.distributed`` is the
    fallback for a build that exposes no group.  The image is the launcher's
    declaration as THIS rank's environment carries it (issue #132): a two-box
    serve can straddle two images, and the head's copy is evidence about the
    head alone.
    """
    import os
    import socket

    import torch
    from tessera.serving.backend import platform_of_this_process
    from tessera.serving.runtime_image import CENSUS_IMAGE_ENV

    rank = local_rank = None
    world_size = None
    try:
        from vllm.distributed.parallel_state import get_world_group
        group = get_world_group()
        rank, world_size = int(group.rank), int(group.world_size)
        local_rank = int(getattr(group, "local_rank", 0))
    except Exception:  # noqa: BLE001 -- the fallback below is the same question
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank, world_size = int(dist.get_rank()), int(dist.get_world_size())
        except Exception:  # noqa: BLE001
            pass
    if rank is None or world_size is None:
        rank, world_size = 0, 1
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "node": socket.gethostname(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform_token": platform_of_this_process(torch),
        "runtime_image": os.environ.get(CENSUS_IMAGE_ENV),
    }


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
    from tessera.serving.weights_mapper import module_name_mapper
    unstacked = module_name_mapper(mapper)
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
    the child's name. A record under two declared targets is ambiguous and
    reported; this join does not require a declaration to own only one record.
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


def declared_rung(scheme):
    """One cell rung only when every declared group and role has that rung."""
    if scheme.get("structure") == "routed_moe":
        groups = list((scheme.get("groups") or {}).values())
    else:
        groups = [scheme]
    rates = []
    for group in groups:
        value = group.get("q256")
        values = value if isinstance(value, (list, tuple)) else [value]
        if not values or any(v is None for v in values):
            return None
        rates.extend(int(v) for v in values)
    distinct = set(rates)
    return next(iter(distinct)) if len(distinct) == 1 else None


def parse_eager_shape(value):
    """Read telemetry.route_shape's canonical concrete M:N:K spelling.

    The grammar lives in ``scheme`` beside ``regime_of_m``, because the census
    tool and the shared cell matcher must read a shape the same way or one of
    them attests a regime the other would refuse.  Kept here as a name its
    existing callers already import.
    """
    from tessera.serving.scheme import parse_eager_shape as _parse
    return _parse(value)


def phase_shape_problems(records_by_phase, *, phase_regimes, compiled=False,
                         require_each_owner=False):
    """Every eager record's own shape against the regime its phase declares.

    Callers requiring owner coverage first join records into owner space; that
    arm additionally refuses a missing or unchanged shape at any owner.  Both
    arms then read EACH record, because a phase attests the forward that ran
    and not the one it was named after: an aggregate "the two phases differ
    somewhere" passes a census in which one module kept its prefill record
    while another moved, and passes an eight-row forward filed under the decode
    phase (#207).  The M -> regime rule is ``scheme.eager_regime_problem``, the
    same one ``census.cell_launch_agreement`` applies to a covered record.
    """
    from tessera.serving.scheme import eager_regime_problem
    batch_phase = next(p for p, regime in phase_regimes.items() if regime == "batch")
    decode_phase = next(p for p, regime in phase_regimes.items() if regime == "decode")
    batch, decode = records_by_phase[batch_phase], records_by_phase[decode_phase]
    if not batch or not decode:
        return ["both driven phases need shape evidence"] if require_each_owner else []
    shapes = [(name, batch[name].get("shape", ""), decode[name].get("shape", ""))
              for name in batch if name in decode]
    if compiled:
        # Trace-time M is symbolic, so this proves polymorphic dispatch, not
        # two concrete shapes. Campaign eager coverage never uses this arm.
        bad = [p for _, p, d in shapes
               if not (str(p).startswith("M*:") and str(d).startswith("M*:"))]
        return ([f"compiled records must be shape-polymorphic (M*); got {bad[:3]}"]
                if bad else [])
    problems = []
    if require_each_owner:
        missing = sorted(set(batch) ^ set(decode))
        bad = [name for name, p, d in shapes
               if not isinstance(p, str) or not p or not isinstance(d, str) or not d or p == d]
        problems = ([f"each owner needs distinct nonempty eager shapes in both driven phases; "
                     f"missing={missing}, unchanged/missing shape={bad}"] if missing or bad else [])
    for phase, records in sorted(records_by_phase.items()):
        for owner, record in sorted(records.items()):
            why = eager_regime_problem(record.get("shape"), phase_regimes.get(phase))
            if why is not None:
                problems.append(f"{phase} {owner}: {why}")
    return problems


def _capability_or_none(torch):
    """The device's compute capability, or ``[]`` where it has none.

    A DIAGNOSTIC, not a key.  On HIP ``get_device_capability`` answers with
    the GCN major/minor -- gfx1201 answers ``(12, 0)``, the tuple NVIDIA's
    sm_120 answers with -- so this value can never again be the thing a cell
    is joined on.  It stays in the receipt because it is what an NVIDIA
    reader has always looked at, and it is allowed to be absent.
    """
    try:
        return list(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001 -- a receipt field never breaks a receipt
        return []


def all_structure_agreement(records_by_phase, *, cells, phase_regimes, platform,
                            declared_rungs, record_owners, families_by_route,
                            runtime_image=None, execution_mode=None):
    """Check each observed structure against its own cells, using declared owners.

    This aggregates existing per-structure checks; it publishes no new cells.
    Missing ownership, mixed rungs, and structures with no cells remain
    unattested. Exact recorded symbols remain in the census records.
    """
    from tessera.serving.census import (
        CELL_AGREEMENT_SCHEMA, STRUCTURE_BY_RECORD_KIND, cell_launch_agreement)
    from tessera.serving.scheme import moe_census_symbol_base as census_symbol_base

    structures = sorted({STRUCTURE_BY_RECORD_KIND.get(str(record.get("kind")), "unknown")
                         for records in records_by_phase.values()
                         for record in records.values()})
    runtime = {"image": runtime_image, "execution_mode": execution_mode}
    blocks, problems = {}, []
    for structure in structures:
        phases, verdicts, unsupported_reasons = {}, [], set()
        for phase, records in sorted(records_by_phase.items()):
            selected = {name: record for name, record in records.items()
                        if STRUCTURE_BY_RECORD_KIND.get(str(record.get("kind")), "unknown")
                        == structure}
            owners = record_owners.get(phase, {})
            rungs = {name: declared_rungs.get(owners.get(name)) for name in selected}
            block, failures = cell_launch_agreement(
                {phase: selected}, cells=cells, phase_regimes=phase_regimes,
                platform=platform, structure=structure, rungs_by_module=rungs,
                families_by_route=families_by_route,
                runtime_image=runtime_image, execution_mode=execution_mode,
                symbol_alias=census_symbol_base if structure == "routed_moe" else None)
            phases.update(block["phases"])
            verdicts.append(block["agrees"])
            problems.extend(failures)
            if block.get("unsupported_reason"):
                unsupported_reasons.add(block["unsupported_reason"])
        agrees = (False if False in verdicts else True if True in verdicts else None)
        blocks[structure] = {"schema": CELL_AGREEMENT_SCHEMA, "platform": platform,
                             "structure": structure, "runtime": dict(runtime),
                             "phases": phases, "agrees": agrees}
        if unsupported_reasons:
            blocks[structure]["unsupported_reasons"] = sorted(unsupported_reasons)
    verdicts = [block["agrees"] for block in blocks.values()]
    return {"schema": "tessera.cell-launch-agreement.by-structure/1", "platform": platform,
            "runtime": runtime, "structures": blocks,
            "agrees": False if False in verdicts else True if True in verdicts else None}, problems


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


def parse_args(argv=None, env=None):
    """Resolve the explicit runtime context before importing a serving runtime."""
    from tessera.serving.topology import add_topology_arguments, validate_topology_arguments
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--runtime-image", required=True,
                    help="exact repository@sha256:digest; cross-checked against the "
                         "reference the container launcher resolved from docker's "
                         "RepoDigests and DECLARED into this container, so it binds cell "
                         "agreement to the image the launcher named rather than to a "
                         "string that was typed. A launcher declaration, never a check "
                         "the container itself makes: a host process can export the same "
                         "pair by hand, and the receipt records which mechanism named the "
                         "image")
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
    add_topology_arguments(ap)
    ap.add_argument("--tessera-commit", default=None,
                    help="the host's `git rev-parse HEAD` for the Tessera checkout under test; "
                         "inside a container a worktree's .git pointer resolves nowhere and the "
                         "receipt would carry None")
    args = ap.parse_args(argv)
    # BEFORE the first model load, like every other check in this function: a
    # topology mistake found after two 85-160 s loads is one found at the cost
    # of the run.
    validate_topology_arguments(ap, args)
    from tessera.serving.contract import require_runtime_image
    from tessera.serving.runtime_image import RuntimeImageError, declared_reference

    try:
        args.runtime_image = require_runtime_image(args.runtime_image, "--runtime-image")
    except ValueError as exc:
        ap.error(str(exc))
    # THE IMAGE IS A JOIN KEY, NOT A LABEL (issue #132).  It scopes every cell
    # this census resolves, so an operator's string that named other bytes
    # would produce a `covered` verdict for a runtime nobody measured, in a
    # receipt shaped exactly like a correct one.  Nothing inside a container
    # can ask the daemon what it is running, so the launcher transcribes
    # `docker image inspect`'s RepoDigests into the environment and this is
    # where the claim meets that table.  Before the first model load, and with
    # no way to opt out: a stamped `operator_asserted` receipt would be the
    # same defect wearing a field name -- and for the same reason this one is
    # published as the launcher's DECLARATION, which is all the mechanism is.
    try:
        args.runtime_image_declaration = declared_reference(args.runtime_image, env=env)
    except RuntimeImageError as exc:
        ap.error(f"--runtime-image {args.runtime_image}: {exc}")
    args.execution_mode = "compiled" if args.compiled else "eager"
    return args


def main() -> int:
    args = parse_args()

    # The census function must run in the process that holds the model.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    import tessera
    import tessera.serving as serving
    from tessera.serving import bf16_route, fp8_gemv, fp8_route, moe_route, nvfp4_route
    from tessera.serving.census import (
        join_rank_histograms, lane_engagement, phase_histogram, rank_census_record)
    from tessera.serving.contract import (
        CENSUS_PHASE_REGIMES, PAYLOAD_FAMILY_BY_ROUTE, load_serving_contract)
    from tessera.serving.lane import TESSERA_MODE_ENV
    from tessera.serving.scheme import (
        ROUTES, TESSERA_BF16, TESSERA_FAMILIES, TESSERA_FP8, TESSERA_NVFP4)
    from tessera.serving.telemetry import DECODER_NATIVE_SPAN2, DECODER_TORCH_WINDOW
    from tessera.serving.backend import platform_of_this_process
    from tessera.serving.topology import topology_kwargs, topology_record

    # THE PLATFORM THIS SERVE RAN ON, read the one honest way (#452/#457).
    # This used to be spelt ``f"sm_{capability[0]}{capability[1]}"`` at the
    # join below.  On a ROCm torch ``get_device_capability`` answers with the
    # GCN major/minor, and gfx1201 answers ``(12, 0)`` -- so an AMD census
    # would have minted ``sm_120``, joined every record to whatever an NVIDIA
    # sm_120 cell published, and reported agreement or disagreement about a
    # platform the serve was not on.  ``gcnArchName`` is the key
    # ``lane_eligibility.platforms`` is written in, and the module that reads
    # it is the module the build and the certification harness read it from.
    # Resolved BEFORE the first model load: it is a join key of every cell
    # this receipt resolves, and a receipt tool must not fail at a per-module
    # lookup with two loaded models behind it.
    served_platform = platform_of_this_process(torch)
    if served_platform is None:
        raise SystemExit(
            "this box names no platform token (no CUDA/HIP device answered), so no "
            "lane_eligibility cell can be joined to what it serves. A census without a "
            "platform key is a receipt about nothing.")

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
    # PER ``(platform, family)``: on a platform whose contract entry executes
    # null for a family, nothing of that family loads, so the expectation is
    # the empty set and any record is a disagreement (``census.platform_
    # expectation``).  On sm_121 and on any platform the contract has not
    # reached, these are the sets they always were.
    fp8_expected = fp8_gemv.census_expected(compiled=args.compiled, platform=served_platform)
    bf16_expected = bf16_route.census_expected(compiled=args.compiled, platform=served_platform)
    # A ROUTED EXPERT STACK IS NOT ITS FAMILY'S DENSE ROUTE.  The stack serves
    # under the same family (``TESSERA_FP8``, same wire, same activation
    # contract) and a different dispatch: one materialised launch through
    # vLLM's modular fused-MoE kernel, in both regimes, with no GEMV lane.
    # Comparing it against the dense pair set reads a correct serve as a
    # refusal on every stack, so the expectation is taken from the route that
    # owns the dispatch -- ``moe_route.census_expected``, which also says why
    # its symbol is compared without the runtime's backend suffix and why no
    # contract cell publishes it yet.
    moe_expected = moe_route.census_expected(compiled=args.compiled, platform=served_platform)

    def _expected(family, regime, kind):
        if kind == "moe":
            return moe_expected[regime]
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
    sidecars = checkpoint_sidecar_hashes(args.model)
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
    declared_rungs = {t: declared_rung(g["scheme"])
                      for g in tessera_groups.values() for t in g.get("targets", [])}

    problems = []
    t0 = time.time()
    # THE TOPOLOGY REACHES THE ENGINE UNCHANGED.  At the default single-process
    # topology these kwargs are ``tensor_parallel_size=1`` and nothing else --
    # the engine's own default -- so a command line written before the topology
    # group existed builds the same engine it always did.
    llm = LLM(model=args.model, enforce_eager=not args.compiled, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=0,
              **topology_kwargs(args))

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

    # EVERY RANK, NOT THE HEAD.  ``apply_model`` returns one result per worker
    # and this tool took ``[0]``, so a census could only ever describe rank 0 --
    # and a receipt of rank 0 alone reads identically at world size 1 and at
    # world size 8.  The list is kept whole from here down; the head's element
    # is still what the single-rank receipt publishes, byte for byte.
    phases_by_rank = {}
    # One forward over M = len(ids) rows, sample one token, stop.
    outs = llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0))
    phases_by_rank[batch_phase] = llm.apply_model(census)
    # Decode steps follow; the last forward is one row wide.
    outs = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0))
    phases_by_rank[decode_phase] = llm.apply_model(census)
    generated = outs[0].outputs[0].text
    # Load facts, so once and after the forwards: which modules' lane refused
    # to prepare, and why.  Same for every phase by construction.
    refusals_by_rank = llm.apply_model(lane_refusals)
    identities = llm.apply_model(rank_identity)
    world_size = len(identities)
    ranks_seen = sorted(int(identity["rank"]) for identity in identities)
    if ranks_seen != list(range(world_size)):
        raise SystemExit(
            f"{world_size} worker(s) answered and they call themselves {ranks_seen}; a joined "
            "census must cover each rank of the world exactly once")
    declared_world = {int(identity["world_size"]) for identity in identities}
    if declared_world != {world_size}:
        raise SystemExit(
            f"{world_size} worker(s) answered but they report world size(s) {sorted(declared_world)}; "
            "a receipt must not name a world only part of it served")
    requested_world = topology_kwargs(args)["tensor_parallel_size"]
    if world_size != requested_world:
        raise SystemExit(
            f"--tensor-parallel-size {requested_world} was asked for and {world_size} worker(s) "
            "answered; pipeline, data and context parallel stay 1 here, so the world this census "
            "observed must be the world it requested")
    # Ordered by rank, never by the order the executor happened to answer in.
    order = sorted(range(world_size), key=lambda i: int(identities[i]["rank"]))
    identities = [identities[i] for i in order]
    refusals_by_rank = [refusals_by_rank[i] for i in order]
    phases_by_rank = {phase: [per_rank[i] for i in order]
                      for phase, per_rank in phases_by_rank.items()}
    phases = {phase: per_rank[0] for phase, per_rank in phases_by_rank.items()}
    refusals = refusals_by_rank[0]

    mode = os.environ.get(TESSERA_MODE_ENV, "")
    prefixes = tuple(f"{family}:" for family in TESSERA_FAMILIES)
    # THE PER-MODULE CHECKS RUN ON EVERY RANK.  Each rank serves its own shard
    # of every module and writes its own route record, so a check run on the
    # head alone would pass a world in which rank 1 fell back on every unit.
    # At one rank the loop below runs once and every problem string it can
    # write is the string it wrote before -- the rank tag appears only above
    # one rank, because a receipt that is the same observation must be the
    # same bytes.
    histogram_by_rank = []
    record_owner_by_rank = []
    tessera_by_rank = []
    for rank in range(world_size):
        tag = "" if world_size == 1 else f"rank {rank} "
        rank_histogram = {}
        rank_owner = {}
        rank_tessera = {}
        for phase, per_rank in phases_by_rank.items():
            recs = per_rank[rank]
            tess = {n: r for n, r in recs.items() if str(r.get("policy", "")).startswith(prefixes)}
            other = {n: r for n, r in recs.items() if n not in tess}
            rank_tessera[phase] = tess
            rank_histogram[phase] = phase_histogram(
                tess, regime=CENSUS_PHASE_REGIMES[phase], other_route_modules=len(other))
            if not tess:
                problems.append(f"{tag}{phase}: no module reports a Tessera route")
            owner, join_problems = join_records_to_declared(tess, declared)
            rank_owner[phase] = owner
            problems.extend(f"{tag}{phase}: {m}" for m in join_problems)
            for name, r in tess.items():
                family = declared.get(owner.get(name, name))
                if family is None:
                    problems.append(
                        f"{tag}{phase}: {name} took a Tessera route but the checkpoint declares none for it")
                    continue
                if r["state"] != "served":
                    problems.append(f"{tag}{phase}: {name} state={r['state']!r} reason={r.get('reason')!r}")
                if r["contract"] != contract_for[family]:
                    problems.append(f"{tag}{phase}: {name} contract={r['contract']!r} != {contract_for[family]!r}")
                if r["policy"] != f"{family}:{mode}":
                    problems.append(f"{tag}{phase}: {name} policy={r['policy']!r} != declared {family}:{mode}")
                # The (symbol, decoder) pair, not each half alone: the streamed FP8
                # route reports the GEMV pair wherever the lane prepared and the
                # kernel-decoded tile under the stock GEMM above the lane's max M
                # (``fp8_gemv.census_expected`` owns the sets), and a half-wise
                # comparison would read either half as a refusal on every module
                # that legitimately took the other launch.
                want = _expected(family, CENSUS_PHASE_REGIMES[phase], r.get("kind"))
                # The expert route's symbol carries the backend the RUNTIME picked
                # (``...modular_kernel:TRITON``), which no expectation of ours may
                # pin; the entry point is what this compares and the histogram
                # above keeps every exact string, backend and all.
                got_symbol = (moe_route.census_symbol_base(r["symbol"])
                              if r.get("kind") == "moe" else r["symbol"])
                if ((got_symbol, r.get("decoder")) not in want
                        and not (args.allow_fallback_decoder and r["symbol"] == symbol_for[family])):
                    problems.append(
                        f"{tag}{phase}: {name} (symbol, decoder)={(r['symbol'], r.get('decoder'))!r} "
                        f"not in {sorted(want)!r}; without --allow-fallback-decoder a serve must "
                        "report a pair its route owns")
            missing = sorted(set(declared) - set(owner.values()))
            if missing:
                problems.append(
                    f"{tag}{phase}: {len(missing)} declared Tessera modules report no route, e.g. {missing[:3]}")
            # PER RANK, NOT OVER THE WORLD.  ``--expect-modules`` names what the
            # CHECKPOINT declares, and every rank builds a module for every
            # declared target -- it holds a shard of it.  Comparing the joined
            # count would refuse a correct two-rank serve for serving twice.
            if args.expect_modules is not None and len(tess) != args.expect_modules:
                problems.append(
                    f"{tag}{phase}: {len(tess)} Tessera modules, the checkpoint declares {args.expect_modules}")
        histogram_by_rank.append(rank_histogram)
        record_owner_by_rank.append(rank_owner)
        tessera_by_rank.append(rank_tessera)
    # THE JOINED HISTOGRAM IS THE SUM OVER RANKS, and it is what the
    # attestation reads.  At one rank it is that rank's histogram unchanged.
    histogram = join_rank_histograms(histogram_by_rank)
    record_owner = record_owner_by_rank[0]
    # THE TESSERA RECORDS ARE WHAT THIS RECEIPT ATTESTS, so they are what the
    # shape and agreement checks below read: a record from another quant
    # method's route is observed and counted, never used as evidence for a
    # Tessera regime.
    # ONE NAMESPACE OVER THE WHOLE WORLD.  Every rank names its modules the
    # same, so merging the ranks' records under their own names would count one
    # module once however many ranks served it -- the engagement and agreement
    # blocks would then read identically at every world size.  Above one rank
    # the name carries the rank that observed it; at one rank it is the module
    # name it always was, and the blocks below are the bytes they always were.
    def _qualified(rank, name):
        return name if world_size == 1 else f"rank{rank}/{name}"

    tessera_by_phase = {
        phase: {_qualified(rank, n): r
                for rank, by_phase in enumerate(tessera_by_rank)
                for n, r in by_phase[phase].items()}
        for phase in phases_by_rank}
    record_owner_world = {
        phase: {_qualified(rank, n): owner
                for rank, by_phase in enumerate(record_owner_by_rank)
                for n, owner in by_phase[phase].items()}
        for phase in phases_by_rank}
    problems.extend(phase_shape_problems(
        tessera_by_phase, phase_regimes=CENSUS_PHASE_REGIMES, compiled=args.compiled))

    # LANE ENGAGEMENT.  The per-module check above is a check on AGREEMENT, and
    # the decode regime legitimately admits both the GEMV pair and the
    # materialised one -- so a serve in which the lane prepared for NOTHING
    # passes it module by module.  This asks the question that cannot: did the
    # lane this arm requested take any units at all (issue #104)?  Emitted
    # unconditionally so a receipt written without --require-lane still carries
    # the decoder counts a gate would need.
    refusals_world = {_qualified(rank, name): reason
                      for rank, by_module in enumerate(refusals_by_rank)
                      for name, reason in by_module.items()}
    engagement, engagement_problems = lane_engagement(
        tessera_by_phase, required_lanes=required_lanes, lane_decoders=lane_decoders or None,
        refusals_by_phase={phase: refusals_world for phase in phases})
    engagement["declared_by_artifact"] = manifest_lanes
    problems.extend(engagement_problems)

    # WHAT THE CONTRACT SAYS THIS SERVE EXECUTES, against what it executed.
    # ``lane_eligibility`` cells publish ``executes`` since schema v4 (#111), a
    # value DERIVED from the dispatch table -- which proves the document agrees
    # with the code.  Only a serve proves the code agrees with the machine, so
    # the join is made here, per module, in both phases, under the actual
    # image and execution mode. Compiled dense records retain an explicit
    # unsupported result; a routed single-launch observation can be checked.
    agreement, agreement_problems = all_structure_agreement(
        tessera_by_phase, cells=load_serving_contract()["lane_eligibility"]["cells"],
        phase_regimes=CENSUS_PHASE_REGIMES,
        platform=served_platform,
        declared_rungs=declared_rungs, record_owners=record_owner_world,
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=args.runtime_image, execution_mode=args.execution_mode)
    problems.extend(agreement_problems)
    # The controller holds the checked artifact immutable; verify that seal
    # again after both forwards, before publishing any served receipt.
    checkpoint_sidecar_hashes(args.model, expected=sidecars)

    receipt = {
        "schema": "tessera.serving.route_census/2",
        "checkpoint": os.path.abspath(args.model),
        "quant_method": qc.get("quant_method"),
        "compiled": bool(args.compiled),
        "runtime": {"image": args.runtime_image, "execution_mode": args.execution_mode},
        # WHERE THAT SCOPE CAME FROM.  The image above is a join key of every
        # cell this receipt resolves; this says which mechanism established it
        # and carries the launcher's record verbatim, so a reader can redo the
        # join rather than trust the name.  Its ABSENCE is the discriminator
        # for a receipt written before #132, when the value was whatever the
        # operator typed.
        "runtime_image_declaration": args.runtime_image_declaration,
        "checkpoint_sidecars": sidecars,
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
        # The token is the KEY; the capability is kept beside it as a
        # diagnostic and is no longer the key, because on HIP it is not one
        # (gfx1201 and sm_120 both answer (12, 0)).
        "device": {"name": torch.cuda.get_device_name(0),
                   "platform_token": served_platform,
                   "capability": list(_capability_or_none(torch))},
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
    # THE WORLD, WRITTEN ONLY WHERE THERE IS ONE.  At a single rank the joined
    # record IS the receipt: a ``ranks`` list would restate it once and a
    # ``topology`` block would state the engine's own default, and both would
    # change the bytes of every single-rank receipt this tool has ever written
    # -- receipts that are quoted, diffed and re-read.  So the blocks appear
    # exactly when they carry something the joined record cannot: more than one
    # rank.  Their ABSENCE is the discriminator, and it means world size 1.
    if world_size > 1:
        topology = topology_record(args)
        topology["observed_world_size"] = world_size
        receipt["topology"] = topology
        receipt["ranks"] = [
            rank_census_record(
                rank=int(identity["rank"]), world_size=world_size,
                node=identity["node"], platform_token=identity["platform_token"],
                runtime_image=identity.get("runtime_image"), device=identity.get("device"),
                local_rank=identity.get("local_rank"),
                histogram=histogram_by_rank[index],
                lane_refusals=refusals_by_rank[index],
                records={phase: per_rank[index] for phase, per_rank in phases_by_rank.items()})
            for index, identity in enumerate(identities)]

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
