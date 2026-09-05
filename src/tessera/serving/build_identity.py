"""Which compiled build served this arm, recorded so a later A/B can refuse.

WHY THIS EXISTS (issue #30).  ``compile_identity`` folds the plugin's residency
into vLLM's compile-cache *key*, so two modes cannot share one cached forward.
This module is its counterpart on the way out: it reads, off a finished serve,
which compiled artifact actually ran, and writes it beside the KL dump so two
measurements can be compared as measurements of the weights rather than of the
compiler.

The measurement that forces it (``docs/measurements/serving-compile-divergence-
2026-09-02.md``): a compiled vLLM artifact **replayed** is bit-identical
(0.000000 / 100%); the same graph **rebuilt** is not (0.017117 / 95.65%, with
120 of 196 autotuned Triton kernels choosing a different ``XBLOCK``/
``num_warps`` the second time).  A cross-arm KL of that size is an ordinary
result on this box, so without a record of the build, a rebuild and a
regression are the same number.

THE TRAP, AND WHY THE OBVIOUS STAMP WOULD BE WORSE THAN NOTHING.  Issue #30
proposed stamping ``grep -oE 'torch_aot_compile/[0-9a-f]+'``.  That key does
**not** identify the build: vLLM keys the cache by its own *inputs*, and the
two divergent builds above sit under one key ``15957ad9…`` with byte-identical
``cache_key_factors.json``.  A stamp that matched on the key would certify a
rebuild as a replay -- provenance that reads like a guarantee and is not one.
What identifies the build is the *content* of that cache slot: the autotune
choices inductor wrote into it.  So the fingerprint here is a digest over
``inductor_cache/**.best_config`` (with ``time_taken_ms`` and
``triton_cache_hash`` removed -- those record the benchmark, not the choice; 74
of the 196 real records differed only there).  The backbone
``computation_graph.py`` differed between the two builds too
(``fused_add_rms_norm.maybe_inplace`` 42 → 0), but it is recorded as
provenance rather than fingerprinted: vLLM writes that log line only when it
compiles, so a replay of one build would otherwise not fingerprint as that
build.  Issue #29 separately established the 42 → 0 itself is a dump
artefact -- every graph in the campaign census names the same 56 + 57 call
sites with only the overload suffix varying, which the functionalization
pass rewrites -- so there was never a second build difference here to
fingerprint; see ``experiments/compile_build_forensics.py``.

WHICH PROGRAM RAN, NOT ONLY WHICH BUILD (issue #16).  On this runtime an
eager arm and a compiled arm are not two ways of running one program.  vLLM
0.28 flips two dispatch defaults together on "is inductor going to run":
``custom_ops`` gains the base mode ``"none"`` instead of ``"all"``
(``vllm/config/vllm.py:1392-1399``), so every ``CustomOp`` runs its torch
decomposition rather than its CUDA kernel; and ``ir_op_priority`` becomes
``["native"]`` instead of ``["vllm_c", "native"]``
(``vllm/platforms/cuda.py:690-700``), which is what picks the RMSNorm kernel
(``RMSNorm.forward_cuda`` and ``forward_native`` both call
``ir.ops.rms_norm``).  vLLM prints both resolved values in its startup config
line, so this module reads them off the log and puts them in the identity: two
arms that ran different implementations of the same math are not comparable as
measurements of the weights, and the record now says so in the field rather
than leaving it to be inferred from ``compiled_forward``.  See
``docs/measurements/serving-compile-dispatch-2026-09-03.md``.

PRINCIPLE 14.  Every field is derived, never asserted: the AOT and backbone
keys and the fresh-compile count come from vLLM's own log lines, and the
digests from the bytes vLLM wrote into the pinned cache root the wrapper
mounted.  Nothing here is a prose claim about the runtime, and nothing here
imports vLLM or torch -- it runs on the host after the container is gone.

WHAT IT DOES NOT CLAIM.  A stamp read from the log alone, without the serve's
cache root, is marked ``complete: false`` and refuses to certify either
sameness or difference. The teacher wrapper now pins and reads that cache root.
An EAGER stamp is a claim too, and it rests on the same kind of evidence: a
missing or unreadable log, a log with no vLLM startup line, or a startup line
that resolved ``enforce_eager=False`` with no compile line anywhere is
``complete: false`` -- because ``compiled_forward`` is false for an eager arm
and for a serve nobody observed alike, and reading the second as the first let
two unobserved arms certify as one build (issue #205).

AND ``complete`` IS DERIVED WHERE IT IS READ, NEVER TRUSTED (issue #279).  The
stamped field is a cached verdict, issued by whichever rule was in force that
day; #205 tightened the eager rule without bumping the schema, so a ``/2``
sidecar written before it carries ``complete: true`` for an eager arm whose
``enforce_eager`` was never read.  ``incomplete_reason`` is the single home of
the rule -- the stamper fills the field from it and every gate re-derives from
the record's own fields -- so an old record reads honestly as incomplete and
says which repair it needs, rather than certifying on a verdict nobody would
issue today.  The schema stays ``/2``: bumping it would refuse 22 sidecars to
correct 12, and re-stamping needs serve logs that mostly no longer exist.

And a build identity says two arms
ran the same compiled code; it does not say the compiled code is correct, and
it is not a substitute for serving both arms.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

__all__ = [
    "SCHEMA",
    "BuildIdentityError",
    "build_identity",
    "compare",
    "deterministic_effective",
    "incomplete_reason",
    "is_complete",
    "load",
    "read_cache_root",
    "read_serve_log",
    "require_deterministic_build",
    "require_distinct_build",
    "require_same_build",
    "require_same_dispatch",
]

SCHEMA = "tessera.serve_build_identity/2"

#: The env var that switches inductor's numerics-affecting autotuning off.
#: ``torch/_inductor/config.py`` reads it at import time.
DETERMINISM_ENV = "TORCHINDUCTOR_DETERMINISTIC"

# Keys inside a .best_config that record the measurement, not the choice.
_TIMING_KEYS = frozenset({"time_taken_ms", "triton_cache_hash"})

# vLLM 0.28 log lines, all four verified against real serve logs under
# /home/rob/tessera-runs/tsplugin.
_AOT_LOADED = re.compile(
    r"Directly load AOT compilation from path \S*?torch_aot_compile/([0-9a-f]{8,})")
_AOT_SAVED = re.compile(
    r"saved AOT compiled function to \S*?torch_aot_compile/([0-9a-f]{8,})")
_AOT_FAILED = re.compile(
    r"Compiling model again due to a load failure from "
    r"\S*?torch_aot_compile/([0-9a-f]{8,})\S*?, reason: (.*)$")
_BACKBONE = re.compile(
    r"Using cache directory: \S*?torch_compile_cache/([0-9a-f]{6,})/rank_[0-9_]+/backbone")
_FRESH = re.compile(r"Dynamo bytecode transform time")
_VLLM_VERSION = re.compile(r"Initializing a V\d+ LLM engine \(v([0-9][^)]*)\)")
#: The execution mode vLLM RESOLVED, printed in the same startup line as the
#: version.  Positive evidence, which is the point: the absence of compile
#: lines is not an eager serve, it is a log that says nothing (#205).
_ENFORCE_EAGER = re.compile(r"enforce_eager=(True|False)")
# The resolved dispatch, printed by vLLM in the same startup config line.
_CUSTOM_OPS = re.compile(r"'custom_ops': \[([^\]]*)\]")
_IR_PRIORITY = re.compile(r"ir_op_priority=IrOpPriorityConfig\(([^)]*)\)")
_IR_ENTRY = re.compile(r"(\w+)=\[([^\]]*)\]")


class BuildIdentityError(RuntimeError):
    """A comparison whose two arms' compiled builds do not support it."""


# ------------------------------------------------------------------ read ---

def read_serve_log(text: str) -> dict:
    """Everything the serve log says about the compiled forward that ran.

    ``compiled_forward`` is derived from evidence (an AOT slot was loaded,
    saved, or a backbone directory was opened), not from the flag the operator
    believes they passed: an ``--enforce-eager`` serve that compiled anyway
    would otherwise stamp as eager.

    Its FALSE is therefore an absence, and an absence is not an eager serve.
    ``vllm_version`` and ``enforce_eager`` come off vLLM's own startup line and
    are the positive evidence a "nothing was compiled here" claim rests on: a
    log that names no engine recorded no serve at all, and one whose engine
    line says ``enforce_eager=False`` while carrying no compile line is a
    contradiction rather than an eager arm (#205).
    """
    loaded: set[str] = set()
    saved: set[str] = set()
    backbone: set[str] = set()
    failures: list[str] = []
    fresh = 0
    version: str | None = None
    enforce_eager: bool | None = None
    dispatch: dict | None = None
    requested: dict | None = None
    for line in text.splitlines():
        if (d := _read_dispatch(line)) is not None:
            # The LAST config line, not the first.  An arm that pins the
            # dispatch logs the line twice: once for what the operator asked
            # for on the CLI, and once for what vLLM resolved -- and only the
            # second says what ran.  Reading the first made the gate compare a
            # request against a resolution and answer "different
            # implementations" for three pairs whose served KL is exactly
            # 0.000000 at 100.00% top-1 over 4088 positions
            # (eager vs compiled-both, compiled-both-noauto,
            # compiled-eagerbackend; /home/rob/tessera-runs/compile-dispatch).
            # Under this rule the gate reproduces all six measured pairs: the
            # three above pass, and eager vs compiled / -ir / -ops refuse,
            # which are the three that moved 0.244-0.249 KL and ~30% of top-1.
            if dispatch is not None and requested is None and d != dispatch:
                # Keep the ask, so a pinned arm is still distinguishable from
                # one that was never pinned.  Only when it differs: an
                # unpinned arm logs one line and has nothing to record.
                requested = dispatch
            dispatch = d
        if (m := _AOT_LOADED.search(line)):
            loaded.add(m.group(1))
        if (m := _AOT_SAVED.search(line)):
            saved.add(m.group(1))
        if (m := _AOT_FAILED.search(line)):
            # The key it failed to load from is the key it will save back to,
            # and the save line records that; what only this line carries is
            # vLLM's reason, which is the difference between "the cache was
            # cold" and "the sources moved under a warm cache".
            failures.append(m.group(2).strip())
        if (m := _BACKBONE.search(line)):
            backbone.add(m.group(1))
        if _FRESH.search(line):
            fresh += 1
        if version is None and (m := _VLLM_VERSION.search(line)):
            version = m.group(1)
            # Read off the SAME line, so the mode recorded is the one the
            # engine that started reported, not a value scraped from any line
            # of a concatenated log.
            mode = _ENFORCE_EAGER.search(line)
            enforce_eager = None if mode is None else mode.group(1) == "True"
    keys = sorted(loaded | saved)
    return {
        "vllm_version": version,
        "enforce_eager": enforce_eager,
        "aot_keys": keys,
        "aot_keys_loaded": sorted(loaded),
        "aot_keys_saved": sorted(saved),
        "backbone_keys": sorted(backbone),
        "fresh_compiles": fresh,
        "reload_failures": failures,
        "compiled_forward": bool(keys or backbone or fresh),
        "dispatch": dispatch,
        "dispatch_requested": requested,
    }


def _strings(inner: str) -> list[str]:
    return [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]


def _read_dispatch(line: str) -> dict | None:
    """The ``custom_ops`` base mode and IR-op priority vLLM resolved, off its own line.

    None when the line is not the config line: an old log, or a log truncated
    before startup finished.  A record whose dispatch is None does not claim
    the arms agreed, and ``require_same_dispatch`` refuses on it rather than
    reading absence as agreement.
    """
    ops = _CUSTOM_OPS.search(line)
    ir = _IR_PRIORITY.search(line)
    if ops is None or ir is None:
        # BOTH halves, or nothing.  Returning a record when only one matched
        # leaves the other key ``None`` while ``dispatch`` itself is no longer
        # None -- so the record reads as "the dispatch is known", and two
        # half-parsed arms compare equal on the missing half.  That is absence
        # read as agreement, which is the one failure
        # ``require_same_dispatch`` exists to prevent, and it is worth 30% of
        # the top-1 predictions when the halves really differ (see
        # docs/measurements/serving-compile-dispatch-2026-09-03.md section 3).
        # The two values are printed on the same startup line, so needing both
        # costs nothing on a log that has it and refuses on a log that does
        # not.
        return None
    priority = {name: _strings(vals)
                for name, vals in _IR_ENTRY.findall(ir.group(1))}
    return {
        "custom_ops": _strings(ops.group(1)),
        "ir_op_priority": priority,
    }


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _autotune_digest(slot: Path) -> tuple[int, str | None]:
    """Digest inductor's per-kernel choices under one AOT slot.

    An unparsable record is digested as such rather than skipped: a truncated
    autotune record is a difference between two caches, not a reason to call
    them the same.
    """
    if not slot.is_dir():
        return 0, None
    records: list[list[Any]] = []
    for p in sorted(slot.rglob("*.best_config")):
        try:
            payload = json.loads(p.read_text())
        except (OSError, ValueError):
            payload = {"__unreadable__": True}
        if isinstance(payload, dict):
            payload = {k: v for k, v in payload.items() if k not in _TIMING_KEYS}
        records.append([str(p.relative_to(slot)), payload])
    if not records:
        return 0, None
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return len(records), hashlib.sha256(blob).hexdigest()


def read_cache_root(root: str | Path, aot_keys: Iterable[str],
                    backbone_keys: Iterable[str]) -> dict:
    """Digest the compile-cache slots the log named, from the bytes on disk."""
    base = Path(root) / "torch_compile_cache"
    aot: dict[str, dict] = {}
    for key in sorted(set(aot_keys)):
        slot = base / "torch_aot_compile" / key
        records, digest = _autotune_digest(slot)
        aot[key] = {
            "present": slot.is_dir(),
            "autotune_records": records,
            "autotune_digest": digest,
        }
    bones: dict[str, dict] = {}
    for key in sorted(set(backbone_keys)):
        slot = base / key
        graphs = sorted(slot.glob("rank_*/backbone/computation_graph.py"))
        factors = sorted(slot.glob("rank_*/backbone/cache_key_factors.json"))
        bones[key] = {
            "present": slot.is_dir(),
            "computation_graph_sha256": _sha256(graphs[0]) if graphs else None,
            "cache_key_factors_sha256": _sha256(factors[0]) if factors else None,
        }
    return {"root": str(root), "aot": aot, "backbone": bones}


def _empty_slots(aot_keys: Iterable[str], backbone_keys: Iterable[str]) -> dict:
    """The shape ``read_cache_root`` returns when no cache root was mounted."""
    return {
        "root": None,
        "aot": {k: {"present": False, "autotune_records": 0, "autotune_digest": None}
                for k in sorted(set(aot_keys))},
        "backbone": {k: {"present": False, "computation_graph_sha256": None,
                         "cache_key_factors_sha256": None}
                     for k in sorted(set(backbone_keys))},
    }


# ----------------------------------------------------------------- build ---

def build_identity(*, serve_log: str | Path, cache_root: str | Path | None = None,
                   image: str | None = None, image_digest: str | None = None,
                   image_local_id: str | None = None, serve_mode: str | None = None,
                   eager: bool | None = None, deterministic: bool = False,
                   artifact_path: str | None = None) -> dict:
    """The record a measurement run writes beside its dump.

    ``identity`` is what the fingerprint is over -- everything that decides the
    arithmetic.  ``provenance`` is everything that does not: the log path, the
    timestamp, the cache root, whether the build was made here or replayed.
    Fingerprinting provenance would give two honest replays two fingerprints
    and the check would be turned off within a week.
    """
    log_path = Path(serve_log)
    try:
        text = log_path.read_text(errors="replace") if log_path.is_file() else None
    except OSError:
        # Unreadable and absent are one state for this record: no serve was
        # observed.  Neither is an eager arm (#205).
        text = None
    parsed = read_serve_log(text or "")

    slots = (read_cache_root(cache_root, parsed["aot_keys"], parsed["backbone_keys"])
             if cache_root is not None
             else _empty_slots(parsed["aot_keys"], parsed["backbone_keys"]))

    identity = {
        "compiled_forward": parsed["compiled_forward"],
        "inductor_deterministic": bool(deterministic),
        "serve_mode": serve_mode,
        "eager": None if eager is None else bool(eager),
        "image": image,
        # WHAT RAN, not what was asked for (issue #100).  ``image`` is the
        # reference the wrapper passed to ``docker run``; under a floating tag
        # that names no bytes, so a receipt carrying only it has recorded
        # nothing about the runtime.  ``image_digest`` is the manifest digest
        # the local daemon resolved that reference to, and it is what belongs
        # in the fingerprint.  The local ``.Id`` deliberately does NOT: it is
        # the manifest digest under the containerd snapshotter and the config
        # digest under overlay2, so identical bytes fingerprint differently on
        # the two GB10s and every cross-box comparison would refuse itself.
        # It rides in provenance instead.  See tessera.serving.runtime_image.
        "image_digest": image_digest,
        "vllm_version": parsed["vllm_version"],
        # Which implementations ran, not only which build: see the module
        # docstring.  It is identity, not provenance -- it decides the
        # arithmetic as directly as the autotune choices do.
        "dispatch": parsed["dispatch"],
        "aot": slots["aot"],
    }
    # The backbone slot is PROVENANCE, not identity, and the asymmetry is the
    # reason: vLLM logs ``Using cache directory: .../backbone`` only when it
    # COMPILES.  A serve that replays that same build logs no such line
    # (measured: the rebuild arm's log has 1, the replay arm's has 0), so
    # fingerprinting the backbone would have given one build two fingerprints
    # and refused the very "replayed by a second serve" row this stamp exists
    # to certify.  The autotune digest already separates the two real caches
    # (04525ea7... vs dbeb2b8b...), so nothing is lost by moving it out.
    record = {
        "schema": SCHEMA,
        "build_fingerprint": _fingerprint(identity),
        # Filled in below by the SAME function the reader decides with, so a
        # stamp cannot be issued under a rule the gate no longer holds (#279).
        "complete": False,
        "identity": identity,
        "provenance": {
            "produced_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"),
            "serve_log": str(log_path),
            "serve_log_read": text is not None,
            "serve_log_sha256": _sha256(log_path),
            # What vLLM's own startup line said it resolved, beside the
            # ``eager`` the caller asked for above.  Provenance by the same
            # rule as ``dispatch_requested``: it is read to decide whether
            # this record ESTABLISHES an execution mode, and the mode that
            # decides the arithmetic is already ``compiled_forward``.
            "enforce_eager": parsed["enforce_eager"],
            "cache_root": slots["root"],
            "backbone": slots["backbone"],
            "artifact_path": artifact_path,
            "image_local_id": image_local_id,
            # What the operator ASKED for, when that is not what vLLM
            # resolved -- provenance, not identity, by the same rule as
            # ``image`` above: a request that lost decides nothing about the
            # arithmetic.  None on an arm that was never pinned, which logs
            # one config line and has no second value to disagree with.
            "dispatch_requested": parsed["dispatch_requested"],
            "fresh_compiles": parsed["fresh_compiles"],
            "reload_failures": parsed["reload_failures"],
            "aot_keys_loaded": parsed["aot_keys_loaded"],
            "aot_keys_saved": parsed["aot_keys_saved"],
        },
    }
    record["complete"] = is_complete(record)
    return record


def _fingerprint(identity: dict) -> str:
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def load(path: str | Path) -> dict:
    """Read a sidecar, schema-checked.  ``complete`` is NOT read from it.

    The stored field is left exactly as the stamp wrote it -- this module never
    edits a sidecar, in memory or on disk -- and every verdict goes through
    ``is_complete``/``incomplete_reason`` instead.  See those for why (#279).
    """
    record = json.loads(Path(path).read_text())
    if record.get("schema") != SCHEMA:
        # /1 records predate the dispatch fields, so they cannot answer "did
        # these two arms run the same implementations?" -- the question #16
        # turned out to be about.  Refusing them is the point: an old record
        # would compare equal on everything it knows and say nothing about the
        # thing that differed.  Re-stamp from the serve log instead.
        raise BuildIdentityError(
            f"{path}: schema {record.get('schema')!r}, expected {SCHEMA!r}")
    return record


# --------------------------------------------------------------- compare ---

def compare(a: dict, b: dict) -> dict:
    """What differs between two build identities, and whether either can certify.

    ``incomplete`` is DERIVED from each record's own fields, never read from its
    stored ``complete`` (#279).  ``stored_complete_disagrees`` names the records
    whose stamped verdict the rule no longer issues, in either direction, so the
    override is never silent.
    """
    incomplete = [name for name, rec in (("a", a), ("b", b)) if not is_complete(rec)]
    disagrees = [name for name, rec in (("a", a), ("b", b))
                 if "complete" in rec and bool(rec["complete"]) is not is_complete(rec)]
    ia, ib = a["identity"], b["identity"]
    differs = sorted(k for k in set(ia) | set(ib) if ia.get(k) != ib.get(k))
    da, db = ia.get("dispatch"), ib.get("dispatch")
    return {
        "same_build": a["build_fingerprint"] == b["build_fingerprint"],
        "same_dispatch": da is not None and db is not None and da == db,
        "dispatch_known": da is not None and db is not None,
        "differs": differs,
        "incomplete": incomplete,
        "stored_complete_disagrees": disagrees,
    }


def is_complete(record: dict) -> bool:
    """Does this record answer "which compiled build served this arm"?

    ``incomplete_reason`` is the rule; this is its yes/no face.
    """
    return incomplete_reason(record) is None


def incomplete_reason(record: dict) -> "str | None":
    """Why this record cannot certify, in its own fields -- or ``None``.

    THIS FUNCTION IS THE COMPLETENESS RULE, and it is the only home of it: the
    stamper calls it to fill in ``complete`` and every reader calls it to decide
    whether a record may certify.  It was two functions -- ``_is_complete`` over
    the parse, this one over the record -- and the pair drifted the moment #205
    tightened one of them (#279).

    WHY THE STORED ``complete`` IS NOT READ.  It is a cached verdict, issued by
    whichever rule was in force on the day of the stamp.  #205 tightened the
    eager rule without bumping the schema, so twelve real ``/2`` sidecars carry
    ``complete: true`` for eager arms whose serve log's ``enforce_eager`` was
    never read -- a verdict the current rule would not issue on the same
    evidence.  Bumping to ``/3`` would refuse the 22 ``/2`` sidecars to correct
    12, and re-stamping needs serve logs that mostly no longer exist.  So the
    verdict is decided where it is read, from the record's own fields; the
    stored field stays for humans, and certifies nothing.  When the two
    disagree, the reason says so.

    An eager serve answers the question with "none" -- but only a serve that was
    OBSERVED does.  ``compiled_forward`` is false both for an eager arm and for
    a log nobody wrote, and reading the second as the first is how two
    unobserved serves certified as one build (#205).  So the eager answer rests
    on vLLM's own startup line: the engine version proves a serve was seen, and
    the ``enforce_eager`` it resolved proves the arm was the eager one.  A log
    that says it was NOT eager and carries no compile line is a contradiction,
    not an eager arm, and neither is a request to compile that left no compile
    evidence.

    A compiled serve is complete only when every cache slot its log named was
    found and digested: the key alone does not distinguish two builds, so a
    key-only record must not be allowed to certify.

    The reason matters because the states are different repairs: a missing cache
    root is re-stamped with the root mounted, a pre-#205 eager record is
    re-stamped from its serve log where that survives, and a log that recorded
    no serve is not a stamping problem at all.
    """
    reason = _incomplete_reason(record)
    if reason is None:
        return None
    if record.get("complete"):
        reason += (" -- and it is stamped complete: true, a verdict issued under an "
                   "older rule; the rule lives here, not in the sidecar, so the stored "
                   "field does not certify")
    return reason


def _incomplete_reason(record: dict) -> "str | None":
    identity, provenance = record["identity"], record["provenance"]
    if identity.get("vllm_version") is None:
        return ("its serve log carries no vLLM startup line"
                + ("" if provenance.get("serve_log_read", True)
                   else f" (no readable log at {provenance.get('serve_log')!r})")
                + ", so no serve was observed; an absent compile record is not an eager "
                  "serve, it is a log that says nothing")
    if not identity["compiled_forward"]:
        if "enforce_eager" not in provenance:
            # A record from before #205 gave the field a home.  It cannot be
            # told from a post-#205 one by its schema -- that is the whole of
            # #279 -- so it is told apart by the field it does not have.
            return ("its serve log's enforce_eager was never read (the record predates "
                    "the field, so nothing here shows the arm was the eager one rather "
                    f"than a serve nobody observed); re-stamp from "
                    f"{provenance.get('serve_log')!r}")
        if provenance["enforce_eager"] is not True:
            return ("its serve log records no compiled forward while vLLM's own startup "
                    f"line resolved enforce_eager={provenance['enforce_eager']!r}, so "
                    "what this arm ran is not established")
        if identity.get("eager") is False:
            return ("it was stamped eager=False -- the arm was asked to compile -- and "
                    "its serve log carries no compile evidence at all")
        return None
    if provenance.get("cache_root") is None or not identity["aot"]:
        return ("its serve log names an AOT key but no compile-cache root was read, and "
                "the key does not identify the build (one key held both of the two "
                "builds that differ by 0.017117); mount the cache root the serve used "
                "and stamp again")
    missing = sorted(k for k, s in identity["aot"].items()
                     if not (s["present"] and s["autotune_digest"]))
    if missing:
        return ("the compile-cache root it read holds no digestible autotune record for "
                f"AOT slot {', '.join(missing)}, which its serve log named; the key "
                "alone does not identify the build, so this record cannot certify one")
    # A replay names no backbone key, so this is vacuously satisfied there; on a
    # build it asserts the mounted cache root is the one the log named.
    absent = sorted(k for k, s in provenance.get("backbone", {}).items()
                    if not (s["present"] and s["computation_graph_sha256"]))
    if absent:
        return (f"its serve log named backbone slot {', '.join(absent)} -- so this serve "
                "COMPILED -- but the cache root it read does not hold that slot's "
                "computation graph, so the root read is not the root the serve wrote")
    return None


def _refuse_incomplete(verdict: dict, why: str, records: dict) -> None:
    if verdict["incomplete"]:
        reasons = "; ".join(
            f"{name}: {incomplete_reason(records[name])}" for name in verdict["incomplete"])
        raise BuildIdentityError(
            f"{why}: build identity is incomplete for {', '.join(verdict['incomplete'])} "
            f"-- {reasons}. Do not read this as agreement.")


def require_same_build(a: dict, b: dict, *, why: str) -> None:
    """Refuse unless both arms provably served one compiled build."""
    verdict = compare(a, b)
    _refuse_incomplete(verdict, why, {"a": a, "b": b})
    if not verdict["same_build"]:
        raise BuildIdentityError(
            f"{why}: the two arms served a different compiled build "
            f"({', '.join(verdict['differs']) or 'no field differs, fingerprints do'}); "
            "any difference between them is a difference of the compiler as much as of "
            "the weights")


def require_same_dispatch(a: dict, b: dict, *, why: str) -> None:
    """Refuse unless both arms ran the same implementations of the same math.

    Weaker than ``require_same_build`` and orthogonal to it: two arms may
    share every kernel choice and still be one eager and one compiled, which
    on this runtime means one ran ``vllm_c``/``forward_cuda`` and the other ran
    ``native``/``forward_native``.  That difference is worth 0.2445 KL on the
    NVFP4 route -- larger than the whole KL-vs-BF16 of a good 4-bit arm -- so a
    cross-regime comparison is a measurement of the dispatch unless the
    dispatch was pinned to match.
    """
    verdict = compare(a, b)
    if not verdict["dispatch_known"]:
        raise BuildIdentityError(
            f"{why}: at least one arm's serve log does not record the dispatch vLLM "
            "resolved (no 'custom_ops'/ir_op_priority in the startup config line), so "
            "whether the two arms ran the same implementations is unknown; do not read "
            "the absence as agreement")
    if not verdict["same_dispatch"]:
        ia, ib = a["identity"]["dispatch"], b["identity"]["dispatch"]
        raise BuildIdentityError(
            f"{why}: the two arms ran different implementations of the same math "
            f"(a: {ia}, b: {ib}); vLLM picks the CUDA kernels when it does not compile "
            "and the torch decompositions when it does, and the difference is a "
            "measurement of that switch, not of the weights")


def require_distinct_build(a: dict, b: dict, *, why: str) -> None:
    """Refuse unless the two arms provably served different compiled builds.

    The mirror image, for a row that exists to measure the rebuild: labelling a
    replay as a rebuild would report 0.000000 as evidence that rebuilds agree.
    """
    verdict = compare(a, b)
    _refuse_incomplete(verdict, why, {"a": a, "b": b})
    if verdict["same_build"]:
        raise BuildIdentityError(
            f"{why}: both arms served the same compiled build "
            f"({a['build_fingerprint'][:16]}); this is a replay, not a rebuild")


# --------------------------------------------------- the determinism knob ---

def deterministic_effective(record: dict) -> bool:
    """Did ``TORCHINDUCTOR_DETERMINISTIC`` actually decide this build?

    Only if inductor ran.  Whether or not vLLM's compile-cache key covers
    ``torch._inductor.config.deterministic`` -- the pinned image's
    ``cache_key_factors.json`` for the surviving backbone slot does not mention
    it, but that is one artifact, not the runtime's own table -- the tell is
    the same either way: if the flag forces a rebuild, ``fresh_compiles`` is
    non-zero and the claim stands; if a warm ``$VLLM_CACHE`` hands back an
    artifact autotuned *without* the flag, ``fresh_compiles`` is 0 and the flag
    decided nothing.
    """
    return bool(record["identity"]["inductor_deterministic"]
                and record["provenance"]["fresh_compiles"] > 0)


def require_deterministic_build(record: dict) -> None:
    """Refuse a record that only *claims* a deterministic build."""
    if deterministic_effective(record):
        return
    if not record["identity"]["inductor_deterministic"]:
        raise BuildIdentityError(
            f"this arm was served without {DETERMINISM_ENV}=1; its autotune choices "
            "were benchmarked on a loaded box")
    if not record["identity"]["compiled_forward"]:
        raise BuildIdentityError(
            f"{DETERMINISM_ENV}=1 was set but this arm served eager: nothing was "
            "compiled, so the flag decided nothing here")
    raise BuildIdentityError(
        f"{DETERMINISM_ENV}=1 was set but the compiled forward was replayed from a warm "
        "cache (0 fresh compiles), so inductor never ran and the flag decided nothing; "
        "empty the compile-cache root for a deterministic build, or drop the claim")


# ------------------------------------------------------------------- CLI ---

def _bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("stamp", help="write a build identity beside a KL dump")
    st.add_argument("--log", required=True, help="the serve log (docker logs > ...)")
    st.add_argument("--out", required=True, help="sidecar to write (<dump>.build.json)")
    st.add_argument("--cache-root", default=None,
                    help="host path mounted at the container's /root/.cache/vllm")
    st.add_argument("--image", default=None,
                    help="the reference passed to docker run (may be a tag)")
    st.add_argument("--image-digest", default=None,
                    help="the manifest digest that reference resolved to")
    st.add_argument("--image-local-id", default=None,
                    help="this daemon's image id; provenance only, never compared")
    st.add_argument("--serve-mode", default=None)
    st.add_argument("--eager", default=None)
    st.add_argument("--deterministic", default="0")
    st.add_argument("--artifact-path", default=None)

    cm = sub.add_parser("compare", help="two sidecars: same build or not")
    cm.add_argument("a")
    cm.add_argument("b")
    cm.add_argument("--require", choices=["same", "distinct", "same-dispatch"],
                    default=None,
                    help="exit 4 unless the two arms are provably that.  "
                         "same/distinct are about the compiled build; "
                         "same-dispatch is about which implementations ran, "
                         "which is the check a cross-regime KL needs and the "
                         "one an eager-vs-compiled pair fails")

    args = ap.parse_args(argv)
    if args.cmd == "stamp":
        record = build_identity(
            serve_log=args.log,
            cache_root=args.cache_root or None,
            image=args.image,
            image_digest=args.image_digest or None,
            image_local_id=args.image_local_id or None,
            serve_mode=args.serve_mode,
            eager=None if args.eager is None else _bool(args.eager),
            deterministic=_bool(args.deterministic),
            artifact_path=args.artifact_path)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")
        ident = record["identity"]
        state = ("eager, no compiled build" if not ident["compiled_forward"]
                 else f"aot {', '.join(ident['aot']) or 'none'}")
        print(f"build identity -> {out}  ({state}; "
              f"fingerprint {record['build_fingerprint'][:16]}; "
              f"complete={record['complete']}; "
              f"fresh_compiles={record['provenance']['fresh_compiles']})")
        return 0

    a, b = load(args.a), load(args.b)
    verdict = compare(a, b)
    print(json.dumps(verdict, indent=1))
    if args.require is None:
        return 0
    try:
        if args.require == "same":
            require_same_build(a, b, why=f"{args.a} vs {args.b}")
        elif args.require == "same-dispatch":
            require_same_dispatch(a, b, why=f"{args.a} vs {args.b}")
        else:
            require_distinct_build(a, b, why=f"{args.a} vs {args.b}")
    except BuildIdentityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
