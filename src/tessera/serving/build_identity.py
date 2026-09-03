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

PRINCIPLE 14.  Every field is derived, never asserted: the AOT and backbone
keys and the fresh-compile count come from vLLM's own log lines, and the
digests from the bytes vLLM wrote into the pinned cache root the wrapper
mounted.  Nothing here is a prose claim about the runtime, and nothing here
imports vLLM or torch -- it runs on the host after the container is gone.

WHAT IT DOES NOT CLAIM.  A stamp read from the log alone -- no cache root, as
``serve_and_dump_kl.sh`` runs today -- is marked ``complete: false`` and refuses
to certify either sameness or difference.  And a build identity says two arms
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
    "load",
    "read_cache_root",
    "read_serve_log",
    "require_deterministic_build",
    "require_distinct_build",
    "require_same_build",
]

SCHEMA = "tessera.serve_build_identity/1"

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


class BuildIdentityError(RuntimeError):
    """A comparison whose two arms' compiled builds do not support it."""


# ------------------------------------------------------------------ read ---

def read_serve_log(text: str) -> dict:
    """Everything the serve log says about the compiled forward that ran.

    ``compiled_forward`` is derived from evidence (an AOT slot was loaded,
    saved, or a backbone directory was opened), not from the flag the operator
    believes they passed: an ``--enforce-eager`` serve that compiled anyway
    would otherwise stamp as eager.
    """
    loaded: set[str] = set()
    saved: set[str] = set()
    backbone: set[str] = set()
    failures: list[str] = []
    fresh = 0
    version: str | None = None
    for line in text.splitlines():
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
    keys = sorted(loaded | saved)
    return {
        "vllm_version": version,
        "aot_keys": keys,
        "aot_keys_loaded": sorted(loaded),
        "aot_keys_saved": sorted(saved),
        "backbone_keys": sorted(backbone),
        "fresh_compiles": fresh,
        "reload_failures": failures,
        "compiled_forward": bool(keys or backbone or fresh),
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
                   image: str | None = None, serve_mode: str | None = None,
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
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    parsed = read_serve_log(text)

    slots = (read_cache_root(cache_root, parsed["aot_keys"], parsed["backbone_keys"])
             if cache_root is not None
             else _empty_slots(parsed["aot_keys"], parsed["backbone_keys"]))

    identity = {
        "compiled_forward": parsed["compiled_forward"],
        "inductor_deterministic": bool(deterministic),
        "serve_mode": serve_mode,
        "eager": None if eager is None else bool(eager),
        "image": image,
        "vllm_version": parsed["vllm_version"],
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
    complete = _is_complete(identity, slots["backbone"], cache_root is not None)
    return {
        "schema": SCHEMA,
        "build_fingerprint": _fingerprint(identity),
        "complete": complete,
        "identity": identity,
        "provenance": {
            "produced_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"),
            "serve_log": str(log_path),
            "serve_log_sha256": _sha256(log_path),
            "cache_root": slots["root"],
            "backbone": slots["backbone"],
            "artifact_path": artifact_path,
            "fresh_compiles": parsed["fresh_compiles"],
            "reload_failures": parsed["reload_failures"],
            "aot_keys_loaded": parsed["aot_keys_loaded"],
            "aot_keys_saved": parsed["aot_keys_saved"],
        },
    }


def _is_complete(identity: dict, backbone: dict, had_cache_root: bool) -> bool:
    """True when the record answers "which compiled build served this arm".

    An eager serve answers it with "none" and is complete.  A compiled serve is
    complete only when every cache slot its log named was found and digested:
    the key alone does not distinguish two builds, so a key-only record must
    not be allowed to certify.
    """
    if not identity["compiled_forward"]:
        return True
    if not had_cache_root or not identity["aot"]:
        return False
    if not all(s["present"] and s["autotune_digest"] for s in identity["aot"].values()):
        return False
    # A replay names no backbone key, so this is vacuously true there; on a
    # build it asserts the mounted cache root is the one the log named.
    return all(s["present"] and s["computation_graph_sha256"]
               for s in backbone.values())


def _fingerprint(identity: dict) -> str:
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def load(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text())
    if record.get("schema") != SCHEMA:
        raise BuildIdentityError(
            f"{path}: schema {record.get('schema')!r}, expected {SCHEMA!r}")
    return record


# --------------------------------------------------------------- compare ---

def compare(a: dict, b: dict) -> dict:
    """What differs between two build identities, and whether either can certify."""
    incomplete = [name for name, rec in (("a", a), ("b", b)) if not rec["complete"]]
    ia, ib = a["identity"], b["identity"]
    differs = sorted(k for k in set(ia) | set(ib) if ia.get(k) != ib.get(k))
    return {
        "same_build": a["build_fingerprint"] == b["build_fingerprint"],
        "differs": differs,
        "incomplete": incomplete,
    }


def _refuse_incomplete(verdict: dict, why: str) -> None:
    if verdict["incomplete"]:
        raise BuildIdentityError(
            f"{why}: build identity is incomplete for {', '.join(verdict['incomplete'])} "
            "-- the serve log names an AOT key but no compile-cache root was read, and "
            "the key does not identify the build (one key held both of the two builds "
            "that differ by 0.017117).  Mount the cache root the serve used and stamp "
            "again; do not read this as agreement.")


def require_same_build(a: dict, b: dict, *, why: str) -> None:
    """Refuse unless both arms provably served one compiled build."""
    verdict = compare(a, b)
    _refuse_incomplete(verdict, why)
    if not verdict["same_build"]:
        raise BuildIdentityError(
            f"{why}: the two arms served a different compiled build "
            f"({', '.join(verdict['differs']) or 'no field differs, fingerprints do'}); "
            "any difference between them is a difference of the compiler as much as of "
            "the weights")


def require_distinct_build(a: dict, b: dict, *, why: str) -> None:
    """Refuse unless the two arms provably served different compiled builds.

    The mirror image, for a row that exists to measure the rebuild: labelling a
    replay as a rebuild would report 0.000000 as evidence that rebuilds agree.
    """
    verdict = compare(a, b)
    _refuse_incomplete(verdict, why)
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
    st.add_argument("--image", default=None)
    st.add_argument("--serve-mode", default=None)
    st.add_argument("--eager", default=None)
    st.add_argument("--deterministic", default="0")
    st.add_argument("--artifact-path", default=None)

    cm = sub.add_parser("compare", help="two sidecars: same build or not")
    cm.add_argument("a")
    cm.add_argument("b")
    cm.add_argument("--require", choices=["same", "distinct"], default=None,
                    help="exit 4 unless the two arms are provably that")

    args = ap.parse_args(argv)
    if args.cmd == "stamp":
        record = build_identity(
            serve_log=args.log,
            cache_root=args.cache_root or None,
            image=args.image,
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
        else:
            require_distinct_build(a, b, why=f"{args.a} vs {args.b}")
    except BuildIdentityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
