"""tessera#702: build a ``tessera.graph_equals_eager.v1`` receipt from equality arms.

  receipt.py RECEIPTS MANIFEST OUT --eager rE1,rE2 --graph rG1,rG3 [--commit SHA]

RECEIPTS holds one directory per arm (arm.sh's $RECEIPTS/<ARM>/), MANIFEST is
submit.py's (arm -> PrismaBuild action key). Every graph arm is judged against
the eager pool named by --eager with the tessera#508 membership predicate
(``eq-member-508.py``'s ``members``); the long-context passes (<ARM>-long)
are judged the same way and recorded as a screen. The rule, the schema and
the verdict are ``tessera.graph_receipt``'s. Exit 0 iff ``equal``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import shlex
import sys
import tempfile
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tessera import graph_receipt  # noqa: E402
from tessera.serving.build_identity import read_serve_log  # noqa: E402

QUAL = ROOT / "experiments" / "glm53_508_graph_qual"
_spec = importlib.util.spec_from_file_location("eq_member", QUAL / "eq-member-508.py")
eq_member = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq_member)

_COUNT = re.compile(r"^(?P<manager>\w+)\|FULL\|tokens=(?P<tokens>\d+)\|reqs=")
#: <arm>.dispatch.<pid>.json (one box) or <arm>.rank<r>.dispatch.<pid>.json (arm_tp2.sh, per rank).
_DISPATCH = r"{arm}(?:\.rank(?P<rank>\d+))?\.dispatch\.\d+\.json"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_args(arm_dir: pathlib.Path, arm: str) -> dict[str, str]:
    out = {}
    for line in (arm_dir / f"engine-args-{arm}.txt").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def graph_record(arm_dir: pathlib.Path, arm: str) -> dict:
    """Captured and replayed FULL sizes per manager class, and Tessera's per-class counts.

    A tensor-parallel arm writes one dispatch log per rank; each rank's managers are
    keyed ``rank<r>:<class>``, so every rank must replay what it captured on its own.
    """
    captured, replayed = defaultdict(set), defaultdict(lambda: defaultdict(int))
    classes = {"captured": {}, "replays": {}}
    pattern = re.compile(_DISPATCH.format(arm=re.escape(arm)))
    for path in sorted(arm_dir.iterdir()):
        if not (found := pattern.fullmatch(path.name)):
            continue
        tag = f"rank{found['rank']}:" if found["rank"] is not None else ""
        rec = json.loads(path.read_text())
        for owner, sizes in rec.get("captured", {}).items():
            if isinstance(sizes, list):
                captured[tag + owner.split("@")[0]].update(int(s) for s in sizes)
        for key, count in rec.get("counts", {}).items():
            if (m := _COUNT.match(key)):
                replayed[tag + m["manager"]][int(m["tokens"])] += count
        tc = rec.get("tessera_classes") or {}
        for manager, by_bound in tc.get("captured", {}).items():
            classes["captured"].setdefault(tag + manager, {}).update(by_bound)
        for key, count in tc.get("replays", {}).items():
            classes["replays"][tag + key] = max(classes["replays"].get(tag + key, 0), count)
    managers = {m: {"captured_sizes": sorted(s), "replayed_sizes": dict(sorted(replayed[m].items()))}
                for m, s in sorted(captured.items()) if s}
    return {"managers": managers, "classes": classes}


def flat_dir(receipts: pathlib.Path, arms: list[str]) -> pathlib.Path:
    """eq-member reads one directory; link every named arm's files into one."""
    flat = pathlib.Path(tempfile.mkdtemp(prefix="ga702-flat-"))
    for arm in arms:
        for path in (receipts / arm).iterdir():
            target = flat / path.name
            if not target.exists():
                target.symlink_to(path)
    return flat


def passes(flat: pathlib.Path, names: list[str], pool: list[str]) -> list[dict]:
    runs = {}
    for p in pool:
        for name in (p, f"{p}-r2"):
            if (flat / f"{name}.eq.summary.json").exists():
                runs[name] = eq_member.load_arm(flat, name)
    out = []
    for name in names:
        if not (flat / f"{name}.eq.summary.json").exists():
            out.append({"name": name, "members": 0, "choices": 0, "missing": True})
            continue
        target = eq_member.load_arm(flat, name)
        choices = sum(len(cs) for cs in target.values())
        out.append({"name": name, "members": len(eq_member.members(target, runs)),
                    "choices": choices})
    return out


#: The scope fields every arm of one receipt must share (graph_receipt.SCOPE_FIELDS minus
#: compilation_config, which is what separates a graph arm from its eager pool).
SHARED_SCOPE = tuple(f for f in graph_receipt.SCOPE_FIELDS if f != "compilation_config")
#: What else every arm must share to be one measurement, though no verify() scope names it:
#: the equality script and the runner hooks that ran, the serve flags other than the
#: execution mode, the kernel config other than the operator priority a graph arm may pin
#: (cause 1, judged by the plugin), the Tessera env and the fabric.
SHARED_SETUP = ("equal_script_sha256", "hooks_sha256", "serve_flags", "kernel_config",
                "tessera_env", "fabric_requested")
#: The flags that are the execution mode under test, or compared on their own.
_MODE_FLAGS = {"--enforce-eager": 0, "--compilation-config": 1, "--kernel-config": 1}
_REFERENCE = re.compile(r"[a-z0-9./:_-]+@sha256:[0-9a-f]{64}")


def _flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def _serve_flags(argv: list[str]) -> list[str]:
    out, skip = [], 0
    for word in argv:
        if skip:
            skip -= 1
        elif word in _MODE_FLAGS:
            skip = _MODE_FLAGS[word]
        else:
            out.append(word)
    return out


def arm_scope(arm_dir: pathlib.Path, arm: str, legacy_scripts: dict | None = None) -> dict:
    """The scope and setup one arm MEASURED, read from what it recorded, never from a default.

    The serve values come from the argv the arm launched (``serve_args`` from arm.sh,
    ``serve_rank0`` from arm_tp2.sh). The image is a whole digest reference: the one the
    runtime gate resolved (``image_resolved_reference``), else the digest-pinned reference the
    arm declared and PrismaBuild verified. A bare ``sha256:`` digest names no image and a
    box-local image id names one box's store, so neither is ever the scope. The equality
    script is the hash the arm recorded, else the legacy provenance (read out of the arm's own
    PrismaBuild snapshot) passed in ``legacy_scripts``.
    """
    args = engine_args(arm_dir, arm)
    argv = shlex.split(args.get("serve_rank0") or args.get("serve_args") or "")
    missing = []
    image = args.get("image_resolved_reference") or ""
    if not _REFERENCE.fullmatch(image):
        image = args.get("image", "")
    if not _REFERENCE.fullmatch(image):
        missing.append("image (no resolved or declared repository@sha256 reference)")
    values = {"tensor_parallel_size": _flag(argv, "--tensor-parallel-size"),
              "max_model_len": _flag(argv, "--max-model-len"),
              "max_num_seqs": _flag(argv, "--max-num-seqs")}
    missing += [f"{k} (not in the recorded serve argv)" for k, v in values.items() if v is None]
    model = args.get("model")
    config = pathlib.Path(model) / "config.json" if model else None
    if config is None or not config.is_file():
        missing.append(f"model config ({model!r})")
    for key in ("src_sha256", "hooks_sha256"):
        if not args.get(key):
            missing.append(key)
    script = args.get("equal_script_sha256") or (legacy_scripts or {}).get(arm, {}).get("sha256")
    if not script:
        missing.append("equal_script_sha256 (not recorded, and no legacy provenance for this arm)")
    if missing:
        raise SystemExit(f"{arm}: the arm did not record {missing}; its scope is unknown")
    spec = json.loads(args["spec_json"]) if args.get("spec_json") else None
    kernel = json.loads(_flag(argv, "--kernel-config") or args.get("kernel_json") or "{}")
    kernel.pop("ir_op_priority", None)
    return {"image": image, "model_config_sha256": sha256(config),
            "tessera_src_sha256": args["src_sha256"],
            "speculative_tokens": int(spec["num_speculative_tokens"]) if spec else 0,
            **{k: int(v) for k, v in values.items()},
            "equal_script_sha256": script, "hooks_sha256": args["hooks_sha256"],
            "serve_flags": _serve_flags(argv), "kernel_config": kernel,
            "tessera_env": args.get("tessera_env") or None,
            "fabric_requested": args.get("fabric_requested") or None}


def one_measurement(scopes: dict[str, dict]) -> dict:
    """The shared scope and setup of every arm, or refuse naming each field they disagree on."""
    fields = (*SHARED_SCOPE, *SHARED_SETUP)
    differ = [f for f in fields
              if len({graph_receipt.canonical(s.get(f)) for s in scopes.values()}) != 1]
    if differ:
        detail = "; ".join(f"{f}: " + ", ".join(f"{a}={scopes[a].get(f)}" for a in sorted(scopes))
                           for f in differ)
        raise SystemExit(f"the arms are not one measurement, they differ in {differ} ({detail})")
    return dict(next(iter(scopes.values())))


def require_every_rank(arm_dir: pathlib.Path, arm: str, tensor_parallel_size: int) -> None:
    """A tensor-parallel arm's replay check reads one dispatch log per rank, or none of them."""
    pattern = re.compile(_DISPATCH.format(arm=re.escape(arm)))
    ranks = {m["rank"] for p in arm_dir.iterdir() if (m := pattern.fullmatch(p.name))}
    want = {None} if tensor_parallel_size == 1 else {str(r) for r in range(tensor_parallel_size)}
    if ranks != want:
        have = sorted("untagged" if r is None else f"rank{r}" for r in ranks)
        need = sorted("untagged" if r is None else f"rank{r}" for r in want)
        raise SystemExit(f"{arm}: dispatch logs from {have}, need {need}: "
                         "a rank without its log cannot show it replayed what it captured")


def arm_record(receipts, manifest, flat, arm, pool, legacy_scripts=None) -> dict:
    d = receipts / arm
    args = engine_args(d, arm)
    scope = arm_scope(d, arm, legacy_scripts)
    require_every_rank(d, arm, scope["tensor_parallel_size"])
    log = (d / f"{arm}.engine.log").read_text(errors="replace") if (d / f"{arm}.engine.log").exists() else ""
    serve = read_serve_log(log)
    spec = json.loads(args["spec_json"]) if args.get("spec_json") else None
    comp = json.loads(args["compilation_json"]) if args.get("compilation_json") else None
    return {
        "name": arm, "pb_action": manifest.get(arm, {}).get("action_key"), "host": args.get("host"),
        "execution": "eager" if args.get("eager") == "1" else "graph",
        "compilation_config": comp,
        "kernel_config": json.loads(args["kernel_json"]) if args.get("kernel_json") else None,
        **{k: v for k, v in scope.items() if k in (
            "speculative_tokens", "max_model_len", "max_num_seqs", "tensor_parallel_size", "image",
            "equal_script_sha256")},
        "image_id": args.get("image_id"), "src_sha256": args.get("src_sha256"),
        "model": args.get("model"), "vllm": serve.get("vllm_version"),
        "resolved": serve.get("dispatch"), "enforce_eager": serve.get("enforce_eager"),
        "graph": graph_record(d, arm),
        "passes": passes(flat, [arm, f"{arm}-r2"], pool),
        "screen_long": passes(flat, [f"{arm}-long"], [f"{p}-long" for p in pool]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("receipts", type=pathlib.Path)
    ap.add_argument("manifest", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--eager", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--not-measured", action="append", default=None, metavar="TEXT",
                    help="what this receipt does not cover, one per flag (default: the stub-B list)")
    ap.add_argument("--legacy-script-provenance", type=pathlib.Path, default=None,
                    help="JSON {arm: {sha256, source}} for arms that predate the recorded "
                         "equal_script_sha256 (snapshot_script_hash.py reads it out of each "
                         "arm's own PrismaBuild snapshot)")
    ap.add_argument("--image", default=None,
                    help="optional cross-check: refuse unless the arms measured this image")
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    eager, graph = a.eager.split(","), a.graph.split(",")
    flat = flat_dir(a.receipts, eager + graph)
    legacy = json.loads(a.legacy_script_provenance.read_text()) if a.legacy_script_provenance else {}
    pool_records = [arm_record(a.receipts, manifest, flat, e, [p for p in eager if p != e], legacy)
                    for e in eager]
    arms = [arm_record(a.receipts, manifest, flat, g, eager, legacy) for g in graph]
    scope = one_measurement({name: arm_scope(a.receipts / name, name, legacy)
                             for name in eager + graph})
    if a.image is not None and a.image != scope["image"]:
        raise SystemExit(f"the arms measured image {scope['image']}, not --image {a.image}")
    model = pool_records[0]["model"]
    config = pathlib.Path(model) / "config.json"
    text = json.loads(config.read_text())
    text = text.get("text_config", text)
    receipt = {
        "schema": graph_receipt.SCHEMA, "issue": "tessera#702",
        "runtime": {"image": scope["image"], "vllm": arms[0]["vllm"], "interface": "nightly-20260929"},
        "tessera": {"commit": a.commit, "src_sha256": scope["tessera_src_sha256"]},
        "model": {"path": model, "config_sha256": scope["model_config_sha256"],
                  "index_topk": text.get("index_topk")},
        "equality_set": {
            "name": "tessera#508", "script_sha256": scope["equal_script_sha256"], "choices_per_pass": 48,
            # Where each arm's script hash comes from: what the arm recorded, or legacy provenance.
            "script_provenance": {
                n: ("recorded by the arm" if engine_args(a.receipts / n, n).get("equal_script_sha256")
                    else legacy[n]["source"]) for n in eager + graph}},
        "eager_pool": pool_records, "arms": arms,
        "screens": {"long_context": {
            "why": "above index_topk eager is not repeat-exact (stock top-k arrival order); "
                   "membership against the eager pool's outcomes is a screen, never the verdict",
            "eager_pool": {r["name"]: r["screen_long"] for r in pool_records},
            "arms": {r["name"]: r["screen_long"] for r in arms}}},
        "not_measured": [
            # Every receipt: the > index_topk class replays only under screen traffic (review #2).
            "equality of the max_seq_len > index_topk class: screen only, its replays come from "
            "the long-context screen, which no verdict reads",
            *(a.not_measured if a.not_measured is not None else [
                "tensor_parallel_size 2", "the full GLM-5.3 artifact (u1 stub B, 8 layers)",
                "served KL against BF16 under graphs", "graph-vs-eager speed"])],
    }
    graph_receipt.finish(receipt)
    a.out.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    for arm in [*pool_records, *arms]:
        print(f"{arm['name']:6s} {arm['execution']:5s} "
              + " ".join(f"{p['name']}={p['members']}/{p['choices']}" for p in arm["passes"])
              + f" long={arm['screen_long'][0]['members']}/{arm['screen_long'][0]['choices']}"
              + (f" replayed_everything={arm.get('replayed_everything')}" if arm in arms else ""))
    print("verdict:", receipt["verdict"])
    return 0 if receipt["verdict"] == "equal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
