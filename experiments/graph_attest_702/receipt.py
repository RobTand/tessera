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
    """Captured and replayed FULL sizes per manager class, and Tessera's per-class counts."""
    captured, replayed = defaultdict(set), defaultdict(lambda: defaultdict(int))
    classes = {"captured": {}, "replays": {}}
    for path in sorted(arm_dir.glob(f"{arm}.dispatch.*.json")):
        rec = json.loads(path.read_text())
        for owner, sizes in rec.get("captured", {}).items():
            if isinstance(sizes, list):
                captured[owner.split("@")[0]].update(int(s) for s in sizes)
        for key, count in rec.get("counts", {}).items():
            if (m := _COUNT.match(key)):
                replayed[m["manager"]][int(m["tokens"])] += count
        tc = rec.get("tessera_classes") or {}
        for manager, by_bound in tc.get("captured", {}).items():
            classes["captured"].setdefault(manager, {}).update(by_bound)
        for key, count in tc.get("replays", {}).items():
            classes["replays"][key] = max(classes["replays"].get(key, 0), count)
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


def arm_record(receipts, manifest, flat, arm, pool) -> dict:
    d = receipts / arm
    args = engine_args(d, arm)
    log = (d / f"{arm}.engine.log").read_text(errors="replace") if (d / f"{arm}.engine.log").exists() else ""
    serve = read_serve_log(log)
    spec = json.loads(args["spec_json"]) if args.get("spec_json") else None
    comp = json.loads(args["compilation_json"]) if args.get("compilation_json") else None
    return {
        "name": arm, "pb_action": manifest.get(arm, {}).get("action_key"), "host": args.get("host"),
        "execution": "eager" if args.get("eager") == "1" else "graph",
        "compilation_config": comp,
        "kernel_config": json.loads(args["kernel_json"]) if args.get("kernel_json") else None,
        "speculative_tokens": int(spec["num_speculative_tokens"]) if spec else 0,
        "max_model_len": int(args["max_model_len"]), "max_num_seqs": int(args["max_num_seqs"]),
        "tensor_parallel_size": 1,
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
    ap.add_argument("--image", default="localhost/prismaquant/spark-vllm-nccl230@sha256:"
                    "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a")
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    eager, graph = a.eager.split(","), a.graph.split(",")
    flat = flat_dir(a.receipts, eager + graph)
    pool_records = [arm_record(a.receipts, manifest, flat, e, [p for p in eager if p != e])
                    for e in eager]
    arms = [arm_record(a.receipts, manifest, flat, g, eager) for g in graph]
    src = {r["src_sha256"] for r in pool_records + arms}
    models = {r["model"] for r in pool_records + arms}
    if len(src) != 1 or len(models) != 1:
        raise SystemExit(f"arms differ in Tessera source {src} or model {models}: not one measurement")
    model = models.pop()
    config = pathlib.Path(model) / "config.json"
    text = json.loads(config.read_text())
    text = text.get("text_config", text)
    receipt = {
        "schema": graph_receipt.SCHEMA, "issue": "tessera#702",
        "runtime": {"image": a.image, "vllm": arms[0]["vllm"], "interface": "nightly-20260929"},
        "tessera": {"commit": a.commit, "src_sha256": src.pop()},
        "model": {"path": model, "config_sha256": sha256(config), "index_topk": text.get("index_topk")},
        "equality_set": {"name": "tessera#508", "script_sha256": sha256(QUAL / "equal-508.py"),
                         "choices_per_pass": 48},
        "eager_pool": pool_records, "arms": arms,
        "screens": {"long_context": {
            "why": "above index_topk eager is not repeat-exact (stock top-k arrival order); "
                   "membership against the eager pool's outcomes is a screen, never the verdict",
            "eager_pool": {r["name"]: r["screen_long"] for r in pool_records},
            "arms": {r["name"]: r["screen_long"] for r in arms}}},
        "not_measured": ["tensor_parallel_size 2", "the full GLM-5.3 artifact (u1 stub B, 8 layers)",
                         "served KL against BF16 under graphs", "graph-vs-eager speed"],
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
