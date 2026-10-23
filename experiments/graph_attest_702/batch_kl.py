"""Bind normalized live-arm KL evidence; never launch a scorer or alter graph equality.

This companion checks provenance and completeness, not a new KL threshold. The
collector/scorer owns the normalized arm report and its target-model handoff.
Even an internally consistent set of reports must match the prepared window,
its native attempts, and a freshly verified graph-equals-eager receipt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tessera import graph_receipt
from managed_window import Refused, atomic_json, read_json
import tp2_recipe as recipe

SCHEMA = "tessera.batch_kl_manifest.v1"
ARM_SCHEMA = "tessera.batch_kl_arm.v1"
NAMES = ("aE1", "aGR", "aE2")
IDENTITY = ("rank", "action_key", "nonce", "scope_id", "run_id", "input_sha256")


def _require(condition, message):
    if not condition:
        raise Refused(message)


def _same(actual, expected, field):
    _require(graph_receipt.canonical(actual) == graph_receipt.canonical(expected),
             f"{field} differs from the window's retained identity")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path):
    return dict(path=str(path), sha256=_sha(path))


def _artifact(directory, record, field):
    """Report paths are arm-local; a report cannot substitute another arm's bytes."""
    relative = Path(record["path"])
    _require(not relative.is_absolute() and ".." not in relative.parts,
             f"{field} must be an arm-relative path")
    path = directory / relative
    _require(path.resolve().is_relative_to(directory.resolve()), f"{field} escapes its arm")
    _same(_sha(path), record["sha256"], f"{field}.sha256")
    return path


def _flags(config, arm):
    # Use the serving recipe rather than duplicating its defaults here.
    argv = recipe.serve(config, arm, 0)
    value = lambda flag: argv[argv.index(flag) + 1]
    spec = json.loads(value("--speculative-config"))
    return dict(enforce_eager="--enforce-eager" in argv,
                compilation_config=(json.loads(value("--compilation-config"))
                                    if "--compilation-config" in argv else None),
                speculative_tokens=spec["num_speculative_tokens"],
                tensor_parallel_size=int(value("--tensor-parallel-size")),
                max_num_seqs=int(value("--max-num-seqs")),
                max_model_len=int(value("--max-model-len")), fabric=config["fabric"])


def _members(root, setup, input_sha):
    members = []
    for rank in (0, 1):
        member = read_json(root / f"outcome-rank{rank}.json")
        _same(member["rank"], rank, "native rank")
        _same(member["run_id"], setup["run_id"], "native run_id")
        _same(member["input_sha256"], input_sha, "native input_sha256")
        _require(member["simulation"] is False, "simulated native outcome")
        _same(member["returncode"], 0, "native returncode")
        _same(member["completed_arms"], list(NAMES), "native completed_arms")
        for field in IDENTITY:
            _require(field in member and member[field] != "", f"missing native {field}")
        _require(not member.get("error") and not member.get("cleanup_error"), "failed native outcome")
        members.append(member)
    for field in ("action_key", "nonce", "scope_id"):
        _require(members[0][field] != members[1][field], f"native ranks share {field}")
    gang = read_json(root / "native-gang.json")
    _same(gang["schema"], "prismabuild.pbgang.v1", "native gang schema")
    _same(gang["members"], [m["action_key"] for m in members], "native gang members")
    return members


def _ownership(directory, reported, members):
    _require(isinstance(reported, list) and len(reported) == 2, "both native members required")
    _same([m["rank"] for m in reported], [0, 1], "report ranks")
    for rank, (report, outcome) in enumerate(zip(reported, members)):
        owned = read_json(directory / f"ownership-rank{rank}.json")
        for field in IDENTITY:
            _same(owned[field], outcome[field], f"rank{rank} ownership {field}")
            _same(report[field], outcome[field], f"rank{rank} report {field}")
        # Preserve any additional native identity fields carried by the handoff.
        for field in ("host", "container_owner", "claimed_unix", "simulation", "returncode", "completed_arms"):
            if field in report:
                _same(report[field], outcome[field], f"rank{rank} {field}")
        cid = owned["container"]["cid"]
        _require(isinstance(cid, str) and bool(cid), f"rank{rank} missing owned container")
        _same(report["container_id"], cid, f"rank{rank} container_id")
        if "container_id" in outcome:
            _same(outcome["container_id"], cid, f"rank{rank} outcome container_id")
        if "container_id" in owned:
            _same(owned["container_id"], cid, f"rank{rank} ownership container_id")


def _complete(directory, report, expected_rows):
    _require(report["complete"] is True and report["status"] == "ok", "collector incomplete or refused")
    _same(report["rows"], expected_rows, "collector rows")
    dump = _artifact(directory, report["raw_dump"], "raw_dump")
    result_path = _artifact(directory, report["result"], "result")
    count = 0
    with dump.open() as stream:
        for line in stream:
            row = json.loads(line)
            _require(isinstance(row, dict), "raw_dump row is not an object")
            count += 1
    _same(count, expected_rows, "raw_dump rows")
    result = read_json(result_path)
    _require(result["status"] == "ok" and result["complete"] is True, "scorer result incomplete or refused")
    _same(result["rows"], expected_rows, "scorer result rows")


def _bind_arm(root, setup, graph, record, arm, members, input_sha):
    name = arm["arm"]
    directory = Path(setup["config"]["receipts"]) / name
    _same(str(directory.resolve()), str((root / "arms" / name).resolve()), "arm receipts directory")
    report_path = directory / "batch-kl.json"
    report = read_json(report_path)
    _same(report["schema"], ARM_SCHEMA, "arm schema")
    _same(report["arm"], name, "arm name")
    _same(report["run_id"], setup["run_id"], "report run_id")
    _same(report["input_sha256"], input_sha, "report input_sha256")
    _ownership(directory, report["members"], members)
    config = setup["config"]
    scope = dict(artifact=config["artifact"], model_config_sha256=config["config_sha256"],
                 image=config["image"], tessera_src_sha256=config["src_sha256"],
                 image_id=record["image_id"])
    _require(isinstance(scope["image_id"], str) and bool(scope["image_id"]), "missing actual image identity")
    _same(report["scope"], scope, "scope")
    _same(graph["model"]["path"], scope["artifact"], "graph artifact")
    _same(graph["model"]["config_sha256"], scope["model_config_sha256"], "graph config")
    _same(graph["runtime"]["image"], scope["image"], "graph image")
    _same(graph["tessera"]["src_sha256"], scope["tessera_src_sha256"], "graph source")
    _same(graph["runtime"]["fabric"], config["fabric"], "graph fabric")
    flags = _flags(config, arm)
    for field in ("requested", "observed"):
        _same(report[field], flags, field)
    _same(record["execution"], "eager" if flags["enforce_eager"] else "graph", "graph execution")
    for field in ("compilation_config", "speculative_tokens", "tensor_parallel_size", "max_num_seqs", "max_model_len"):
        _same(record[field], flags[field], f"graph {field}")
    _same(record["pb_action"], members[0]["action_key"], "graph pb_action")
    scoring = config["batch_kl"]["scoring"]
    for field in ("corpus_sha256", "teacher_sha256", "tokenizer_sha256", "regime"):
        _require(isinstance(scoring[field], str) and bool(scoring[field]), f"missing scoring {field}")
    rows = scoring["expected_rows"]
    _require(type(rows) is int and rows > 0, "expected_rows must be a positive integer")
    _same(report["scoring"], scoring, "scoring")
    provenance = report["target_provenance"]
    _same(provenance["model_role"], "target", "target model_role")
    _same(provenance["config_sha256"], scope["model_config_sha256"], "target config")
    _same(provenance["speculative_tokens"], flags["speculative_tokens"], "target MTP")
    _same(provenance["ranks"], [0, 1], "target ranks")
    _complete(directory, report, rows)
    # Reapply the authoritative gate, not receipt.verdict or its cached attests.
    if not flags["enforce_eager"]:
        serve = {key: flags[key] for key in ("compilation_config", "speculative_tokens", "tensor_parallel_size", "max_num_seqs", "max_model_len", "fabric")}
        serve.update({key: scope[key] for key in ("image", "model_config_sha256", "tessera_src_sha256")})
        why = graph_receipt.verify(graph, serve)
        _require(why is None, f"graph gate refused: {why}")
    return dict(report, report=_file(report_path))


def bind_batch_kl(root: Path, graph_receipt_path: Path) -> dict:
    """Persist a fail-closed companion. Missing/malformed evidence is an error, not a crash.

    The graph receipt is read-only and its not_measured declarations remain intact.
    Qualification here is identity/completeness binding, not a GPU measurement or
    a replacement for the scorer owner's live TP2/MTP target-model support.
    """
    root, graph_receipt_path = Path(root), Path(graph_receipt_path)
    bound = dict(schema=SCHEMA, qualified=False, errors=[], arms={})
    try:
        bound["graph_receipt"] = _file(graph_receipt_path)
        graph = read_json(graph_receipt_path)
        setup = read_json(root / "inputs.json")
        input_sha = _sha(root / "inputs.json")
        bound.update(run_id=setup["run_id"], input_sha256=input_sha)
        _same([arm["arm"] for arm in setup["arms"]], list(NAMES), "prepared arm population")
        _same([arm["eager"] for arm in setup["arms"]], ["1", "0", "1"], "prepared arm roles")
        members = _members(root, setup, input_sha)
        manifest = read_json(root / "receipt-manifest.json")
        _same(sorted(manifest), sorted(NAMES), "receipt manifest population")
        records = graph["eager_pool"] + graph["arms"]
        _same(sorted(record["name"] for record in records), sorted(NAMES), "graph arm population")
        _same([a["name"] for a in graph["arms"]], ["aGR"], "graph roles")
        by_name = {record["name"]: record for record in records}
        _require(len({record["image_id"] for record in records}) == 1, "actual images differ across arms")
        for arm in setup["arms"]:
            name = arm["arm"]
            try:
                _same(manifest[name]["action_key"], members[0]["action_key"], "receipt action_key")
                _same(manifest[name]["rank_action_keys"], [m["action_key"] for m in members], "receipt rank_action_keys")
                bound["arms"][name] = _bind_arm(root, setup, graph, by_name[name], arm, members, input_sha)
            except (Refused, OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                bound["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
        bound["qualified"] = not bound["errors"] and len(bound["arms"]) == len(NAMES)
    except (Refused, OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        bound["errors"].append(f"window: {type(exc).__name__}: {exc}")
    atomic_json(root / "batch-kl-manifest.json", bound)
    return bound


__all__ = ["bind_batch_kl"]
