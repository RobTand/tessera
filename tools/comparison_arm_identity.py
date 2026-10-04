#!/usr/bin/env python3
"""Derive a serve-comparison arm's identities from its audit roster and config.

Published from the independently reviewed deployment intake (issue #885): the
comparison artifact-identity owner. Besides the single-arm derivation below it
owns the shared export/comparison identity used by the input intake and the
orchestration gates:

  read_artifact_roster   audit roster authenticated by SHA-256, bound to the
                         exact artifact path, duplicate-free, with config.json
                         byte/SHA agreement
  export_identity        SHA-bound audited export metadata: config, index and
                         serving-manifest content and byte agreement, exact
                         export commit and contract version, serialized shard,
                         metadata and full-file byte currencies kept separate
  read_comparison_arms   ordered closed-world owned arm population; frozen
                         bindings authenticate name/bytes before sourcing;
                         only shell-safe TS_PIN stems ([A-Za-z0-9_])
  comparison_artifact    frozen binding consumed at runtime before any
                         resource is taken; fieldless historical manifests
                         keep their explicit pathname-only check and a
                         malformed or drifting new binding never falls back

An arm env needs only ARM_ARTIFACT, ARM_AUDIT_ROSTER and
ARM_AUDIT_ROSTER_SHA256. The single-arm mode prints shell assignments for
every other ARM_* value the window uses that the arm env left unset; values
the env sets are kept and checked by the deployment preflight against the
same roster. Deriving or comparing this identity confers no quality, native,
admission or serve authorization; every existing gate keeps its owner.

The roster is authenticated by its SHA-256 before anything is read from it.
Derived values:
  ARM_CONFIG_SHA256 / ARM_INDEX_SHA256   roster files[] entries
  ARM_MANIFEST_SHA256                    roster manifest_sha256
  ARM_SHARDS                             count of roster *.safetensors entries
  ARM_DIGEST_CACHE / ARM_DIGEST_RECEIPT  $STAGE/<arm>/candidate-digest-cache.sparklina{,.receipt}.json
  ARM_UNSERVED_PRICED_TARGETS            quantization_config targets in layers >= num_hidden_layers
                                         (never constructed without a speculative config)
  ARM_SERVE_MODE=resident, ARM_MOE_BACKEND=triton

usage: comparison_arm_identity.py ARM   (reads ARM_* and STAGE from the environment)
       comparison_arm_identity.py comparison-artifact --manifest M --pin P --arm A.env [--arm ...]
"""
from __future__ import annotations

import hashlib
import argparse
import subprocess
import json
import os
import re
import shlex
import sys
from pathlib import Path


def fail(msg: str) -> int:
    print(f"comparison_arm_identity.py: {msg}", file=sys.stderr)
    return 2


def unserved_targets(config: dict) -> list[str]:
    text = config.get("text_config", config)
    layers = int(text["num_hidden_layers"])
    spec = re.compile(r"(^|\.)layers\.(\d+)\.")
    out = set()
    for group in (config.get("quantization_config") or {}).get("config_groups", {}).values():
        for target in group.get("targets", []):
            m = spec.search(target + ".")
            if m and int(m.group(2)) >= layers:
                out.add(target)
    return sorted(out)


def read_artifact_roster(artifact: Path, roster_path: Path, expected_sha256: str):
    raw = roster_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("audit roster sha256 differs from ARM_AUDIT_ROSTER_SHA256")
    roster = json.loads(raw)
    if roster.get("artifact") != str(artifact):
        raise ValueError(f"roster names {roster.get('artifact')}, not {artifact}")
    files = {row["name"]: row for row in roster["files"]}
    if len(files) != len(roster["files"]):
        raise ValueError("audit roster has duplicate file names")
    config_raw = (artifact / "config.json").read_bytes()
    if len(config_raw) != files["config.json"]["bytes"] or hashlib.sha256(config_raw).hexdigest() != files["config.json"]["sha256"]:
        raise ValueError("config.json on disk differs from the roster")
    return roster, files, config_raw


def export_identity(artifact: Path, roster_path: Path, expected_sha256: str):
    """Bind existing audited export metadata; never confer quality/native admission.

    Weight digests stay with the existing six-field digest-cache/live preflight
    owner. Serialized shard bytes, metadata bytes and resident estimates are
    different currencies. Export quantized_params is not the PQ eligible census.
    """
    roster, files, _ = read_artifact_roster(artifact, roster_path, expected_sha256)
    metadata = {}
    for name in ("model.safetensors.index.json", "tessera_serving_manifest.json"):
        raw = (artifact / name).read_bytes()
        row = files[name]
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"{name} on disk differs from the roster")
        metadata[name] = json.loads(raw)
    manifest = metadata["tessera_serving_manifest.json"]
    if files["tessera_serving_manifest.json"]["sha256"] != roster["manifest_sha256"]:
        raise ValueError("export manifest identity differs within roster")
    export_commit = manifest["git"]
    version = manifest["serving_gate"]["contract_version"]
    if not isinstance(export_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", export_commit):
        raise ValueError("export must name its exact Tessera commit")
    if type(version) is not int or version <= 0:
        raise ValueError("export contract version must be a positive integer")
    sizes = [row["bytes"] for row in files.values()]
    if any(type(size) is not int or size < 0 for size in sizes):
        raise ValueError("invalid audit byte count")
    weights = {name: row for name, row in files.items() if name.endswith(".safetensors")}
    indexed = set(metadata["model.safetensors.index.json"]["weight_map"].values())
    if indexed != set(weights):
        raise ValueError("export index and audited shard roster differ")
    weight_bytes = sum(row["bytes"] for row in weights.values())
    total = sum(sizes)
    if total != roster["all_files_bytes"] or weight_bytes != manifest["totals"]["checkpoint_bytes"]:
        raise ValueError("export serialized byte currencies disagree")
    return {"artifact": str(artifact), "audit_roster": str(roster_path),
            "audit_sha256": expected_sha256, "manifest_sha256": roster["manifest_sha256"],
            "config_sha256": files["config.json"]["sha256"],
            "index_sha256": files["model.safetensors.index.json"]["sha256"],
            "export_commit": export_commit, "contract_version": version,
            "serialized_weight_bytes": weight_bytes, "metadata_bytes": total - weight_bytes,
            "all_files_bytes": total, "export_quantized_params": manifest["totals"]["quantized_params"],
            "export_passthrough_bytes": manifest["totals"]["passthrough_bytes"],
            "source": manifest["source"], "plan_json": manifest.get("plan_json"),
            "input_scales_from": manifest.get("input_scales_from")}



def read_comparison_arms(paths, pin, expected_inputs=None):
    """Read actual owned arm files; a frozen binding authenticates bytes before sourcing."""
    paths = [Path(path).resolve() for path in paths]
    if expected_inputs is not None and len(paths) != len(expected_inputs):
        raise ValueError("comparison arm population differs from frozen binding")
    arms = []
    for index, path in enumerate(paths):
        if path.suffix != '.env' or not re.fullmatch(r'[A-Za-z0-9_]+', path.stem):
            raise ValueError("comparison arm cannot form a shell-compatible TS_PIN identifier")
        raw = path.read_bytes()
        source = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        if expected_inputs is not None:
            expected = expected_inputs[index]
            if (path.stem != Path(expected["path"]).stem or source["sha256"] != expected["sha256"]
                    or source["bytes"] != expected["bytes"]):
                raise ValueError("actual loaded arm name/bytes differ from frozen binding")
        command = ('set -e; set -a; . "$1"; "$2" -c ' + shlex.quote(
            'import json,os;print(json.dumps({k:os.environ[k] for k in '
            '("ARM_ARTIFACT","ARM_AUDIT_ROSTER","ARM_AUDIT_ROSTER_SHA256")}))'))
        environment = {key: value for key, value in os.environ.items() if not key.startswith('ARM_')}
        environment['VAL787_PIN'] = pin
        try:
            values = json.loads(subprocess.check_output(
                ['bash', '-c', command, 'comparison-arm', str(path), sys.executable], env=environment))
        except subprocess.CalledProcessError as exc:
            raise ValueError("owned comparison arm did not declare its artifact identity") from exc
        if path.read_bytes() != raw:
            raise ValueError("owned arm changed while reading its artifact identity")
        arms.append((path.stem, values, source))
    if not arms or len({arm[0] for arm in arms}) != len(arms):
        raise ValueError("comparison arms must have distinct names")
    return arms


def comparison_artifact(manifest_path, arm_paths, pin):
    """Only input/model binding: existing artifact/native/quality gates still apply."""
    identities = json.loads(Path(manifest_path).read_bytes())["identities"]
    if "tessera_artifact_binding" not in identities:
        # Retained v6/7410 historical manifests predate export bindings. Preserve
        # their pathname check explicitly, not as a fallback from a bad new binding.
        arms = read_comparison_arms(arm_paths, pin)
        if any(arm[1]["ARM_ARTIFACT"] != identities["tessera_artifact"] for arm in arms):
            raise ValueError("historical comparison artifact differs from actual arm artifact")
        return identities["tessera_artifact"]
    binding = identities["tessera_artifact_binding"]
    if not isinstance(binding, dict) or not isinstance(binding.get("arm_inputs"), list):
        raise ValueError("frozen export/arm binding is malformed")
    arms = read_comparison_arms(arm_paths, pin, binding["arm_inputs"])
    values = arms[0][1]
    if any(arm[1] != values for arm in arms):
        raise ValueError("actual matched arms bind different artifact/audit identities")
    if (values["ARM_ARTIFACT"] != identities["tessera_artifact"]
            or values["ARM_ARTIFACT"] != binding["artifact"]
            or values["ARM_AUDIT_ROSTER"] != binding["audit_roster"]
            or values["ARM_AUDIT_ROSTER_SHA256"] != binding["audit_sha256"]):
        raise ValueError("actual artifact/audit differs from frozen export identity")
    actual = export_identity(Path(values["ARM_ARTIFACT"]), Path(values["ARM_AUDIT_ROSTER"]),
                             values["ARM_AUDIT_ROSTER_SHA256"])
    expected = {key: value for key, value in binding.items() if key != "arm_inputs"}
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("actual audited export metadata differs from frozen binding")
    return actual["artifact"]



def main(argv: list[str]) -> int:
    if argv[:1] == ["comparison-artifact"]:
        parser = argparse.ArgumentParser(description="Bind actual arm/audit bytes to an existing frozen comparison")
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--pin", required=True)
        parser.add_argument("--arm", action="append", required=True)
        args = parser.parse_args(argv[1:])
        try:
            print(comparison_artifact(args.manifest, args.arm, args.pin))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return fail(str(exc))
        return 0
    if len(argv) != 1:
        return fail("usage: comparison_arm_identity.py ARM")
    arm = argv[0]
    need = [k for k in ("ARM_ARTIFACT", "ARM_AUDIT_ROSTER", "ARM_AUDIT_ROSTER_SHA256", "STAGE") if not os.environ.get(k)]
    if need:
        return fail(f"unset: {' '.join(need)}")
    artifact = Path(os.environ["ARM_ARTIFACT"])
    try:
        roster, files, config_raw = read_artifact_roster(artifact, Path(os.environ["ARM_AUDIT_ROSTER"]),
                                                     os.environ["ARM_AUDIT_ROSTER_SHA256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return fail(str(exc))
    stage_arm = Path(os.environ["STAGE"]) / arm.lower()
    derived = {
        "ARM_CONFIG_SHA256": files["config.json"]["sha256"],
        "ARM_INDEX_SHA256": files["model.safetensors.index.json"]["sha256"],
        "ARM_MANIFEST_SHA256": roster["manifest_sha256"],
        "ARM_SHARDS": str(sum(1 for n in files if n.endswith(".safetensors"))),
        "ARM_DIGEST_CACHE": str(stage_arm / "candidate-digest-cache.sparklina.json"),
        "ARM_DIGEST_RECEIPT": str(stage_arm / "candidate-digest-cache.sparklina.receipt.json"),
        "ARM_UNSERVED_PRICED_TARGETS": " ".join(unserved_targets(json.loads(config_raw))),
        "ARM_SERVE_MODE": "resident",
        "ARM_MOE_BACKEND": "triton",
    }
    filled = []
    for key, value in derived.items():
        if os.environ.get(key):
            continue
        print(f"{key}={shlex.quote(value)}")
        filled.append(key)
    print(f"ARM_DERIVED={shlex.quote(' '.join(filled))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
