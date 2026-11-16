"""Bind worker-start environment inputs to the frozen launch-lever policy."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from experiments.numerics_rows.common import Invalid, sha256

LEVERS = Path(__file__).with_name("launch-levers.tsv")


def digest(mapping):
    """SHA-256 of sorted compact ensure_ascii JSON, ASCII, with no final newline.

    The object maps included environment names to their exact string values.
    Absent names are omitted. Empty strings are retained. No value is parsed.
    """
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise Invalid("launch state must map names to exact strings")
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest()


class LaunchState:
    def __init__(self, path=LEVERS, *, fresh=False):
        if fresh and any(name == "tessera" or name.startswith("tessera.") for name in sys.modules):
            raise Invalid("launch state requires a fresh worker before its first Tessera import")
        self.path, self.policy = Path(path), {}
        for row in csv.DictReader(self.path.read_text().splitlines(), delimiter="\t"):
            name, policy = row.get("name"), row.get("launch_state")
            if not isinstance(name, str) or not name:
                raise Invalid("launch lever policy has no environment name")
            if policy not in ("include", "exclude", "not launch-changing") or self.policy.get(name, policy) != policy:
                raise Invalid("launch lever policy is malformed or inconsistent")
            self.policy[name] = policy
        if not self.policy or self.policy.get("TESSERA_SERVE_MODE") != "include":
            raise Invalid("launch lever policy must retain the residency selector")
        self.levers_sha256 = sha256(self.path)
        self.mapping = self.current()
        self.sha256 = digest(self.mapping)

    def current(self):
        return {k: v for k, v in sorted(os.environ.items()) if (
            k.startswith("TESSERA_") and self.policy.get(k) != "exclude" or self.policy.get(k) == "include")}

    def check(self):
        if self.current() != self.mapping:
            raise Invalid("launch state changed after worker start")

    def fields(self):
        return {"launch_state": dict(self.mapping), "launch_state_sha256": self.sha256,
                "launch_levers_sha256": self.levers_sha256}


def check_binding(record, expected):
    if not isinstance(record.get("launch_levers_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["launch_levers_sha256"]):
        raise Invalid("launch lever file digest is malformed")
    for field in ("launch_state", "launch_state_sha256", "launch_levers_sha256"):
        if record.get(field) != expected[field]:
            raise Invalid(f"C1 launch state or lever file differs: {field}")
    if digest(record["launch_state"]) != record["launch_state_sha256"]:
        raise Invalid("launch state object does not match its digest")
