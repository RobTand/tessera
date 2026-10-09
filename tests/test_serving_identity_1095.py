"""The published serving identity for the qualified serving commit (tessera#1095).

The packet names the producer commit, the serving commit, the packaged
contract digest, and the tessera.package_source.v1 digest of that exact
source tree. Each test recomputes its value from the committed blobs.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from tessera.serving.source_identity import (
    SOURCE_IDENTITY_ALGORITHM,
    serving_source_sha256,
)
from tessera.source_profiles import ENCODER_SOURCE_V1, source_profiles

PACKET = Path(__file__).resolve().parents[1] / "docs" / "measurements" / "2026-10-09-serving-identity-fca4c6ce0.json"
SERVING_DIR = "src/tessera/serving/"
CONTRACT_PATH = "src/tessera/serving/runtime_contract.json"


def _packet():
    return json.loads(PACKET.read_bytes())


def _blob(commit, path):
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=Path(__file__).resolve().parents[1],
        stderr=subprocess.STDOUT,
    )


def test_the_packet_names_full_producer_and_serving_commits():
    packet = _packet()
    assert re.fullmatch(r"[0-9a-f]{40}", packet["producer_commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", packet["serving_commit"])
    assert packet["serving_commit"] == "fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb"


def test_the_v1_digest_reproduces_through_the_serving_api(tmp_path):
    packet = _packet()
    assert packet["algorithm"] == SOURCE_IDENTITY_ALGORITHM
    serving = packet["serving_commit"]
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", serving, "src/tessera"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).split()
    code = [n for n in names if Path(n).suffix in (".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")]
    assert len(code) == packet["file_count"]
    root = tmp_path / "src"
    for name in code:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_blob(serving, name))
    digest = serving_source_sha256(root)
    assert digest == packet["digest"]


def test_the_contract_digest_is_the_raw_packaged_bytes():
    packet = _packet()
    raw = _blob(packet["serving_commit"], CONTRACT_PATH)
    assert len(raw) == packet["contract"]["bytes"]
    assert hashlib.sha256(raw).hexdigest() == packet["contract"]["sha256"]
    assert json.loads(raw)["contract_version"] == packet["contract"]["version"]


def test_the_installed_projection_reproduces_the_worker_digest():
    packet = _packet()
    serving = packet["serving_commit"]
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", serving, "src/tessera"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).split()
    shipped = [n for n in names
               if Path(n).suffix in (".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")
               and "/_dev/" not in n]
    assert len(shipped) == packet["installed_projection"]["file_count"]
    profiles = source_profiles(
        ((Path(n).relative_to("src").as_posix(), _blob(serving, n)) for n in shipped),
        legacy_profile=SOURCE_IDENTITY_ALGORITHM,
        legacy_prefix=(SOURCE_IDENTITY_ALGORITHM.encode() + b"\0"),
    )
    assert profiles[SOURCE_IDENTITY_ALGORITHM] == packet["installed_projection"]["digest"]


def test_producer_identity_uses_its_own_recipe(tmp_path):
    packet = _packet()
    assert packet["producer_identity"]["algorithm"] == ENCODER_SOURCE_V1
    assert packet["producer_identity"]["algorithm"] != packet["algorithm"]
    serving = packet["producer_commit"]
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", serving, "src/tessera"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).split()
    code = [n for n in names if Path(n).suffix in (".py", ".cu", ".cuh", ".cpp", ".h")]
    assert len(code) == packet["producer_identity"]["file_count"]
    profiles = source_profiles(
        ((Path(n).relative_to("src/tessera").as_posix(), _blob(serving, n)) for n in code),
        legacy_profile=ENCODER_SOURCE_V1,
    )
    assert profiles[ENCODER_SOURCE_V1] == packet["producer_identity"]["digest"]
    assert profiles[ENCODER_SOURCE_V1] != packet["digest"]


def test_the_qualification_record_exists_at_the_serving_commit():
    packet = _packet()
    record = packet["qualification_record"]
    raw = _blob(packet["serving_commit"], record["path"])
    body = json.loads(raw)
    assert body["schema"] == record["schema"]
    assert body["action_key"] == record["action_key"]
