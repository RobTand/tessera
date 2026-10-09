"""The serving identity the qualification action measured (tessera#1095).

The packet names the source tree that action 6a5389858382 ran: snapshot
da97efbb37f44daded3efd8cb173bd953a1f8c37 whose parent 9eef9fea6e is the
serving and producer commit. The packet vendors the sealed snapshot block
with the sealed request digest, so each check reads committed bytes only.
Each test recomputes its value from the committed packet or blobs.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from tessera.serving.source_identity import (
    SOURCE_IDENTITY_ALGORITHM,
    serving_source_sha256,
)
from tessera.source_profiles import ENCODER_SOURCE_V1, source_profiles

PACKET = Path(__file__).resolve().parents[1] / "docs" / "measurements" / "2026-10-09-serving-identity-9eef9fea6.json"
CONTRACT_PATH = "src/tessera/serving/runtime_contract.json"
ACTION = "6a53898583824d17b622a54e4722b88b6d2b2b2c052ffa2ce31392fabeb617ed"
SNAPSHOT = "da97efbb37f44daded3efd8cb173bd953a1f8c37"
PARENT = "9eef9fea6edce32f4e64abf87f0058b11dab2287"
SUPERSEDED = "fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb"
SEALED_REQUEST_SHA256 = "55390c025d6758eb03df649cdf0d5573db27f7f284e7eb91af893ea48a9cda01"


def _packet():
    return json.loads(PACKET.read_bytes())


def _blob(commit, path):
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=Path(__file__).resolve().parents[1],
        stderr=subprocess.STDOUT,
    )


def _tree_names(commit):
    return subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "src/tessera"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).split()


def test_the_packet_names_full_producer_and_serving_commits():
    packet = _packet()
    assert re.fullmatch(r"[0-9a-f]{40}", packet["producer_commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", packet["serving_commit"])
    assert packet["serving_commit"] == PARENT
    assert packet["producer_commit"] == PARENT


def test_the_action_snapshot_matches_the_named_source_tree():
    packet = _packet()
    action = packet["action"]
    assert action["key"] == ACTION
    assert action["snapshot_commit"] == SNAPSHOT
    assert action["snapshot_parent"] == PARENT
    assert action["status"] == "executed"
    assert action["src_tree_diff_vs_parent"] == []
    assert action["sealed_request"] is None
    snap = action["sealed_snapshot"]
    assert snap["sealed_request_sha256"] == SEALED_REQUEST_SHA256
    assert snap["commit"] == SNAPSHOT
    assert snap["parent"] == PARENT
    assert snap["input"]["sha256"] == action["input"]["sha256"]
    assert snap["input"]["bytes"] == action["input"]["bytes"]


def test_the_v1_digest_reproduces_through_the_serving_api(tmp_path):
    packet = _packet()
    assert packet["algorithm"] == SOURCE_IDENTITY_ALGORITHM
    serving = packet["serving_commit"]
    code = [n for n in _tree_names(serving)
            if Path(n).suffix in (".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")]
    assert len(code) == packet["file_count"]
    root = tmp_path / "src"
    for name in code:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_blob(serving, name))
    assert serving_source_sha256(root) == packet["digest"]


def test_the_action_tree_differs_from_the_superseded_merge():
    packet = _packet()
    assert packet["supersedes"]["commit"] == SUPERSEDED
    action_names = {n for n in _tree_names(PARENT)
                    if Path(n).suffix in (".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")}
    merged_names = {n for n in _tree_names(SUPERSEDED)
                    if Path(n).suffix in (".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")}
    assert merged_names - action_names == {
        "src/tessera/residency_plan.py",
        "src/tessera/serving/csrc/mhc_fused.cu",
        "src/tessera/serving/mhc_fusion.py",
        "src/tessera/window_geometry.py",
    }


def test_the_contract_digest_is_the_raw_packaged_bytes():
    packet = _packet()
    raw = _blob(packet["serving_commit"], CONTRACT_PATH)
    assert len(raw) == packet["contract"]["bytes"]
    assert hashlib.sha256(raw).hexdigest() == packet["contract"]["sha256"]
    assert json.loads(raw)["contract_version"] == packet["contract"]["version"]


def test_the_installed_projection_uses_the_action_tree():
    packet = _packet()
    serving = packet["serving_commit"]
    shipped = [n for n in _tree_names(serving)
               if Path(n).suffix in (".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")
               and "/_dev/" not in n]
    assert len(shipped) == packet["installed_projection"]["file_count"]
    profiles = source_profiles(
        ((Path(n).relative_to("src").as_posix(), _blob(serving, n)) for n in shipped),
        legacy_profile=SOURCE_IDENTITY_ALGORITHM,
        legacy_prefix=(SOURCE_IDENTITY_ALGORITHM.encode() + b"\0"),
    )
    assert profiles[SOURCE_IDENTITY_ALGORITHM] == packet["installed_projection"]["digest"]


def test_producer_identity_uses_its_own_recipe():
    packet = _packet()
    assert packet["producer_identity"]["algorithm"] == ENCODER_SOURCE_V1
    assert packet["producer_identity"]["algorithm"] != packet["algorithm"]
    serving = packet["producer_commit"]
    code = [n for n in _tree_names(serving)
            if Path(n).suffix in (".py", ".cu", ".cuh", ".cpp", ".h")]
    assert len(code) == packet["producer_identity"]["file_count"]
    profiles = source_profiles(
        ((Path(n).relative_to("src/tessera").as_posix(), _blob(serving, n)) for n in code),
        legacy_profile=ENCODER_SOURCE_V1,
    )
    assert profiles[ENCODER_SOURCE_V1] == packet["producer_identity"]["digest"]
    assert profiles[ENCODER_SOURCE_V1] != packet["digest"]


def test_the_qualification_record_names_the_bound_action():
    packet = _packet()
    record = packet["qualification_record"]
    raw = _blob(record["published_by"], record["path"])
    assert hashlib.sha256(raw).hexdigest() == record["record_sha256"]
    body = json.loads(raw)
    assert body["schema"] == record["schema"]
    assert body["action_key"] == record["action_key"] == ACTION
    assert body["weights_loaded"] is False
    assert body["forward_executed"] is False
