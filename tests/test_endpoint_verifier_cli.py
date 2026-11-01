"""Check consumer expectations and offline byte proof through the public CLI."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tessera import endpoint_witness as ew
from _endpoint_fixture import expectation_args, expectations, receipt, resign

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "verify_endpoint_witness.py"

# Exercise the real entry point. Any network or serving import is a failure.
OFFLINE_DRIVER = """import importlib.abc, runpy, socket, sys, urllib.request
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in ('torch', 'vllm', 'transformers')
                or fullname.startswith('tessera.serving')
                or fullname == 'tessera.endpoint_observer'):
            raise RuntimeError('offline import: ' + fullname)
def no_network(*args, **kwargs):
    raise RuntimeError('offline network request')
sys.meta_path.insert(0, Block())
socket.socket.connect = no_network
urllib.request.urlopen = no_network
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "artifact"
    value = receipt(root)
    witness = tmp_path / "witness.json"
    witness.write_text(json.dumps(value))
    args = [str(witness), "--served-dir", str(root), *expectation_args(value, tmp_path), "--offline"]
    return root, value, witness, args


def run(args, *, offline=True):
    command = [sys.executable, "-c", OFFLINE_DRIVER, str(CLI)] if offline else [sys.executable, str(CLI)]
    result = subprocess.run([*command, *args], capture_output=True, text=True, timeout=45)
    verdict = json.loads(result.stdout)  # Refuse text, zero objects, or multiple objects.
    assert verdict["schema"] == "tessera.endpoint_witness_verdict.v1"
    assert result.stderr == "", result.stderr
    return result, verdict


def refused(args, fact, *, offline=True):
    result, verdict = run(args, offline=offline)
    assert result.returncode != 0
    assert verdict["verdict"] == "refused"
    assert verdict["proof_scope"] is None
    assert verdict["current_endpoint_verified"] is False
    assert fact in verdict["reason"]


def test_offline_cli_proves_recorded_bytes_without_current_endpoint_claim(sample):
    _, _, _, args = sample
    result, verdict = run(args)
    assert result.returncode == 0, verdict
    assert verdict["verdict"] == "valid"
    assert verdict["mode"] == "offline"
    assert verdict["proof_scope"] == "recorded_runtime_byte_binding"
    assert verdict["current_endpoint_verified"] is False
    assert verdict["reason"] is None


def test_offline_api_requires_expectations_but_not_a_live_listener(sample):
    root, value, _, _ = sample
    assert ew.verify_recorded_witness(value, expected=expectations(value), served_dir=root) is None
    assert "expected" in ew.verify_recorded_witness(value, expected=None, served_dir=root)
    assert "artifact directory" in ew.verify_recorded_witness(value, expected=expectations(value))
    assert "live runtime binding" in ew.verify_witness(value, served_dir=root)


@pytest.mark.parametrize("flag", ["endpoint", "alias", "artifacts", "tokenizer", "attempt", "ranks"])
@pytest.mark.parametrize("mode", ["offline", "live"])
def test_each_expected_fact_is_required_before_any_endpoint_request(sample, flag, mode):
    _, _, _, args = sample
    index = args.index("--expect-" + flag)
    del args[index:index + 2]
    if mode == "live":
        args.remove("--offline")
    refused(args, "--expect-" + flag)


@pytest.mark.parametrize("fact", ["endpoint", "alias", "artifacts", "tokenizer", "attempt", "ranks"])
@pytest.mark.parametrize("mode", ["offline", "live"])
def test_conflicting_expected_facts_refuse_before_any_endpoint_request(sample, fact, mode):
    _, _, _, args = sample
    index = args.index("--expect-" + fact) + 1
    if fact in ("artifacts", "tokenizer"):
        path = Path(args[index])
        expected = json.loads(path.read_text())
        files = expected if fact == "artifacts" else expected["files"]
        files[next(iter(files))]["sha256"] = "f" * 64
        path.write_text(json.dumps(expected))
    else:
        args[index] = {"endpoint": "http://127.0.0.1:9001", "alias": "other", "attempt": "other", "ranks": "0"}[fact]
    if mode == "live":
        args.remove("--offline")
    refused(args, {"alias": "alias", "artifacts": "artifact", "attempt": "attempt"}.get(fact, fact))


@pytest.mark.parametrize("part", ["files", "backend", "vocab", "special_ids"])
def test_each_tokenizer_expectation_is_required(sample, part):
    _, _, _, args = sample
    path = Path(args[args.index("--expect-tokenizer") + 1])
    expected = json.loads(path.read_text())
    del expected[part]
    path.write_text(json.dumps(expected))
    refused(args, "expected tokenizer")


@pytest.mark.parametrize("part", ["backend", "vocab", "special_ids"])
def test_tokenizer_facts_with_equal_vocab_size_cannot_replace_expected_facts(sample, part):
    _, _, _, args = sample
    path = Path(args[args.index("--expect-tokenizer") + 1])
    expected = json.loads(path.read_text())
    if part == "backend":
        expected[part]["model"]["vocab"] = {"<s>": 1, "a": 0}
    elif part == "vocab":
        expected[part] = {"<s>": 1, "a": 0}
    else:
        expected[part]["bos"] = 1
    path.write_text(json.dumps(expected))
    refused(args, "expected tokenizer")


@pytest.mark.parametrize("ranks", ["", "1", "1,0", "0,0", "0,2", "0,1,2", "0,x"])
def test_incomplete_or_invalid_expected_ranks_refuse(sample, ranks):
    _, _, _, args = sample
    args[args.index("--expect-ranks") + 1] = ranks
    refused(args, "ranks")


@pytest.mark.parametrize("evidence", ["alias_only", "size_only", "missing_rank", "missing_loader_input"])
def test_partial_runtime_evidence_cannot_supply_recorded_byte_proof(sample, evidence):
    _, value, witness, args = sample
    if evidence == "alias_only":
        value = {"listener": {"served_alias": "fixture"}}
    elif evidence == "size_only":
        for rank in value["artifacts"]:
            del rank["models"][0]["files"]["model.safetensors"]["sha256"]
        resign(value)
    elif evidence == "missing_rank":
        value["artifacts"].pop()
        resign(value)
    else:
        for rank in value["artifacts"]:
            rank["models"][0]["inputs"] = []
        resign(value)
    witness.write_text(json.dumps(value))
    refused(args, {"alias_only": "receipt", "size_only": "loaded source", "missing_rank": "incomplete",
                   "missing_loader_input": "loader inputs"}[evidence])


@pytest.mark.parametrize("fact", ["artifacts", "tokenizer"])
@pytest.mark.parametrize("change", ["size_only", "absent_file", "extra_file", "bool_size"])
def test_expectations_require_complete_file_bytes_not_just_sizes(sample, fact, change):
    _, _, _, args = sample
    path = Path(args[args.index("--expect-" + fact) + 1])
    expected = json.loads(path.read_text())
    files = expected if fact == "artifacts" else expected["files"]
    name = next(iter(files))
    if change == "size_only":
        del files[name]["sha256"]
    elif change == "absent_file":
        del files[name]
    elif change == "extra_file":
        files["unused.bin"] = copy.deepcopy(files[name])
    else:
        files[name]["bytes"] = True
    path.write_text(json.dumps(expected))
    refused(args, "expected " + ("artifact" if fact == "artifacts" else "tokenizer"))


@pytest.mark.parametrize("name", ["model.safetensors", "tokenizer.json"])
@pytest.mark.parametrize("dev_mode", [None, "1", "0"])
def test_same_size_corrupt_bytes_refuse_in_every_dev_mode(sample, monkeypatch, name, dev_mode):
    root, _, _, args = sample
    if dev_mode is None:
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    else:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    path = root / name
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    refused(args, "bytes differ")


@pytest.mark.parametrize("dev_mode", [None, "1", "0"])
def test_consumer_runtime_expectations_do_not_become_suspended_source_seals(sample, monkeypatch, dev_mode):
    _, _, _, args = sample
    if dev_mode is None:
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    else:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    args[args.index("--expect-attempt") + 1] = "another-runtime-attempt"
    refused(args, "expected attempt")


def test_separate_tokenizer_directory_keeps_its_byte_proof(sample, tmp_path):
    root, _, _, args = sample
    token_root = tmp_path / "tokenizer"
    token_root.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).rename(token_root / name)
    args.extend(["--tokenizer-dir", str(token_root)])
    result, verdict = run(args)
    assert result.returncode == 0, verdict
    assert verdict["verdict"] == "valid"
    path = token_root / "tokenizer.json"
    path.write_bytes(path.read_bytes() + b"x")
    refused(args, "tokenizer file bytes differ")


@pytest.mark.parametrize("change", ["missing_witness", "invalid_json", "unknown_argument", "missing_directory"])
def test_cli_input_errors_return_one_json_refusal(sample, change):
    _, _, witness, args = sample
    if change == "missing_witness":
        witness.unlink()
    elif change == "invalid_json":
        witness.write_text("{")
    elif change == "unknown_argument":
        args.append("--unknown")
    else:
        index = args.index("--served-dir")
        del args[index:index + 2]
    refused(args, "--served-dir" if change == "missing_directory" else "")


def test_default_live_mode_refuses_a_stopped_listener(sample):
    from test_endpoint_observer import server

    root, value, witness, args = sample
    with server(value, root.parent / "public") as (_, observed, _):
        witness.write_text(json.dumps(observed))
        args = [str(witness), "--served-dir", str(root), *expectation_args(observed, root.parent)]
    refused(args, "runtime evidence is unavailable", offline=False)
