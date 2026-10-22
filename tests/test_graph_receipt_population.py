"""CPU population/independence regressions for #1258, through the real producer.

These are protocol fixtures, not GPU/compiled qualification. The legacy AST
fallback reads the runner's own roster without executing its unsafe startup;
after extraction the fixture reads CASES from the import-safe runner instead.
No second case roster or receipt-supplied expected total lives in these tests.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import urllib.request

import pytest

from tessera import graph_receipt as gr

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/glm53_508_graph_qual/equal-508.py"
PRODUCER = ROOT / "experiments/graph_attest_702/receipt.py"
IMAGE = "example.invalid/fixture/vllm@sha256:" + "a" * 64
CC = {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}


def _cases():
    # Permit measuring the unfixed producer without importing the unsafe runner.
    # Only a top-level literal definition qualifies, never its long-screen branch.
    for node in ast.parse(RUNNER.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CASES" for t in node.targets
        ) and isinstance(node.value, ast.Dict):
            return ast.literal_eval(node.value)
    return runpy.run_path(str(RUNNER), run_name="_population_fixture")["CASES"]


def _write(path, value):
    path.write_text(json.dumps(value))


def _digest(choice):
    h = hashlib.sha256()
    h.update(json.dumps(choice["token_ids"]).encode())
    h.update(json.dumps(choice["logprobs"]["top_logprobs"], sort_keys=True).encode())
    return h.hexdigest()


def _pass(directory, name):
    summary = {"_admission": "paused"}
    for case, (specs, tokens) in _cases().items():
        choices = []
        for index, (length, offset) in enumerate(specs):
            choices.append({
                "index": index, "prompt_token_ids": list(range(offset, offset + length)),
                "token_ids": [42] * tokens,
                "logprobs": {"token_logprobs": [-0.5] * tokens,
                             "top_logprobs": [{str(t): -float(t) for t in range(20)}
                                              for _ in range(tokens)]},
            })
        _write(directory / f"{name}.eq.{case}.json", {"choices": choices})
        summary[case] = {"batch": len(specs), "lengths": [n for n, _ in specs],
                         "max_tokens": tokens, "digests": [_digest(c) for c in choices],
                         "elapsed_s": 0.1, "steps": {}}
    _write(directory / f"{name}.eq.summary.json", summary)


def _arm(receipts, model, name, eager):
    directory = receipts / name
    directory.mkdir()
    args = {
        "host": "cpu-fixture", "model": str(model), "image": IMAGE,
        "src_sha256": "b" * 64, "hooks_sha256": "c" * 64,
        "equal_script_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "eager": "1" if eager else "0", "kernel_json": "{}",
        "compilation_json": "null" if eager else json.dumps(CC),
        "serve_args": shlex.join(["vllm", "serve", str(model), "--tensor-parallel-size", "1",
                                  "--max-model-len", "8448", "--max-num-seqs", "8"]
                                 + (["--enforce-eager"] if eager else
                                    ["--compilation-config", json.dumps(CC)])),
    }
    (directory / f"engine-args-{name}.txt").write_text(
        "".join(f"{k}={v}\n" for k, v in args.items()))
    (directory / f"{name}.engine.log").write_text(
        f"Initializing a V1 LLM engine (v0.fixture) with config: enforce_eager={eager}\n")
    _write(directory / f"{name}.dispatch.1.json", {
        "captured": {"ModelCudaGraphManager@fixture": [1]},
        "counts": {"ModelCudaGraphManager|FULL|tokens=1|reqs=1": 1},
    })
    for p in (name, f"{name}-r2"):
        _pass(directory, p)


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    _write(model / "config.json", {"index_topk": 2048})
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    for name in ("e1", "e2", "g"):
        _arm(receipts, model, name, name != "g")
    manifest = tmp_path / "manifest.json"
    _write(manifest, {n: {"action_key": "fixture-" + n} for n in ("e1", "e2", "g")})
    return receipts, manifest, tmp_path / "receipt.json", model


def _produce(inputs, eager="e1,e2", graph="g"):
    receipts, manifest, out, _ = inputs
    result = subprocess.run([sys.executable, str(PRODUCER), str(receipts), str(manifest),
                             str(out), "--eager", eager, "--graph", graph],
                            text=True, capture_output=True, timeout=30)
    receipt = json.loads(out.read_text()) if out.exists() else None
    return result, receipt


def _serve(receipt):
    return {"image": IMAGE, "model_config_sha256": receipt["model"]["config_sha256"],
            "tessera_src_sha256": "b" * 64, "compilation_config": CC,
            "speculative_tokens": 0, "max_model_len": 8448, "max_num_seqs": 8,
            "tensor_parallel_size": 1, "fabric": "none"}


def _valid(inputs):
    result, receipt = _produce(inputs)
    assert result.returncode == 0, result.stdout + result.stderr
    assert receipt["verdict"] == "equal"
    assert gr.verify(receipt, _serve(receipt)) is None
    expected = sum(len(specs) for specs, _ in _cases().values())
    assert receipt["equality_set"]["choices_per_pass"] == expected
    assert all(p["choices"] == expected for a in receipt["arms"] for p in a["passes"])
    return receipt


def _text_keyed_maps_remain_valid(inputs):
    # The live runner requests top-20 but not ID-keyed map keys. Decoded
    # strings may collide; the sampled token may also add a 21st key. Do not
    # reintroduce the historical repair's incompatible 20-key minimum.
    for keys in (1, 19, 21, 20):
        for arm in ("e1", "e2", "g"):
            directory = inputs[0] / arm
            for name in (arm, f"{arm}-r2"):
                path = directory / f"{name}.eq.summary.json"
                summary = json.loads(path.read_text())
                for case in _cases():
                    raw = directory / f"{name}.eq.{case}.json"
                    record = json.loads(raw.read_text())
                    for choice in record["choices"]:
                        choice["logprobs"]["top_logprobs"] = [
                            {str(i): -float(i) for i in range(keys)}
                            for _ in choice["token_ids"]]
                    summary[case]["digests"] = [_digest(c) for c in record["choices"]]
                    _write(raw, record)
                _write(path, summary)
        _valid(inputs)


def _pooled_outcomes_remain_valid(inputs):
    # Select repeated requests by their actual owner-defined signature, not names.
    groups = {}
    for case, spec in _cases().items():
        groups.setdefault(json.dumps(spec), []).append(case)
    source, target = next(names[:2] for names in groups.values() if len(names) >= 2)

    def replace(arm, case, choice=None):
        raw = inputs[0] / arm / f"{arm}.eq.{case}.json"
        record = json.loads(raw.read_text())
        if choice is not None:
            record["choices"][0] = choice
            _write(raw, record)
            path = inputs[0] / arm / f"{arm}.eq.summary.json"
            summary = json.loads(path.read_text())
            summary[case]["digests"] = [_digest(c) for c in record["choices"]]
            _write(path, summary)
        return copy.deepcopy(record["choices"][0])

    unique = replace("e1", source)
    unique["token_ids"][0] = 43
    replace("e1", source, unique)
    replace("g", target, unique)
    _valid(inputs)  # graph mixes e2 outcomes and a different-case e1 outcome
    for mutation in ("token", "logprob", "batch-signature"):
        bad = copy.deepcopy(unique)
        if mutation == "token":
            bad["token_ids"][0] = 44
        elif mutation == "logprob":
            bad["logprobs"]["top_logprobs"][0]["0"] = -0.125
        else:
            bad["prompt_token_ids"][0] = 12345
        replace("g", target, bad)
        _refused(inputs)  # complete populations do not waive arithmetic membership
    replace("g", target, unique)
    _valid(inputs)


def _refused(inputs, **roles):
    out = inputs[2]
    if out.exists():
        out.unlink()  # never mistake the preceding valid receipt for new output
    result, receipt = _produce(inputs, **roles)
    assert result.returncode != 0, "producer falsely attested equality:\n" + result.stdout
    if receipt is not None:
        assert receipt["verdict"] != "equal"
        assert gr.verify(receipt, _serve(receipt)) is not None


def test_runner_exposes_its_population_without_startup_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("importing the equality runner performed startup IO")
    monkeypatch.setattr(sys, "argv", [str(RUNNER)])
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    try:
        namespace = runpy.run_path(str(RUNNER), run_name="_import_safety")
    except (IndexError, SystemExit) as exc:
        pytest.fail(f"runner consumes CLI arguments at import instead of exposing CASES: {exc}")
    assert namespace["CASES"]


@pytest.mark.parametrize("damage", [
    "positive-partial", "missing-case-file", "substituted-case", "duplicate-choice-index",
    "summary-batch", "summary-digest", "raw-prompt", "raw-generation-length",
    "self-consistent-wrong-geometry",
])
@pytest.mark.parametrize("role", ["g", "e1"])
def test_producer_requires_complete_consistent_population(inputs, damage, role):
    _valid(inputs)  # complete valid production and verify are exercised before every attack
    if damage == "positive-partial" and role == "g":
        _text_keyed_maps_remain_valid(inputs)
        _pooled_outcomes_remain_valid(inputs)
    directory = inputs[0] / role
    cases = _cases()
    for name in (role, f"{role}-r2"):
        path = directory / f"{name}.eq.summary.json"
        summary = json.loads(path.read_text())
        first = next(iter(cases))
        if damage == "positive-partial":
            for case in list(cases)[1:]:
                summary.pop(case)
                (directory / f"{name}.eq.{case}.json").unlink()
        elif damage == "missing-case-file":
            (directory / f"{name}.eq.{first}.json").unlink()
        elif damage == "substituted-case":
            summary["invented-case"] = summary.pop(first)
            (directory / f"{name}.eq.{first}.json").rename(
                directory / f"{name}.eq.invented-case.json")
        elif damage == "summary-batch":
            summary[first]["batch"] += 1
        elif damage == "summary-digest":
            summary[first]["digests"][0] = "0" * 64
        else:
            case = next(c for c, (specs, _) in cases.items() if len(specs) > 1)
            raw = directory / f"{name}.eq.{case}.json"
            record = json.loads(raw.read_text())
            if damage == "duplicate-choice-index":
                record["choices"][1]["index"] = record["choices"][0]["index"]
            elif damage == "raw-prompt":
                record["choices"][0]["prompt_token_ids"] = []
            elif damage == "self-consistent-wrong-geometry":
                record["choices"][0]["prompt_token_ids"].append(12345)
                summary[case]["lengths"][0] += 1
            else:
                record["choices"][0]["token_ids"].pop()
            _write(raw, record)
        _write(path, summary)
    if role == "g" and damage in {
        "raw-prompt", "raw-generation-length", "self-consistent-wrong-geometry"
    }:
        # Keep the membership predicate satisfied even for malformed geometry:
        # all runs share the corrupt raw input, but not the protocol population.
        case = next(c for c, (specs, _) in cases.items() if len(specs) > 1)
        for arm in ("e1", "e2"):
            for suffix in ("", "-r2"):
                (inputs[0] / arm / f"{arm}{suffix}.eq.{case}.json").write_bytes(
                    (inputs[0] / "g" / f"g{suffix}.eq.{case}.json").read_bytes())
                if damage == "self-consistent-wrong-geometry":
                    path = inputs[0] / arm / f"{arm}{suffix}.eq.summary.json"
                    summary = json.loads(path.read_text())
                    summary[case]["lengths"][0] += 1
                    _write(path, summary)
    # A matching long screen cannot replace equality cases.
    for arm in ("e1", "e2", "g"):
        directory = inputs[0] / arm
        _write(directory / f"{arm}-long.eq.summary.json", {
            "_admission": "paused", "screen": {"batch": 1}})
        _write(directory / f"{arm}-long.eq.screen.json", {"choices": [{
            "index": 0, "prompt_token_ids": [1] * 2100, "token_ids": [42],
            "logprobs": {"top_logprobs": [{"42": -0.5}]}}]})
    _refused(inputs)


@pytest.mark.parametrize("top", [
    {}, {str(i): -float(i) for i in range(22)}, {"text": float("inf")},
    {"text": True}, [],
], ids=["empty", "oversized", "nonfinite", "boolean", "nonmap"])
def test_producer_requires_valid_matching_raw_logprobs(inputs, top):
    _valid(inputs)
    # An invalid wire representation is not evidence even when every run agrees.
    # Recompute the digest to isolate raw validity from summary integrity.
    case = next(iter(_cases()))
    for arm in ("e1", "e2", "g"):
        directory = inputs[0] / arm
        for name in (arm, f"{arm}-r2"):
            path = directory / f"{name}.eq.summary.json"
            summary = json.loads(path.read_text())
            raw = directory / f"{name}.eq.{case}.json"
            record = json.loads(raw.read_text())
            for choice in record["choices"]:
                choice["logprobs"]["top_logprobs"] = [top for _ in choice["token_ids"]]
            summary[case]["digests"] = [_digest(c) for c in record["choices"]]
            _write(raw, record)
            _write(path, summary)
    _refused(inputs)


def test_producer_requires_both_eager_passes(inputs):
    _valid(inputs)
    # Graph membership remains complete against e2. A missing eager pass still
    # invalidates the declared population; it must not be silently dropped.
    (inputs[0] / "e1" / "e1-r2.eq.summary.json").unlink()
    _refused(inputs)


@pytest.mark.parametrize("attack", ["overlap", "duplicate-eager", "duplicate-graph", "pass-alias"])
def test_producer_requires_independent_role_and_pass_names(inputs, attack):
    _valid(inputs)
    if attack == "pass-alias":
        _arm(inputs[0], inputs[3], "g-r2", True)
        _refused(inputs, eager="g-r2")
    else:
        roles = {"overlap": {"graph": "e1"},
                 "duplicate-eager": {"eager": "e1,e1,e2"},
                 "duplicate-graph": {"graph": "g,g"}}[attack]
        _refused(inputs, **roles)


@pytest.mark.parametrize("reverse", [False, True])
def test_producer_refuses_conflicting_equality_basenames_in_either_order(inputs, reverse):
    _valid(inputs)
    case = next(iter(_cases()))
    name = "g.eq.summary.json"
    genuine = json.loads((inputs[0] / "g" / name).read_text())
    _write(inputs[0] / "e2" / name, genuine)
    _valid(inputs)  # byte-identical shared evidence is not a conflict
    forged = copy.deepcopy(genuine)
    forged[case]["digests"][0] = "0" * 64
    _write(inputs[0] / "e1" / name, forged)
    _refused(inputs, eager="e2,e1" if reverse else "e1,e2")


@pytest.mark.parametrize("attack", [
    "partial", "declared-total", "float-counters", "bool-counters", "duplicate-pass",
    "substituted-pass", "extra-pass", "eager-partial", "missing-eager",
    "overlap", "pass-alias", "duplicate-eager", "duplicate-graph", "declared-total-only",
    "eager-negative-members", "eager-excess-members", "eager-bool-members",
])
def test_verify_rejects_forged_population_and_independence_claims(inputs, attack):
    receipt = copy.deepcopy(_valid(inputs))
    graph = receipt["arms"][0]
    passes = graph["passes"]
    if attack in {"partial", "declared-total", "bool-counters", "float-counters"}:
        counter = {"partial": 1, "declared-total": 1,
                   "bool-counters": True, "float-counters": float(passes[0]["choices"])}[attack]
        for p in passes:
            p.update(members=counter, choices=counter)
        if attack == "declared-total":
            receipt["equality_set"]["choices_per_pass"] = counter
    elif attack == "declared-total-only":
        receipt["equality_set"]["choices_per_pass"] = 1
    elif attack.startswith("eager-") and attack.endswith("-members"):
        p = receipt["eager_pool"][0]["passes"][0]
        p["members"] = {"eager-negative-members": -1,
                        "eager-excess-members": p["choices"] + 1,
                        "eager-bool-members": True}[attack]
    elif attack == "duplicate-pass":
        passes[1] = copy.deepcopy(passes[0])
    elif attack == "substituted-pass":
        passes[1]["name"] = "unrelated-pass"
    elif attack == "extra-pass":
        passes.append(copy.deepcopy(passes[1]))
    elif attack == "eager-partial":
        for p in receipt["eager_pool"][0]["passes"]:
            p.update(members=1, choices=1)
    elif attack == "missing-eager":
        receipt["eager_pool"] = []
    elif attack == "overlap":
        receipt["eager_pool"][0]["name"] = graph["name"]
        receipt["eager_pool"][0]["passes"] = copy.deepcopy(passes)
    elif attack == "pass-alias":
        receipt["eager_pool"][0]["name"] = passes[1]["name"]
        receipt["eager_pool"][0]["passes"] = [
            {"name": passes[1]["name"], "members": passes[0]["choices"], "choices": passes[0]["choices"]},
            {"name": passes[1]["name"] + "-r2", "members": passes[0]["choices"], "choices": passes[0]["choices"]}]
    elif attack == "duplicate-eager":
        receipt["eager_pool"].append(copy.deepcopy(receipt["eager_pool"][0]))
    else:
        receipt["arms"].append(copy.deepcopy(graph))
    # Do not let a stale verdict/attests explain the refusal: explicitly forge them.
    receipt["verdict"] = "equal"
    for arm in receipt["arms"]:
        arm["equal"] = True
        arm["replayed_everything"] = True
    receipt["attests"] = [gr.attestation(receipt, arm) for arm in receipt["arms"]]
    why = gr.verify(receipt, _serve(receipt))
    assert isinstance(why, str) and why, f"verify accepted forged {attack} receipt"
