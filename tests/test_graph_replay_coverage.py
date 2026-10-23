"""CPU acceptance for #1260: replay evidence is local, complete and integral.

The real receipt CLI consumes complete equality populations from graph_equality's
roster. Every attack first produces AND verifies a valid receipt; this is not a
second equality implementation or CUDA qualification.

Proposed additive evidence contract (shared with the parent's hook tests):
* dispatch JSON's owner_evidence maps full Class@id identities to
  {capture_complete: bool, capture_error: null | str, captured: [descriptor],
   replays: [{**descriptor, count: nonnegative int}]}.
* descriptor = {mode: "FULL", tokens: positive int, num_reqs: positive int,
  max_seq_len: positive int | null}; null denotes unsplit capture.
* graph.dispatches retains those JSON objects by dispatch basename, preserving
  filename rank/PID and owner identity. Legacy managers/classes are diagnostic
  marginals, NEVER sufficient evidence of owner-local tuple coverage.

Every required rank/process/owner must discharge its own captured descriptors.
Counters are integral JSON numbers, never bools or coercions. Main accepts the
extension but drops it; direct fixtures supply it only when absent. After the
behavior assertion, every case also requires the producer itself to retain it.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from tessera import graph_equality as ge
from tessera import graph_receipt as gr

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "experiments/graph_attest_702/receipt.py"
RUNNER = ROOT / "experiments/glm53_508_graph_qual/equal-508.py"
IMAGE = "example.invalid/replay-fixture/vllm@sha256:" + "a" * 64
CC = {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}
MANAGER = "ModelCudaGraphManager"
OWNER = MANAGER + "@a"
OTHER = MANAGER + "@b"


def _write(path, value):
    path.write_text(json.dumps(value))


def _descriptor(tokens, bound, num_reqs=1):
    return {"mode": "FULL", "tokens": tokens, "num_reqs": num_reqs, "max_seq_len": bound}


def _owner():
    captured = [_descriptor(n, b) for n in (1, 2) for b in (2048, 8448)]
    return {"capture_complete": True, "capture_error": None, "captured": captured,
            "replays": [{**d, "count": 1} for d in captured]}


def _dispatch(pid):
    return {
        "pid": pid, "captured": {OWNER: [1, 2]},
        "counts": {f"{MANAGER}|FULL|tokens={n}|reqs=1": 2 for n in (1, 2)},
        "tessera_classes": {
            "captured": {MANAGER: {"2048": 2, "8448": 2}},
            "replays": {f"{MANAGER}|2048": 2, f"{MANAGER}|8448": 2}},
        "owner_evidence": {OWNER: _owner()},
    }


def _pass(directory, name):
    summary = {"_admission": "paused"}
    for case, (specs, tokens) in ge.CASES.items():
        choices = [{
            "index": i, "prompt_token_ids": list(range(offset, offset + length)),
            "token_ids": [42] * tokens,
            "logprobs": {"token_logprobs": [-0.5] * tokens,
                         "top_logprobs": [{"fixture-token": -0.5} for _ in range(tokens)]},
        } for i, (length, offset) in enumerate(specs)]
        _write(directory / f"{name}.eq.{case}.json", {"choices": choices})
        summary[case] = {"batch": len(specs), "lengths": [n for n, _ in specs],
                         "max_tokens": tokens, "digests": [ge.digest(c) for c in choices],
                         "elapsed_s": 0.1, "steps": {}}
    _write(directory / f"{name}.eq.summary.json", summary)


class Measurement:
    def __init__(self, root, tp=2):
        self.tp = tp
        self.receipts = root / "receipts"
        self.receipts.mkdir()
        self.graph_dir = self.receipts / "g"
        self.out = root / "receipt.json"
        self.manifest = root / "manifest.json"
        model = root / "model"
        model.mkdir()
        _write(model / "config.json", {"index_topk": 2048})
        _write(self.manifest, {n: {"action_key": "fixture-" + n} for n in ("e1", "e2", "g")})
        for arm in ("e1", "e2", "g"):
            directory = self.receipts / arm
            directory.mkdir()
            eager = arm != "g"
            argv = ["vllm", "serve", str(model), "--tensor-parallel-size", str(tp),
                    "--max-model-len", "8448", "--max-num-seqs", "8"]
            argv += ["--enforce-eager"] if eager else ["--compilation-config", json.dumps(CC)]
            args = {
                "host": "cpu-fixture", "model": str(model), "image": IMAGE,
                "src_sha256": "b" * 64, "hooks_sha256": "c" * 64,
                "equal_script_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
                "eager": "1" if eager else "0", "kernel_json": "{}",
                "compilation_json": "null" if eager else json.dumps(CC),
                "serve_args": shlex.join(argv), "fabric_requested": "socket",
                "fabric_observed": ";".join(f"rank{r}:Using network Socket" for r in range(tp)),
            }
            (directory / f"engine-args-{arm}.txt").write_text(
                "".join(f"{key}={value}\n" for key, value in args.items()))
            (directory / f"{arm}.engine.log").write_text(
                f"Initializing a V1 LLM engine (v0.fixture) with config: enforce_eager={eager}\n")
            for rank in range(tp):
                name = self.basename(rank, 11 + rank, arm)
                _write(directory / name, _dispatch(11 + rank))
            for name in (arm, f"{arm}-r2"):
                _pass(directory, name)

    def basename(self, rank=0, pid=11, arm="g"):
        tag = f".rank{rank}" if self.tp > 1 else ""
        return f"{arm}{tag}.dispatch.{pid}.json"

    def path(self, rank=0, pid=11):
        return self.graph_dir / self.basename(rank, pid)

    def serve(self, receipt):
        return {"image": IMAGE, "model_config_sha256": receipt["model"]["config_sha256"],
                "tessera_src_sha256": "b" * 64, "compilation_config": CC,
                "speculative_tokens": 0, "max_model_len": 8448, "max_num_seqs": 8,
                "tensor_parallel_size": self.tp, "fabric": "socket" if self.tp > 1 else "none"}

    def produce(self):
        self.out.unlink(missing_ok=True)  # a previous successful receipt is not fresh evidence
        done = subprocess.run(
            [sys.executable, str(PRODUCER), str(self.receipts), str(self.manifest), str(self.out),
             "--eager", "e1,e2", "--graph", "g"],
            text=True, capture_output=True, timeout=30)
        assert "ModuleNotFoundError" not in done.stderr and "ImportError" not in done.stderr, done.stderr
        return done, json.loads(self.out.read_text()) if self.out.exists() else None

    def valid(self):
        done, receipt = self.produce()
        assert done.returncode == 0, done.stdout + done.stderr
        assert receipt["verdict"] == "equal"
        assert gr.verify(receipt, self.serve(receipt)) is None
        assert all(p["choices"] == ge.CHOICES_PER_PASS
                   for a in receipt["arms"] for p in a["passes"])
        self.expected = {p.name: json.loads(p.read_text())
                         for p in self.graph_dir.glob("g*.dispatch.*.json")}
        graph = receipt["arms"][0]["graph"]
        self.retained = graph.get("dispatches") == self.expected
        if "dispatches" in graph:
            assert self.retained, "producer changed or lost dispatch evidence"
        return receipt

    def require_retention(self):
        # Deliberately after the behavior assertion: main must fail for accepting
        # the attack, not for an unrelated missing-extension setup assertion.
        assert self.retained, "producer must itself retain graph.dispatches, not rely on the fixture"

    def refused(self):
        done, receipt = self.produce()
        assert done.returncode != 0, "producer borrowed/accepted invalid replay evidence:\n" + done.stdout
        if receipt is not None:
            assert receipt["verdict"] != "equal"
            _verify_refused(receipt, self.serve(receipt))
        self.require_retention()

    def direct(self):
        receipt = self.valid()
        # Proposed wire fixture, not a shadow verifier. Do not overwrite producer
        # evidence: a fixed CLI must retain it exactly and pass without this fallback.
        receipt["arms"][0]["graph"].setdefault("dispatches", copy.deepcopy(self.expected))
        assert gr.verify(receipt, self.serve(receipt)) is None
        return receipt

    def verify_refused(self, receipt):
        _verify_refused(receipt, self.serve(receipt))
        self.require_retention()

    def target(self, consumer):
        receipt = self.direct() if consumer == "verify" else self.valid()
        record = (receipt["arms"][0]["graph"]["dispatches"][self.basename()]
                  if consumer == "verify" else json.loads(self.path().read_text()))
        return receipt, record

    def reject(self, consumer, receipt, record):
        if consumer == "verify":
            self.verify_refused(receipt)
        else:
            _write(self.path(), record)
            self.refused()


def _verify_refused(receipt, serve):
    # Keep declared equal/attests/equal-arm flags intact: verify must rederive.
    try:
        why = gr.verify(receipt, serve)
    except Exception as exc:
        pytest.fail(f"verify raised {type(exc).__name__} instead of refusing replay evidence: {exc}")
    assert isinstance(why, str) and why.strip(), "shared verifier accepted invalid replay evidence"


@pytest.mark.parametrize("consumer", ["producer", "verify"])
def test_legacy_marginals_never_establish_owner_tuple_coverage(tmp_path, consumer):
    m = Measurement(tmp_path)
    receipt, record = m.target(consumer)
    if consumer == "verify":
        receipt["arms"][0]["graph"].pop("dispatches")
    else:
        record.pop("owner_evidence")
    # Complete legacy marginals remain; they cannot substitute for strict evidence.
    m.reject(consumer, receipt, record)


@pytest.mark.parametrize("damage", ["missing", "untagged", "out-of-range", "extra"])
def test_verify_requires_all_and_only_dispatch_ranks(tmp_path, damage):
    m = Measurement(tmp_path)
    receipt = m.direct()
    dispatches = receipt["arms"][0]["graph"]["dispatches"]
    record = dispatches.pop(m.basename(1, 12))
    if damage != "missing":
        name = "g.dispatch.12.json" if damage == "untagged" else "g.rank2.dispatch.12.json"
        dispatches[name] = record
        if damage == "extra":
            dispatches[m.basename(1, 12)] = copy.deepcopy(record)
    # Leave legacy rank marginals complete: the strict dispatch contract decides.
    m.verify_refused(receipt)


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("consumer", ["producer", "verify"])
def test_replays_cannot_be_borrowed_from_another_process(tmp_path, tp, consumer):
    m = Measurement(tmp_path, tp=tp)
    first, second = _dispatch(11), _dispatch(21)
    # A complete observed inventory need not be the Cartesian product of sizes
    # and bounds. Verify this valid non-Cartesian owner before the attack.
    for field in ("captured", "replays"):
        second["owner_evidence"][OWNER][field] = [
            d for d in second["owner_evidence"][OWNER][field]
            if (d["tokens"], d["max_seq_len"]) in ((1, 2048), (2, 8448))]
    # Same Class@id across processes is not the same owner. Initially both
    # processes independently cover every tuple they captured.
    _write(m.path(), first)
    _write(m.path(0, 21), second)
    receipt = m.direct() if consumer == "verify" else m.valid()
    if consumer == "verify":
        dispatches = receipt["arms"][0]["graph"]["dispatches"]
        first, second = dispatches[m.basename()], dispatches[m.basename(0, 21)]
    for replay in second["owner_evidence"][OWNER]["replays"]:
        replay["count"] = 2
    first["owner_evidence"][OWNER]["replays"] = []
    # All replays still exist globally; process 11 alone is now uncovered.
    if consumer == "producer":
        _write(m.path(0, 21), second)
    m.reject(consumer, receipt, first)


@pytest.mark.parametrize("consumer", ["producer", "verify"])
@pytest.mark.parametrize("damage", ["same-class-owner", "tuple-marginals", "wrong-mode", "num-reqs"])
def test_only_exact_owner_descriptors_discharge_capture(tmp_path, consumer, damage):
    m = Measurement(tmp_path)
    record = json.loads(m.path().read_text())
    record["captured"][OTHER] = [1, 2]
    record["owner_evidence"][OTHER] = _owner()
    _write(m.path(), record)
    receipt, record = m.target(consumer)
    evidence = record["owner_evidence"]
    if damage == "same-class-owner":
        evidence[OWNER]["replays"] = []
        for replay in evidence[OTHER]["replays"]:
            replay["count"] = 2
    elif damage == "tuple-marginals":
        evidence[OWNER]["replays"] = [
            {**_descriptor(1, 2048), "count": 2}, {**_descriptor(2, 8448), "count": 2},
            *[r for r in evidence[OWNER]["replays"] if r["max_seq_len"] is None]]
        # Both sizes AND both bounds replay, but (1,8448)/(2,2048) never do.
    else:
        field, value = ("mode", "PIECEWISE") if damage == "wrong-mode" else ("num_reqs", 2)
        for replay in evidence[OWNER]["replays"]:
            replay[field] = value
    m.reject(consumer, receipt, record)


@pytest.mark.parametrize("consumer", ["producer", "verify"])
@pytest.mark.parametrize("damage", ["failed", "incomplete"])
def test_capture_status_cannot_hide_behind_complete_replays(tmp_path, consumer, damage):
    m = Measurement(tmp_path)
    receipt, record = m.target(consumer)
    evidence = record["owner_evidence"][OWNER]
    if damage == "failed":
        evidence["capture_error"] = "capture raised after partial success"
    else:
        evidence["capture_complete"] = False
    m.reject(consumer, receipt, record)


BAD_COUNTERS = [True, 1.5, -1, float("nan"), float("inf"), -float("inf"), "1", None, {}, [], 0]
COUNTER_IDS = ["bool", "fraction", "negative", "nan", "inf", "minus-inf", "string", "null",
               "object", "list", "zero-uncovered"]


@pytest.mark.parametrize("value", BAD_COUNTERS, ids=COUNTER_IDS)
@pytest.mark.parametrize("consumer", ["producer", "verify"])
def test_owner_replay_counters_must_be_integral_and_cover_capture(tmp_path, consumer, value):
    m = Measurement(tmp_path)
    receipt, record = m.target(consumer)
    record["owner_evidence"][OWNER]["replays"][0]["count"] = value
    m.reject(consumer, receipt, record)


@pytest.mark.parametrize("consumer,damage", [
    (consumer, damage) for consumer in ("producer", "verify")
    for damage in ("empty-rank", "no-inventory", "unknown-owner")
] + [("verify", "malformed-rank")])
def test_required_rank_content_and_owner_mapping_are_not_vacuous(tmp_path, consumer, damage):
    m = Measurement(tmp_path)
    receipt, record = m.target(consumer)
    if damage == "empty-rank":
        record.clear()
    elif damage == "malformed-rank":
        receipt["arms"][0]["graph"]["dispatches"][m.basename()] = "not-a-dispatch-object"
    elif damage == "no-inventory":
        record["owner_evidence"][OWNER]["captured"] = []
    else:
        # Legacy capture owner cannot be silently dropped because its strict
        # mapping is absent. A different owner's complete history is no proof.
        record["owner_evidence"][OTHER] = record["owner_evidence"].pop(OWNER)
    m.reject(consumer, receipt, record)


@pytest.mark.parametrize("consumer", ["producer", "verify"])
@pytest.mark.parametrize("damage", ["tokens", "num-reqs", "bound", "unused-replay"])
def test_descriptor_and_unused_counters_are_validated_before_matching(tmp_path, consumer, damage):
    m = Measurement(tmp_path)
    receipt, record = m.target(consumer)
    evidence = record["owner_evidence"][OWNER]
    if damage == "unused-replay":
        evidence["replays"].append({**_descriptor(99, None), "count": float("nan")})
    else:
        field, value = {"tokens": ("tokens", float("nan")),
                        "num-reqs": ("num_reqs", 1.5),
                        "bound": ("max_seq_len", float("inf"))}[damage]
        evidence["captured"][0][field] = value
        evidence["replays"][0][field] = value
    m.reject(consumer, receipt, record)


@pytest.mark.parametrize("value", [7, []], ids=["scalar", "wrong-shaped-list"])
def test_verify_malformed_dispatches_returns_reason_not_exception(tmp_path, value):
    m = Measurement(tmp_path)
    receipt = m.direct()
    receipt["arms"][0]["graph"]["dispatches"] = value
    m.verify_refused(receipt)
