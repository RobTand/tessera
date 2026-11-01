"""Synthetic orchestration/binding contract for tessera#1365; no GPU qualification.

The fixture describes a normalized collector report, NOT an invented scorer wire
protocol. A production adapter must first obtain the existing scorer owner's
live TP2/MTP1 target-model handoff. Until then it must refuse, not run an isolated
scorer or the server-owning serve_and_dump_kl.sh. The new public binding seam is
window_driver.bind_batch_kl(root, graph_receipt_path): it returns and persists a
companion batch-kl-manifest.json, never rewrites the graph receipt. LocalArm's
collect_kl(arm, peer_meta) is called by probes before the live arm is stopped.

Research evidence (base 17b3fdb396410698c9a142615345f046d2f94a5d):
Command: PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest tests/test_graph_attest_batch_kl.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
Pre-fix, x86 only: 50 assertion failures in 11.63s, no setup/collection errors.
47 cases reach the explicit absent-companion assertion. Behavioral regressions:
"live arm never collected batch KL"; missing handoff DID NOT RAISE Refused;
collector failure was ignored (run_rank returned 0). Earlier harness iterations
were corrected (replay-count mapping and fake Rendezvous keyword arguments);
those harness failures are not claimed as product regression evidence.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"76962336007bca959d709c744808f8b35804cf127491998df8a7a635b167f35d","size":194744}},"operation":"c298e07d3c74d9e670be4c46e94a3492e0175a2cc36fa04138ddcdce15e9bfa0","receipt":"e73e005dc435619b5a93bc481081b5a4f670cf87fd405ce6036fc8eba8111163"}]

Existing-coverage command: PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest tests/test_graph_attest_tp2.py tests/test_graph_attest_managed_window.py tests/test_graph_receipt.py tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
x86: 291 passed, 9 skipped in 122.28s; skips require published PB SDK.
NO CUDA, zero device allocations. No GPU run or measured TP2 outcome claimed.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3b3c40dba257d03aaed2f34f0683142165530ef4fd6697a8e47f9fbc96bd80e4","size":2030}},"operation":"9b5a0a04cb9611fcb9cfb6dd7518332a3bd8326a22892ed37614fd874c295573","receipt":"854aafe09fbe8ddd8e5060c3fa97477462f65fb6ffd85ec5a5893fb7ce020b2f"}]
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tessera import graph_receipt

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/graph_attest_702"
FIXTURE = ROOT / "tests/fixtures/graph_batch_kl/window.json"
sys.path.insert(0, str(HERE))
import managed_window as window
import rank_window
import tp2_recipe as recipe
import window_driver

NAMES = ("aE1", "aGR", "aE2")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flags(name):
    return dict(enforce_eager=name != "aGR",
                compilation_config=None if name != "aGR" else recipe.GRAPH,
                speculative_tokens=1, tensor_parallel_size=2,
                max_num_seqs=4, max_model_len=8448, fabric="socket")


@pytest.fixture
def evidence(tmp_path):
    """Committed identity fixture; tiny dump bytes are not model probabilities."""
    f = json.loads(FIXTURE.read_text())
    root = tmp_path / "window"
    root.mkdir()
    model = tmp_path / "nominated-artifact"
    model.mkdir()
    # Identity is nominated/retained, not replaced with hash('{}') in a mock.
    scope = dict(model_config_sha256=f["config_sha256"], image=f["image"],
                 image_id=f["image_id"], tessera_src_sha256=f["src_sha256"],
                 artifact=str(model))
    config = dict(artifact=str(model), config_sha256=f["config_sha256"],
                  image=f["image"], src_sha256=f["src_sha256"], fabric="socket",
                  receipts=str(root / "arms"), batch_kl=dict(scoring=f["scoring"]))
    arms = [dict(arm=n, eager="1" if n != "aGR" else "0",
                 compilation="" if n != "aGR" else json.dumps(recipe.GRAPH),
                 spec=json.dumps(recipe.MTP)) for n in NAMES]
    _write(root / "inputs.json", dict(run_id=f["run_id"], config=config, arms=arms))
    input_sha256 = _sha(root / "inputs.json")
    members = [dict(m, run_id=f["run_id"], input_sha256=input_sha256,
                    simulation=False, returncode=0, completed_arms=list(NAMES)) for m in f["members"]]
    for m in members:
        _write(root / f"outcome-rank{m['rank']}.json", m)
    _write(root / "native-gang.json", dict(schema="prismabuild.pbgang.v1",
                                         members=[m["action_key"] for m in members]))
    _write(root / "receipt-manifest.json", {n: dict(action_key=members[0]["action_key"],
           rank_action_keys=[m["action_key"] for m in members]) for n in NAMES})
    records = []
    for n in NAMES:
        d = root / "arms" / n
        d.mkdir(parents=True)
        for m in members:
            _write(d / f"ownership-rank{m['rank']}.json", dict(m, container=dict(cid=m["container_id"])))
        dump = d / "target-dump.jsonl"
        dump.write_text("".join(json.dumps(dict(row=i, fixture_only=True)) + "\n" for i in range(4)))
        result = d / "kl-result.json"
        _write(result, dict(status="ok", complete=True, rows=4, fixture_only=True))
        report = dict(schema="tessera.batch_kl_arm.v1", arm=n, run_id=f["run_id"],
                      input_sha256=input_sha256, members=members, scope=scope,
                      requested=_flags(n), observed=_flags(n), scoring=f["scoring"],
                      target_provenance=dict(model_role="target", config_sha256=f["config_sha256"],
                                             speculative_tokens=1, ranks=[0, 1]),
                      complete=True, rows=4, status="ok",
                      raw_dump=dict(path=dump.name, sha256=_sha(dump)),
                      result=dict(path=result.name, sha256=_sha(result)))
        _write(d / "batch-kl.json", report)
        records.append(dict(name=n, pb_action=members[0]["action_key"],
                            execution="eager" if n != "aGR" else "graph",
                            compilation_config=_flags(n)["compilation_config"],
                            speculative_tokens=1, tensor_parallel_size=2, max_num_seqs=4,
                            max_model_len=8448, image_id=f["image_id"],
                            graph=dict(managers={f"rank{r}:manager": dict(captured_sizes=[2], replayed_sizes={"2": 1})
                                                 for r in (0, 1)}, classes={}),
                            passes=[dict(name=n + suffix, members=48, choices=48) for suffix in ("", "-r2")]))
    graph = dict(schema=graph_receipt.SCHEMA, issue="tessera#702",
                 runtime=dict(image=f["image"], fabric="socket"),
                 tessera=dict(src_sha256=f["src_sha256"]),
                 model=dict(path=str(model), config_sha256=f["config_sha256"]),
                 equality_set=dict(choices_per_pass=48), eager_pool=[records[0], records[2]],
                 arms=[records[1]], not_measured=["served KL against BF16 under graphs"])
    graph_receipt.finish(graph)
    assert graph["verdict"] == "equal", "test fixture must satisfy the existing graph gate"
    graph_path = root / "graph-receipt.json"
    _write(graph_path, graph)
    return SimpleNamespace(root=root, graph_path=graph_path, graph=graph, fixture=f)


def _bind(e):
    bind = getattr(window_driver, "bind_batch_kl", None)
    assert callable(bind), "missing three-arm batch-KL companion binding at the native window driver"
    before = e.graph_path.read_bytes()
    result = bind(e.root, e.graph_path)
    assert e.graph_path.read_bytes() == before, "KL must not relabel or rewrite the graph receipt"
    stored = json.loads((e.root / "batch-kl-manifest.json").read_text())
    assert result == stored
    return result


def _report(e, name="aGR"):
    path = e.root / "arms" / name / "batch-kl.json"
    return path, json.loads(path.read_text())


def test_companion_binds_three_arms_hashes_and_both_native_members(evidence):
    e = evidence
    bound = _bind(e)
    assert bound["qualified"] is True, bound
    assert bound["graph_receipt"]["sha256"] == _sha(e.graph_path)
    assert set(bound["arms"]) == set(NAMES)
    for n in NAMES:
        _, report = _report(e, n)
        arm = bound["arms"][n]
        for field in ("members", "scope", "requested", "observed", "scoring", "target_provenance"):
            assert arm[field] == report[field], field
        for field in ("raw_dump", "result"):
            assert arm[field]["sha256"] == report[field]["sha256"]
    assert "served KL against BF16 under graphs" in json.loads(e.graph_path.read_text())["not_measured"]


@pytest.mark.parametrize("missing", NAMES)
def test_each_missing_arm_makes_window_unqualified(evidence, missing):
    (evidence.root / "arms" / missing / "batch-kl.json").unlink()
    bound = _bind(evidence)
    assert bound["qualified"] is False and bound["errors"], bound
    assert missing in json.dumps(bound["errors"])


@pytest.mark.parametrize("mutation", [
    "config", "artifact", "image", "actual_image", "source", "corpus", "teacher", "tokenizer", "regime",
    "observed_eager", "requested_graph", "observed_graph", "mtp_disabled", "max_seqs", "tp",
    "stale_run", "stale_input", "rank1_action", "rank1_nonce", "rank1_scope", "rank1_container",
    "draft_logits", "target_config", "missing_target_rank", "partial", "row_count", "refusal",
    "raw_hash", "result_hash", "missing_dump", "missing_result", "truncated_dump", "failed_result",
])
def test_inconsistent_partial_stale_or_refused_evidence_never_qualifies(evidence, mutation):
    path, report = _report(evidence)
    if mutation in {"config", "artifact", "image", "actual_image", "source"}:
        field = dict(config="model_config_sha256", artifact="artifact", image="image",
                     actual_image="image_id", source="tessera_src_sha256")[mutation]
        report["scope"][field] = "wrong-identity"
    elif mutation in {"corpus", "teacher", "tokenizer", "regime"}:
        field = mutation if mutation == "regime" else mutation + "_sha256"
        report["scoring"][field] = "other-scoring-population"
    elif mutation == "observed_eager": report["observed"]["enforce_eager"] = True
    elif mutation == "requested_graph": report["requested"]["compilation_config"] = {"mode": "VLLM_COMPILE"}
    elif mutation == "observed_graph": report["observed"]["compilation_config"] = None
    elif mutation == "mtp_disabled": report["observed"]["speculative_tokens"] = 0
    elif mutation == "max_seqs": report["observed"]["max_num_seqs"] = 1
    elif mutation == "tp": report["observed"]["tensor_parallel_size"] = 1
    elif mutation == "stale_run": report["run_id"] = "old-run"
    elif mutation == "stale_input": report["input_sha256"] = "0" * 64
    elif mutation.startswith("rank1_"):
        field = dict(rank1_action="action_key", rank1_nonce="nonce", rank1_scope="scope_id",
                     rank1_container="container_id")[mutation]
        report["members"][1][field] = "old-or-foreign-member"
    elif mutation == "draft_logits": report["target_provenance"]["model_role"] = "drafter"
    elif mutation == "target_config": report["target_provenance"]["config_sha256"] = "0" * 64
    elif mutation == "missing_target_rank": report["target_provenance"]["ranks"] = [0]
    elif mutation == "partial": report["complete"] = False
    elif mutation == "row_count": report["rows"] = 3
    elif mutation == "refusal": report["status"] = "refused: target-model MTP collection unsupported"
    elif mutation in {"raw_hash", "result_hash"}:
        report["raw_dump" if mutation == "raw_hash" else "result"]["sha256"] = "0" * 64
    elif mutation in {"missing_dump", "missing_result"}:
        (path.parent / report["raw_dump" if mutation == "missing_dump" else "result"]["path"]).unlink()
    elif mutation == "truncated_dump":
        dump = path.parent / report["raw_dump"]["path"]
        dump.write_text('{"row":0,"fixture_only":true}\n')
        report["raw_dump"]["sha256"] = _sha(dump)  # internally rehashed partial output is still partial
    elif mutation == "failed_result":
        result = path.parent / report["result"]["path"]
        _write(result, dict(status="refused", complete=False, rows=0))
        report["result"]["sha256"] = _sha(result)
    _write(path, report)
    bound = _bind(evidence)
    assert bound["qualified"] is False and bound["errors"], (mutation, bound)


@pytest.mark.parametrize("name", ["aE1", "aE2"])
def test_both_eager_controls_are_bound_not_just_the_graph_arm(evidence, name):
    path, report = _report(evidence, name)
    report["observed"]["enforce_eager"] = False
    _write(path, report)
    assert _bind(evidence)["qualified"] is False


def test_claimed_equal_cannot_hide_a_failed_graph_population(evidence):
    broken = deepcopy(evidence.graph)
    broken["arms"][0]["passes"][1]["members"] = 47
    broken["verdict"] = "equal"  # don't trust a caller's label
    _write(evidence.graph_path, broken)
    assert _bind(evidence)["qualified"] is False


@pytest.mark.parametrize("field", ["model_config_sha256", "image_id", "corpus_sha256", "teacher_sha256"])
def test_consistent_but_wrong_population_is_not_a_valid_window(evidence, field):
    for name in NAMES:
        path, report = _report(evidence, name)
        report["scoring" if field in {"corpus_sha256", "teacher_sha256"} else "scope"][field] = "wrong-for-all-arms"
        _write(path, report)
    assert _bind(evidence)["qualified"] is False


@pytest.mark.parametrize("mutation", ["old_action", "rank1_no_replay", "eager_missing_pass"])
def test_graph_receipt_must_belong_to_the_same_native_window(evidence, mutation):
    graph = deepcopy(evidence.graph)
    if mutation == "old_action":
        graph["arms"][0]["pb_action"] = "0" * 64
    elif mutation == "rank1_no_replay":
        graph["arms"][0]["graph"]["managers"]["rank1:manager"]["replayed_sizes"] = {}
    else:
        graph["eager_pool"][1]["passes"].pop()
    _write(evidence.graph_path, graph)
    assert _bind(evidence)["qualified"] is False


def _local(tmp_path, monkeypatch):
    """Real LocalArm.probes with all external commands inert; no second serve."""
    local = rank_window.LocalArm.__new__(rank_window.LocalArm)
    local.config = dict(ts=str(ROOT), fabric="socket", image="fixture-image", artifact="fixture-artifact",
                        equal_script_sha256="1" * 64, producer_commit="fixture", producer_sha256="2" * 64,
                        source_commit="fixture", src_sha256="3" * 64, hooks_sha256="4" * 64,
                        batch_kl=dict(handoff=None))
    local.identity = dict(action_key="1" * 64, nonce="2" * 32, scope_id="fixture.slice", rank=0)
    local.active = dict(out=str(tmp_path), cid="5" * 64, arm="aGR")
    local.image_id = "sha256:" + "6" * 64
    local.fabric_banner = "Using network Socket"
    local.envelope = SimpleNamespace(remaining=lambda **kw: 30)
    local.events = []
    local.tick = lambda: None
    def command(argv, **kwargs):
        local.events.append(("command", argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="")
    local.command = command
    # Skip only the dispatch-flush delay; don't alter the global time module.
    clock = iter(range(0, 1000, 5))
    monkeypatch.setattr(rank_window, "time", SimpleNamespace(
        time=lambda: 100.0, monotonic=lambda: next(clock), sleep=lambda _: None))
    arm = dict(arm="aGR", eager="0", compilation=json.dumps(recipe.GRAPH), spec=json.dumps(recipe.MTP))
    peer = dict(banner="Using network Socket", image_id=local.image_id)
    return local, arm, peer


def test_collector_runs_at_live_probe_seam_and_propagates_evidence(tmp_path, monkeypatch):
    local, arm, peer = _local(tmp_path, monkeypatch)
    def collect(self, measured_arm, measured_peer):
        assert self.active["cid"] == "5" * 64
        assert measured_arm == arm and measured_peer == peer
        self.events.append(("collect",))
        return dict(report="batch-kl.json", sha256="7" * 64)
    monkeypatch.setattr(rank_window.LocalArm, "collect_kl", collect, raising=False)
    result = local.probes(arm, peer)
    assert sum(event[0] == "collect" for event in local.events) == 1, "live arm never collected batch KL"
    assert result["batch_kl"] == dict(report="batch-kl.json", sha256="7" * 64)
    for event in local.events:
        if event[0] == "command":
            assert "tick" in event[2], "collector/probes must share cancellation/deadline ownership"
            assert "serve_and_dump_kl.sh" not in " ".join(event[1])


def test_unsupported_mtp_handoff_refuses_instead_of_qualifying(tmp_path, monkeypatch):
    local, arm, peer = _local(tmp_path, monkeypatch)
    with pytest.raises(window.Refused, match=r"(?i)(target|MTP|handoff|2566|1318)"):
        local.probes(arm, peer)
    assert "rc=0" not in (tmp_path / "engine-args-aGR.txt").read_text()


def test_collector_failure_uses_existing_coupled_abort_and_cleanup(tmp_path, monkeypatch):
    local, arm, peer = _local(tmp_path, monkeypatch)
    local.config.update(source_commit="fixture", producer_commit="fixture")
    local.rdv = tmp_path
    local.release_inputs = lambda: None
    local.preflight = lambda: dict(image="fixture", src_sha256="fixture", config_sha256="fixture")
    local.start = lambda a: dict(cid=local.active["cid"])
    local.ready = lambda a: dict(fabric="socket", **peer)
    cleaned = []
    local.cleanup = lambda a: (cleaned.append(a["arm"]) or
                               dict(containers_empty=True, gpu_descendants_empty=True))
    def collect(self, a, p):
        raise window.Refused("collector refused: target-model MTP provenance unavailable")
    monkeypatch.setattr(rank_window.LocalArm, "collect_kl", collect, raising=False)
    events = []
    class Meeting:
        def __init__(self, *args, **kwargs):
            self.path = tmp_path / "rank0.json"
            self.peer = dict(local.identity, rank=1)
        def bind_peer(self, **kw): pass
        def check(self): pass
        def publish(self, stage, **kw): events.append(stage)
        def wait(self, stage, **kw):
            return dict(metadata=local.preflight(), fabric="socket", **peer)
        def checked(self, *args): return {}
    monkeypatch.setattr(rank_window, "Rendezvous", Meeting)
    _write(tmp_path / "aGR-cleaned-rank1.json", {})
    _write(tmp_path / "failed-cleaned-rank1.json", {})
    local.identity.update(run_id="fixture", window_end_unix=200.0)
    local.envelope = SimpleNamespace(remaining=lambda **kw: 30, end_unix=200.0,
                                     tighten=lambda _: None)
    rc = rank_window.run_rank(local.config, local.identity, tmp_path / "queue", tmp_path,
                              [arm], local, local.envelope, poll_seconds=.001)
    assert rc != 0, "collector failure was ignored and the arm was declared successful"
    outcome = json.loads((tmp_path / "outcome-rank0.json").read_text())
    assert "collector refused" in outcome["error"]
    assert cleaned == ["aGR"] and "failed" in events and "failed-cleaned" in events
    assert "aGR-probes" not in events and outcome["completed_arms"] == []
    assert outcome["local_cleanup"][0]["containers_empty"] is True
