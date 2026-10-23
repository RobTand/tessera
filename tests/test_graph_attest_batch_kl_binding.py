"""CPU regressions for prepared KL configuration and native replay populations.

Reuse the protected normalized evidence fixture; these checks qualify neither a
scorer handoff nor a GPU measurement.
"""
import json

import pytest

from test_graph_attest_batch_kl import (
    _bind, _write, evidence, recipe, window, window_driver,
)


@pytest.mark.parametrize("enabled", [False, True])
def test_batch_kl_configuration_reaches_both_native_ranks(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    env = dict(TS=str(tmp_path / "source"), ARTIFACT=str(artifact),
               RECEIPTS=str(tmp_path / "arms"), FABRIC="socket",
               SOURCE_COMMIT="c" * 40, SOURCE_SHA256="a" * 64,
               PRODUCER_COMMIT="d" * 40, PRODUCER_SHA256="e" * 64)
    if enabled:
        env["BATCH_KL"] = json.dumps(dict(handoff="owner-pending", scoring=dict(expected_rows=4)))
    config = recipe.inputs(env, live=False)
    _write(tmp_path / "inputs.json", dict(config=config))
    rows = window_driver.rows(tmp_path, config, env)
    assert len(rows) == 2
    for row in rows:
        current = recipe.inputs(row["env"], live=False)
        recipe.check_control_record(config, current, where="Window4 rank",
                                    refusal=window.Refused("configuration lost during preparation"))
        if enabled:
            assert row["env"]["BATCH_KL"] == env["BATCH_KL"]
        else:
            assert "BATCH_KL" not in row["env"] and "batch_kl" not in current


# Deliberately not json.dumps spelling: inputs retain the parsed object, while
# the native manifest seal covers the exact action-environment string.
DECLARATION = '{\n "scoring" : {"expected_rows" : 4}, "handoff" : "owner-\\u0070ending"\n}'


class ReachedPreparedReview(RuntimeError):
    pass


@pytest.fixture
def prepared_submission(tmp_path, monkeypatch):
    def prepare(declaration, dev):
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
        monkeypatch.delenv("BATCH_KL", raising=False)
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        (artifact / "config.json").write_text("{}")
        env = dict(TS=str(tmp_path / "source"), ARTIFACT=str(artifact),
                   RECEIPTS=str(tmp_path / "arms"), FABRIC="socket",
                   SOURCE_COMMIT="c" * 40, SOURCE_SHA256="a" * 64,
                   PRODUCER_COMMIT="d" * 40, PRODUCER_SHA256="e" * 64)
        if declaration is not None:
            env["BATCH_KL"] = declaration
        inputs = recipe.inputs
        # Only external source/artifact authentication is inert. Exercise the
        # real parser, comparability gate, rows and manifest seal in both modes.
        def inert_inputs(env, *, live):
            return inputs({**env, "PRISMAQUANT_DEV_MODE": "1"}, live=False)
        monkeypatch.setattr(recipe, "inputs", inert_inputs)
        monkeypatch.setattr(recipe, "require_producer", lambda *a, **kw: None)
        config = recipe.inputs(env, live=False)
        _write(tmp_path / "inputs.json", dict(config=config))
        stored = window_driver.rows(tmp_path, config, env)
        _write(tmp_path / "manifest.json", stored)
        reviews = tmp_path / "reviews.json"
        read = window_driver.read_json
        def stop_at_review(path):
            if path == reviews:
                raise ReachedPreparedReview("configuration and manifest validated; no publication")
            return read(path)
        monkeypatch.setattr(window_driver, "read_json", stop_at_review)
        build_rows = window_driver.rows
        rebuilt = []
        def capture_rows(*args):
            result = build_rows(*args)
            rebuilt.extend(result)
            return result
        monkeypatch.setattr(window_driver, "rows", capture_rows)
        return tmp_path, reviews, stored, rebuilt
    return prepare


@pytest.mark.parametrize("dev", ["0", "1"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ambient", [None, "not JSON from a different submitting shell"])
def test_submit_restores_exact_prepared_batch_kl(prepared_submission, monkeypatch, dev, enabled, ambient):
    root, reviews, stored, rebuilt = prepared_submission(DECLARATION if enabled else None, dev)
    before = {name: (root / name).read_bytes() for name in ("inputs.json", "manifest.json")}
    if ambient is not None:
        monkeypatch.setenv("BATCH_KL", ambient)
    with pytest.raises(ReachedPreparedReview):
        window_driver.submit(root, reviews)
    assert rebuilt == stored
    for row in rebuilt:
        if enabled:
            assert row["env"]["BATCH_KL"] == DECLARATION
        else:
            assert "BATCH_KL" not in row["env"]
    assert before == {name: (root / name).read_bytes() for name in before}
    assert not (root / "submission-started.json").exists()


@pytest.mark.parametrize("dev", ["0", "1"])
@pytest.mark.parametrize("fault", ["missing-rank0", "missing-rank1", "missing-both", "non-string",
                                  "different-value", "different-spelling", "wrong-config"])
def test_submit_refuses_inconsistent_prepared_batch_kl(prepared_submission, dev, fault):
    root, reviews, stored, _ = prepared_submission(DECLARATION, dev)
    if fault.startswith("missing"):
        ranks = [0, 1] if fault == "missing-both" else [int(fault[-1])]
        for rank in ranks:
            del stored[rank]["env"]["BATCH_KL"]
    elif fault == "non-string":
        stored[1]["env"]["BATCH_KL"] = json.loads(DECLARATION)
    elif fault == "different-spelling":
        stored[1]["env"]["BATCH_KL"] = json.dumps(json.loads(DECLARATION))
    else:
        for rank in ([0, 1] if fault == "wrong-config" else [1]):
            stored[rank]["env"]["BATCH_KL"] = '{"handoff":"a different declaration"}'
    _write(root / "manifest.json", stored)
    with pytest.raises(window.Refused, match="BATCH_KL|scope/geometry/comparability"):
        window_driver.submit(root, reviews)
    assert not (root / "submission-started.json").exists()


@pytest.mark.parametrize("dev", ["0", "1"])
def test_submit_refuses_batch_kl_added_to_disabled_manifest(prepared_submission, dev):
    root, reviews, stored, _ = prepared_submission(None, dev)
    for row in stored:
        row["env"]["BATCH_KL"] = DECLARATION
    _write(root / "manifest.json", stored)
    with pytest.raises(window.Refused, match="BATCH_KL.*disabled window"):
        window_driver.submit(root, reviews)
    assert not (root / "submission-started.json").exists()


def test_submit_keeps_full_manifest_seal(prepared_submission):
    root, reviews, stored, _ = prepared_submission(DECLARATION, "0")
    stored[0]["argv"].append("unreviewed-argument")
    _write(root / "manifest.json", stored)
    with pytest.raises(window.Refused, match="prepared PB manifest changed"):
        window_driver.submit(root, reviews)
    assert not (root / "submission-started.json").exists()


@pytest.mark.parametrize("missing_rank", [0, 1])
def test_companion_requires_each_native_rank_replay_population(evidence, missing_rank):
    graph = evidence.graph
    del graph["arms"][0]["graph"]["managers"][f"rank{missing_rank}:manager"]
    # Keep the cached equal verdict, just as a stale or forged receipt might.
    _write(evidence.graph_path, graph)
    bound = _bind(evidence)
    assert bound["qualified"] is False, bound
    assert any("aGR" in error and f"rank{missing_rank}" in error
               for error in bound["errors"]), bound
