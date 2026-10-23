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
