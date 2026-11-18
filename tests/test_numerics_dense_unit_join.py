"""Check disjoint receipt units and their streamed code proofs."""
import copy
import inspect

import pytest

from experiments.numerics_rows import dependency_code as code, phase_artifact as streams, row
from experiments.numerics_rows.common import Invalid
from experiments.numerics_rows.join import join
from tests.test_numerics_rows_receipts import joined_inputs


def split_receipts(receipts):
    result = []
    for digest, receipt in receipts:
        cuts = sorted({(unit["tp"]["size"], unit["tp"]["rank"]) for unit in receipt["units"]})
        for size, rank in cuts:
            part = copy.deepcopy(receipt)
            part["units"] = [unit for unit in part["units"] if (unit["tp"]["size"], unit["tp"]["rank"]) == (size, rank)]
            result.append((f"{digest}-{size}-{rank}", part))
    return result


def accept_split(population, digest, manifests, receipts):
    try:
        return join(population, digest, manifests, receipts)
    except Invalid as error:
        assert str(error) != "duplicate row module selection", "feature absent: disjoint full-unit receipts"
        raise


@pytest.mark.parametrize("missing", (False, True))
def test_join_accepts_disjoint_module_cut_receipts_and_requires_the_union(monkeypatch, missing):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    split = split_receipts(receipts)
    assert accept_split(population, digest, manifests, split)["verdict"] == "PASS"
    if missing:
        split[0][1]["units"].pop()
        result = accept_split(population, digest, manifests, split)
        assert result["verdict"] == "UNATTESTED"
        assert any(record["missing_evidence"] for unit in result["units"] for record in unit["modules"])


@pytest.mark.parametrize("mutation", ("duplicate", "foreign", "known-fail"))
def test_disjoint_join_keeps_scope_refusals_and_worse_outcomes(monkeypatch, mutation):
    population, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(population, digest, manifests, receipts)["verdict"] == "PASS"
    split = split_receipts(receipts)
    assert accept_split(population, digest, manifests, split)["verdict"] == "PASS"
    if mutation == "known-fail":
        split[0][1]["units"][0]["outcome"] = "FAIL"
        split.pop()
        assert accept_split(population, digest, manifests, split)["verdict"] == "FAIL"
    else:
        if mutation == "duplicate":
            split[1][1]["units"].append(copy.deepcopy(split[0][1]["units"][0]))
        else:
            unit = split[0][1]["units"][0]
            unit["m_config"]["M"] = 9999
            unit["unit_id"] = row.unit_identity(unit)
        with pytest.raises(Invalid, match="duplicate module unit|outside the required signed scope"):
            join(population, digest, manifests, split)


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "unit", "origin", "observer", "digest", "stream-bytes"))
def test_join_code_reader_checks_each_bound_case(tmp_path, mutation):
    assert "common" in inspect.signature(code.check_unit_code).parameters, "feature absent: streamed unit code proofs"
    from tests.test_numerics_rows_phases import rewrite_stream
    unit = row.new_unit({"cell": "cell", "module": "A", "m_config": {"M": 1}, "tp": {"size": 1, "rank": 0}})
    row.finish_unit(unit)
    common = {"runtime": {"image_digest": "image"}}
    proof = {"unit_id": unit["unit_id"], "dependency_code": {"observation_complete": True},
             "triton_specializations": [], "unresolved_kernel_origins": [],
             "arm": {"profiler_kernels": [{"name": "case-kernel"}], "preparation_kernels": [{"name": "prepare-kernel"}],
                     "loaded_modules": [{"path": "/mapped/native.so", "library_so_sha256": "a" * 64}]},
             "kernel_origins": [{"profiler_name": name, "origin": "tessera", "path": "/mapped/native.so", "sha256": "a" * 64}
                                for name in ("case-kernel", "prepare-kernel")]}
    code.check_unit_code(proof, "image")
    artifact = streams.Artifact(tmp_path / "compare", "compare", common)
    scope = streams.unit_scope(unit, "wire")
    artifact.document["scopes"] = [scope]
    case = artifact.add(scope, {"M": 1}, "dense", {}, category="sweep", completion=True, code=proof)
    unit["code_stream"] = {"path": str(artifact.root / "phase.json"), "sha256": artifact.finish()}
    code.check_unit_code(unit, "image", common)
    values = [case]
    if mutation == "missing": values.clear()
    if mutation == "duplicate": values.append(copy.deepcopy(case))
    if mutation == "unit": proof["unit_id"] = ["another-unit"]
    if mutation == "origin": proof["kernel_origins"].pop()
    if mutation == "observer": proof["dependency_code"]["observation_complete"] = False
    if mutation == "digest": unit["code_stream"]["sha256"] = "b" * 64
    elif mutation == "stream-bytes":
        with (artifact.root / "cases.jsonl").open("a") as target:
            target.write("{}\n")
    else:
        rewrite_stream(artifact, "cases", values)
        from experiments.numerics_rows.common import sha256
        unit["code_stream"]["sha256"] = sha256(artifact.root / "phase.json")
    with pytest.raises(ValueError):
        code.check_unit_code(unit, "image", common)
