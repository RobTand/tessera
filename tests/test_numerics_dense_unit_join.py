"""Check each bound case's streamed code proof."""
import copy
import inspect

import pytest

from experiments.numerics_rows import dependency_code as code, phase_artifact as streams, row


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
