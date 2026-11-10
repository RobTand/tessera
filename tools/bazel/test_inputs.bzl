"""Inputs a test reads that no static evidence names, each with its reason.

tools/bazel/build_files.py derives every other input from the source. A test
listed here reads files through a root or a table the path analysis can't
follow. The remote executor makes such a read fail rather than pass, which is
how an entry is found. Keep each entry to the files named, and the reason true.
Entries are keyed by the test file's target; its `.bytes` lane shares them.
`packages` names pip packages from tools/bazel/requirements.txt that only the
torch-lane target installs.
"""

_CONSTRUCTION_T8 = "//:docs/measurements/construction/2026-10-07/glm53-flash-4layer-selected-t8.json"
_CONSTRUCTION_REASON = "experiments/t8r_speed/bench_dense_module.py reads this construction record through os.path.join over a root computed at run time."

_PRISMAQUANT = "it reads PrismaQuant, pinned in tools/ci/gate_prismaquant_requirements.txt; its own dependencies are heavy, so only the tests that read it carry it."

TEST_INPUTS = {
    "//tests:test_bazel_gate": struct(
        data = ["//tools/bazel:BUILD.bazel"],
        reason = "the pool contract reads the handwritten platform BUILD file; BUILD files are excluded from inferred source inputs.",
    ),
    "//tests:test_seeded_breaks": struct(
        data = ["//tests:BUILD.bazel"],
        reason = "the exact-label contract reads the generated test target declarations; BUILD files are excluded from inferred source inputs.",
    ),
    "//tests:test_standalone_boundary": struct(
        data = ["//src/tessera:files"],
        reason = "scan() reads every module under src/tessera through the root its caller passes (default the repository root).",
    ),
    "//tests:test_native_moe_glm_owner": struct(data = [], packages = ["prismaquant"], reason = _PRISMAQUANT),
    "//tests:test_plan_from_layer_config": struct(data = [], packages = ["prismaquant"], reason = _PRISMAQUANT),
    "//tests:test_ig790_verify": struct(data = [], packages = ["prismaquant"], reason = _PRISMAQUANT),
    "//tests:test_uniform_control": struct(data = [], packages = ["prismaquant"], reason = _PRISMAQUANT),
    "//tests:test_bench_kda_geometry": struct(data = [_CONSTRUCTION_T8], reason = _CONSTRUCTION_REASON),
    "//tests:test_bench_kda_timing_scope": struct(data = [_CONSTRUCTION_T8], reason = _CONSTRUCTION_REASON),
    "//tests:test_export_attention_owners": struct(data = [_CONSTRUCTION_T8], reason = _CONSTRUCTION_REASON),
    "//tests:test_serving_construction": struct(data = [_CONSTRUCTION_T8], reason = _CONSTRUCTION_REASON),
    "//tests:test_collection_probe": struct(
        data = ["//tests:test_hardware_byte_grid_lib"],
        reason = "it runs pytest on test_hardware_byte_grid.py by bare file name in a child process.",
    ),
    "//tests:test_contract_is_portable": struct(
        data = ["//src/tessera:runtime_contract_paths"],
        reason = "it checks that every repository path runtime_contract.json names, in prose too, exists.",
    ),
    "//tests:test_serving_contract": struct(
        data = ["//src/tessera:runtime_contract_paths"],
        reason = "it checks that the receipts runtime_contract.json names exist.",
    ),
    "//tests:test_kda_bank_verifier": struct(
        data = ["//experiments:mhc/mhc_probe.py", "//experiments:kda/conv_progress.py"],
        reason = "experiments/kda/conv_gate.py digests SOURCE_OWNERS, paths relative to experiments/.",
    ),
    "//tests:test_kda_probe_gate": struct(
        data = ["//experiments:mhc/mhc_probe.py", "//experiments:kda/conv_progress.py"],
        reason = "experiments/kda/conv_gate.py digests SOURCE_OWNERS, paths relative to experiments/.",
    ),
    "//tests:test_native_operator_receipt": struct(
        data = ["//experiments:csrc/native_operator_resources.cpp"],
        reason = "it hashes the collector source through a root the fixture passes.",
    ),
    "//tests:test_native_operator_resources": struct(
        data = ["//tests:fixtures/native_operator_resource_qualification.json"],
        reason = "a helper receives the fixture path as an argument.",
    ),
    "//tests:test_rung_allowability_structure_spec": struct(
        data = [
            "//experiments:t8r_speed/structure_spec_glm53_tp2.json",
            "//tests:files/fixtures/d41_structure_spec_harvest",
            "//:docs/schema/allowable-rung-table.v2.schema.json",
            "//:docs/schema/index.v2.schema.json",
        ],
        reason = "it joins fixture and schema names onto directories computed with os.path.dirname chains, and the table tool globs the harvest root it is given.",
    ),
    "//tests:test_tp2_run_plan_bootstrap_env": struct(
        data = ["//experiments:resource_bootstrap/sitecustomize.py"],
        reason = "a child interpreter imports sitecustomize from the bootstrap directory the launcher puts on PYTHONPATH.",
    ),
    "//tests:test_vllm_stub_cohort_isolation": struct(
        data = ["//tests:test_serving_dispatch_lib", "//tests:test_serving_moe_dispatch_lib"],
        reason = "a child interpreter imports test_serving_dispatch and test_serving_moe_dispatch by names built in its -c snippet.",
    ),
}
