"""Independent code-owner rosters and a model of the pinned Triton launch API."""
from collections import namedtuple
from contextlib import nullcontext
from pathlib import Path
import json
import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.numerics_rows import common, h1, row
from experiments.numerics_rows.common import BoundaryRecorder, Invalid


@pytest.fixture
def code(monkeypatch):
    # Pure fixture support for the old recorder/row/join, not an old-policy patch.
    if not hasattr(common, "Unattested"):
        monkeypatch.setattr(common, "Unattested", type("Unattested", (ValueError,), {}), raising=False)
    return importlib.import_module("experiments.numerics_rows.dependency_code")


def kernel_type():
    target = namedtuple("Target", "backend arch warp_size")("cuda", 121, 32)
    metadata = namedtuple("Metadata", "target num_warps shared name")
    class Kernel:
        def __init__(self):
            self.kernel = b"actual cubin bytes"
            self.asm = {"cubin": self.kernel}
            self.name = "triton_entry"
            self.metadata = metadata(target, 4, 0, self.name)
            self.packed_metadata = (4, 0)
            self.src = SimpleNamespace(constants={(1,): 64, (2,): False})
            self.module = self.function = self._run = None
            self.calls = []
        def _init_handles(self):
            self.module, self.function = 10, 20
            self._run = lambda *args, **kwargs: self.calls.append(args)
        @property
        def run(self):
            if self._run is None:
                self._init_handles()
            return self._run
    return Kernel


def launch(kernel, grid=(2, 1, 1)):
    function = kernel.run
    return function(*grid, 30, kernel.function, kernel.packed_metadata, None, None, None)


def test_triton_observer_binds_loaded_bytes_case_target_constants_and_grid(code, tmp_path):
    kind = kernel_type()
    original = kind.run
    audit = code.CodeAudit("image", tmp_path).install(kind)
    try:
        k = kind()
        with audit.context("sweep:0"):
            launch(k)
        record, = audit.records
        identity = record["identity"]
        assert record["case"] == "sweep:0" and identity["entry"] == "triton_entry"
        assert identity["target"] == {"backend": "cuda", "arch": 121, "warp_size": 32}
        assert identity["launch"]["grid"] == [2, 1, 1]
        assert sorted(identity["constants"]) == [[[1], 64], [[2], False]]
        assert (tmp_path / (identity["cubin_sha256"] + ".cubin")).read_bytes() == k.kernel
        assert len(k.calls) == 1
    finally:
        audit.close()
    assert kind.run is original


@pytest.mark.parametrize("field", ("cubin", "entry", "target", "constants", "parameters"))
def test_serve_refuses_changed_compiled_specialization_before_launch(code, tmp_path, field):
    kind = kernel_type()
    audit = code.CodeAudit("image", tmp_path).install(kind)
    try:
        k = kind()
        with audit.context("sweep:0"):
            launch(k)
        k.calls.clear()
        with audit.context("sweep:0"):
            launch(k)
        assert len(k.calls) == 1
        k.calls.clear()
        grid = (2, 1, 1)
        if field == "cubin": k.kernel = k.asm["cubin"] = b"changed cubin bytes"
        elif field == "entry": k.name = "another_entry"
        elif field == "target": k.metadata = k.metadata._replace(target=k.metadata.target._replace(arch=120))
        elif field == "constants": k.src.constants[(1,)] = 128
        elif field == "parameters": k.packed_metadata = (8, 0)
        with audit.context("sweep:0"):
            with pytest.raises(Invalid, match="Triton"):
                launch(k, grid)
        assert k.calls == []
    finally:
        audit.close()


def test_missing_load_observation_refuses_before_triton_execution(code, tmp_path):
    kind = kernel_type()
    k = kind()
    launch(k)
    k.calls.clear()
    audit = code.CodeAudit("image", tmp_path).install(kind)
    try:
        with pytest.raises(code.Unattested, match="before the code observer"):
            launch(k)
        assert k.calls == []
    finally:
        audit.close()


def test_setup_byte_reference_detects_planted_faults():
    assert hasattr(h1, "setup_reference"), "H1_SETUP_BYTE_CHECKS_MISSING"
    packed = np.array([0xAB, 0x10, 0x2F, 0xC3], dtype=np.uint8)
    unpacked = np.array([[10, 2], [11, 15], [1, 12], [0, 3]], dtype=np.uint8)

    plane = np.zeros(1024, dtype=np.uint8)
    repacked = np.zeros_like(plane)
    plane[:4], plane[256:260], plane[512:516], plane[768:772] = (
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16])
    repacked[:4], repacked[256:260], repacked[512:516], repacked[768:772] = (
        [4, 3, 2, 1], [12, 11, 10, 9], [8, 7, 6, 5], [16, 15, 14, 13])
    cases = (
        (packed, 4, {"repack": False}, unpacked, 3, 7),
        (plane, 1024, {"repack": True}, repacked, 513, 258),
        (plane, 1024, {"repack": True, "row0": 512, "rows_local": 512}, repacked[512:], 769, 258),
    )
    for data, rows, args, expected, byte_index, output_index in cases:
        np.testing.assert_array_equal(h1.setup_reference(data, rows, 2, **args), expected)
        changed = data.copy()
        changed[byte_index] ^= 1
        expected_changed = expected.copy()
        expected_changed.flat[output_index] ^= 1
        reference_changed = h1.setup_reference(changed, rows, 2, **args)
        np.testing.assert_array_equal(reference_changed, expected_changed)
        check = h1.check_setup_bytes(expected, expected, expected_changed, reference_changed, byte_index)
        delta = {"count": 1, "indices": [output_index]}
        assert check == {"bytes": expected.size, "mismatches": 0, "result": "PASS",
                         "fault": {"byte_index": byte_index, "input_bit": 0, "predicted": delta,
                                   "observed": delta, "detected": True}}
        wrong_index, wrong_value, extra = (output.copy() for output in (expected, expected, expected_changed))
        wrong_index.flat[0] ^= 1
        wrong_value.flat[output_index] ^= 2
        extra.flat[0] ^= 1
        for output in (expected, wrong_index, wrong_value, extra):
            check = h1.check_setup_bytes(expected, expected, output, reference_changed, byte_index)
            assert check["mismatches"] == 0 and check["result"] == "FAIL" and not check["fault"]["detected"]


def test_dense_keys_have_h1_p2_records(code, tmp_path):
    # Use the November 17 H1 P2 key and the five keys from the compile receipt.
    target = {"backend": "cuda", "arch": 121, "warp_size": 32}
    legacy = {
        "unit_id": None, "case": "P2:ties",
        "sha256": "aa0f2952b76b1efa8da1554492f8140237a33b40918857bc922df8f235c33065",
        "identity": {
            "entry": "_window_gemm_kernel", "target": target,
            "cubin_sha256": "042c944ddb9024cdbe565cdfae923a6e7b23f56068810598506010b97ae71ed5",
            "constants": [[[19], 64], [[20], 64], [[22], True], [[18], 64], [[16], 14], [[17], 512], [[21], False], [[10], 1]],
            "launch": {"grid": [4, 1, 1], "packed_metadata": [8, 1, 4096], "metadata": [
                ["tensordesc_meta", []], ["arch", "sm121"], ["target", [["backend", "cuda"], ["warp_size", 32], ["arch", 121]]],
                ["debug", False], ["profile_scratch_align", 1], ["ptx_version", None], ["enable_reflect_ftz", True],
                ["instrumentation_mode", ""], ["extern_libs", [["libdevice", "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/lib/libdevice.10.bc"]]],
                ["maxnreg", None], ["global_scratch_align", 1], ["allowed_dot_input_precisions", ["tf32", "tf32x3", "ieee", "bf16x3", "bf16x6"]],
                ["hash", "612453067f84b77ad4708093ea46d40b73b174d6841d476ddeab866b19e41c31"],
                ["num_warps", 8], ["triton_version", "3.7.1"], ["profile_scratch_size", 0], ["shared", 4096],
                ["supported_fp8_dtypes", ["fp8e4b15", "fp8e4nv", "fp8e5"]], ["max_num_imprecise_acc_default", 0],
                ["enable_fp_fusion", True], ["warp_size", 32], ["tmem_size", 0], ["launch_pdl", False],
                ["launch_cooperative_grid", False], ["ir_override", None], ["global_scratch_size", 0],
                ["deprecated_fp8_dot_operand_dtypes", ["fp8e4b15"]], ["backend_name", "cuda"], ["sanitize_overflow", True],
                ["default_dot_input_precision", "tf32"], ["num_stages", 3], ["num_ctas", 1], ["ptx_options", None],
                ["name", "_window_gemm_kernel"],
            ]},
        },
    }
    keys = [
        ("_window_repack_kernel", "4096x1024", "d84293f5b6af75ac381516a75993f6a548096fec489e6d18d03f05e0494190b0",
         [[[14], 256]]),
        ("_unpack_body_kernel", "4096x12288", "667889ef2d96e7e0ad8206a00d2a3c815806366d4a4c46cea0957e59c4366540",
         [[[5], 1], [[6], 256], [[3], 4096], [[4], 12288]]),
        ("_unpack_body_kernel", "12288x4096", "34e10f5b900c25d1ab59b27de3f1e8ce3c0aa8b3113750d25c410a612f05a9df",
         [[[4], 4096], [[5], 1], [[3], 12288], [[6], 256]]),
        ("_unpack_body_kernel", "2048x4096", "9ee921a3cd388db00421af43b65dd604ea7cba131fccbf9a4bcc10ae37316949",
         [[[4], 4096], [[3], 2048], [[5], 1], [[6], 256]]),
        ("_unpack_body_kernel", "4096x2048", "620cde4bff8b1c0e4648f3756e011b37d5b222960be0acdb7104b60bcd5f007d",
         [[[4], 2048], [[5], 1], [[6], 256], [[3], 4096]]),
    ]
    audit = code.CodeAudit("sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a", tmp_path,
                           inventory=tmp_path / "inventory.json")
    p2 = {"triton_specializations": [legacy]}
    audit.p2 = p2["triton_specializations"]

    def accepted(identity):
        unit = {"unattested": [], "triton_specializations": [{"identity": identity}]}
        audit.bind_profile(unit, [identity["entry"]])
        return (unit["dependency_code"]["p2_verified"]
                and "H1 P2 did not check a prepared Triton specialization" not in unit["unattested"])

    assert accepted(legacy["identity"])
    refused = []
    declared = {case: key for case, *_, key in h1.setup_cases()} if hasattr(h1, "setup_cases") else {}
    rank1 = "P2:bytes:_window_repack_kernel:2048x4096"
    if declared:
        assert rank1 in declared, "H1_RANK1_REPACK_MISSING"
        assert declared[rank1] == declared["P2:bytes:_window_repack_kernel:4096x1024"]
    for entry, shape, digest, constants in keys:
        case = f"P2:bytes:{entry}:{shape}"
        identity = {"entry": entry, "cubin_sha256": digest, "target": target, "constants": constants}
        assert not accepted(identity)
        if hasattr(h1, "record_setup_p2"):
            key = {**declared[case], "cubin_sha256": digest}
            assert key == identity
            captured = code.binding(None, case, identity)
            expected = np.array([0xAB, 0xCD], dtype=np.uint8)
            expected_changed = np.array([0xAB, 0xCC], dtype=np.uint8)
            wrong = h1.check_setup_bytes(expected, expected, expected, expected_changed, 0)
            h1.record_setup_p2(p2, case, key, [captured], wrong)
            assert not accepted(identity)
            check = h1.check_setup_bytes(expected, expected, expected_changed, expected_changed, 0)
            wrong = h1.check_setup_bytes(expected[::-1], expected, expected_changed, expected_changed, 0)
            h1.record_setup_p2(p2, case, key, [captured], wrong)
            assert not accepted(identity)
            changed = {"entry": "another_entry", "cubin_sha256": "0" * 64,
                       "target": {**target, "arch": 120}, "constants": []}
            for field, value in changed.items():
                with pytest.raises(Invalid, match="Loaded Triton cubin identity differs"):
                    h1.record_setup_p2(p2, case, key, [code.binding(None, case, {**identity, field: value})], check)
            h1.record_setup_p2(p2, case, key, [captured], check)
            assert audit.p2[-1] == captured
            for field, value in changed.items():
                assert not accepted({**identity, field: value})
        if not accepted(identity):
            refused.append(case)
    assert not refused, f"DENSE_KEYS_LACK_H1_RECORDS: {', '.join(refused)}"
    assert not h1.setup_p2_passed(p2)
    key = {**declared[rank1], "cubin_sha256": keys[0][2]}
    h1.record_setup_p2(p2, rank1, key, [code.binding(None, rank1, key)], check)
    assert h1.setup_p2_passed(p2)


@pytest.mark.parametrize("dev", ("0", "1"))
def test_unattested_code_survives_native_boundary(code, monkeypatch, dev):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    recorder = BoundaryRecorder(lambda: None, [])
    assert recorder.call("positive", "tessera", lambda: 7) == 7
    def missing():
        raise code.Unattested("missing compiled code")
    with pytest.raises(code.Unattested, match="missing compiled code"):
        recorder.call("missing", "tessera", missing)


# Exact, independently supplied symbols. CPU span names are not CUDA identities.
CANARY = ["void vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<float>()",
          "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)",
          "void routed_fused_kernel<true, 0, false, false, 4, false, 64, true, false>(Params)",
          "void dense_reduce_kernel<4>(Params)"]
CUTLASS = "void cutlass::Kernel<at::cuda::ScaledMM>()"


@pytest.fixture
def profiled(code, tmp_path, monkeypatch):
    from experiments import routed_pair_oracle
    from experiments.t4_code import sass_dump

    native_functions = ["_Zrouted", "_Zreduce"]
    native_names = CANARY[2:]
    names = CANARY + [CUTLASS]
    inventory = json.loads(code.IMAGE_CODE.read_text())
    quantizer, = [r for r in inventory["libraries"] if r["path"].endswith("/vllm/_C_stable_libtorch.abi3.so")]
    torch_cuda, = [r for r in inventory["libraries"] if r["path"].endswith("/torch/lib/libtorch_cuda.so")]
    image = [{**quantizer, "origin": "image",
              "symbols": [{"symbol": "_Zquant", "demangled": CANARY[0]}]},
             {**torch_cuda, "origin": "image",
              "symbols": [{"symbol": "_Zfill", "demangled": CANARY[1]}, {"symbol": "_Zscaled_mm", "demangled": CUTLASS}]}]
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: nullcontext())
    monkeypatch.setattr(routed_pair_oracle, "kernel_table", lambda *args, **kwargs: [{"name": n} for n in names])
    monkeypatch.setattr(sass_dump, "parse_sass", lambda text: dict.fromkeys(native_functions))
    monkeypatch.setattr(row.subprocess, "check_output", lambda *args, **kwargs: "\n".join(native_names))
    original = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: original(*args, **{**kwargs, "device": "cpu"}))
    path = tmp_path / "library.sass"
    path.write_text("independent native function roster")
    library = {"path": "/mapped/tessera.so", "sass": {"path": str(path), "sha256": common.sha256(path)},
               "library_so_sha256": "c" * 64, "library_code_sha256": "d" * 64, "cubin_sha256s": []}
    adapter = SimpleNamespace(cols=2, library="e4m3mma", served_identity=library, apply=lambda *args: None)
    audit = code.CodeAudit("image", tmp_path)
    audit.saved = (object, None, None)
    monkeypatch.setattr(audit, "image_libraries", lambda: image)
    monkeypatch.setattr(audit, "unaudited_libraries", lambda libraries: [], raising=False)
    recorder = BoundaryRecorder(lambda: None, [])
    recorder.code_audit = audit
    def run():
        unit = row.new_unit({"m_config": {"M": 2}, "arm": {}})
        row.profile_unit(adapter, unit, recorder, routed=False)
        return unit
    return run, names, image, audit


def test_canary_kernel_origins_cover_vllm_aten_native_and_scaled_mm(profiled):
    run, _, _, _ = profiled
    unit = run()
    assert not unit["findings"] and not unit["unattested"]
    assert len(unit["kernel_origins"]) == 5
    origins = {r["profiler_name"]: r["origin"] for r in unit["kernel_origins"]}
    assert origins == dict(zip(CANARY + [CUTLASS], ["image", "image", "tessera", "tessera", "image"]))


@pytest.fixture
def offline_image_audit(code, tmp_path, monkeypatch):
    import ctypes
    import sys
    from experiments.numerics_rows import dependency_audit as audit

    libraries = []
    names = {}
    for suffix, symbol, name in (("vllm/_C_stable_libtorch.abi3.so", "_Zquant", CANARY[0]),
                                 ("torch/lib/libtorch_cuda.so", "_Zfill", CANARY[1]),
                                 ("vllm/_moe_C_stable_libtorch.abi3.so", "_Zmoe", "moe_kernel")):
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pinned library bytes: " + symbol.encode())
        libraries.append({"path": str(path), "sha256": common.sha256(path)})
        names[str(path)] = (symbol, name)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"image_digest": "pinned-image", "libraries": libraries}))
    monkeypatch.setattr(code, "IMAGE_CODE", inventory)
    monkeypatch.setattr(audit, "image_inputs", lambda out: {"runtime": {"image_digest": "pinned-image"}})
    def no_driver(*args, **kwargs):
        raise OSError("libcuda.so.1: cannot open shared object file")
    monkeypatch.setattr(ctypes, "CDLL", no_driver)
    monkeypatch.setattr(code.CodeAudit, "install", lambda *args: pytest.fail("offline inspection installed a runtime observer"))
    inspected = []
    def symbols(command, **kwargs):
        assert command[:2] == ["cuobjdump", "-symbols"]
        path = command[2]
        inspected.append(path)
        return SimpleNamespace(returncode=0, stdout="STT_FUNC STB_GLOBAL STO_ENTRY " + names[path][0] + "\n", stderr="")
    monkeypatch.setattr(code.subprocess, "run", symbols)
    def demangle(command, **kwargs):
        assert command == ["c++filt"]
        table = {symbol: name for symbol, name in names.values()}
        return "\n".join(table[s] for s in kwargs["input"].splitlines()) + "\n"
    monkeypatch.setattr(code.subprocess, "check_output", demangle)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["dependency_audit", "--out", str(out), "--code-rosters"])
    return audit.main, out, inventory, libraries, names, inspected


def test_image_symbol_audit_reads_files_without_loading_driver(offline_image_audit):
    run, out, _, libraries, _, inspected = offline_image_audit
    run()
    result = json.loads((out / "image-code-rosters.json").read_text())
    assert inspected == [r["path"] for r in libraries]
    assert result["device_launches"] == 0
    for key, marker in (("quantizer", "vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided"), ("fill", "FillFunctor")):
        witness = result["canary_2_image_witnesses"][key]
        assert any(marker in r["demangled"] for r in witness["symbols"])
        assert any((r["path"], r["sha256"]) == (witness["path"], witness["sha256"]) for r in libraries)


@pytest.mark.parametrize("mutation", ["image", "bytes", "quantizer", "fill"])
def test_image_symbol_audit_refuses_changed_or_missing_code(offline_image_audit, mutation):
    run, out, inventory, libraries, names, _ = offline_image_audit
    run()
    target = out / "image-code-rosters.json"
    target.unlink()
    if mutation == "image":
        record = json.loads(inventory.read_text())
        record["image_digest"] = "another-image"
        inventory.write_text(json.dumps(record))
    elif mutation == "bytes":
        Path(libraries[0]["path"]).write_bytes(b"changed library")
    else:
        path = libraries[0 if mutation == "quantizer" else 1]["path"]
        names[path] = (names[path][0], "another_kernel")
    with pytest.raises((Invalid, RuntimeError), match="inventory|identity|roster"):
        run()
    assert not target.exists()


@pytest.mark.parametrize("mutation", ("unknown", "image_missing", "ambiguous"))
def test_profile_refuses_a_kernel_without_a_unique_code_owner(profiled, mutation, monkeypatch):
    run, names, image, audit = profiled
    positive = run()
    assert not positive["unattested"] and not positive["findings"]
    if mutation == "unknown": names.append("unknown_generated_kernel")
    elif mutation == "image_missing": image.pop()
    else: image.append({**copy.deepcopy(image[0]), "path": "/other/image.so"})
    # A changed library set comes with a changed mapping; the audit runs again only then.
    monkeypatch.setattr(importlib.import_module(type(audit).__module__), "mapped_executables", lambda: ["remapped"], raising=False)
    assert "profiled kernel origin is missing or ambiguous" in run()["unattested"]


@pytest.mark.parametrize("dev", ("0", "1"))
def test_profile_checks_image_file_identity_before_accepting_origin(profiled, monkeypatch, dev):
    run, _, _, audit = profiled
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    assert not run()["unattested"]
    def changed():
        raise Invalid("image library identity: loaded bytes changed")
    monkeypatch.setattr(audit, "image_libraries", changed)
    monkeypatch.setattr(importlib.import_module(type(audit).__module__), "mapped_executables", lambda: ["remapped"], raising=False)
    with pytest.raises(Invalid, match="loaded bytes changed"):
        run()


def test_source_read_audit_is_not_an_attestation_gate(profiled):
    run, _, _, _ = profiled
    unit = run()
    assert not unit["unattested"]
    assert "dependency_audit_complete" not in unit["dependency_code"]


@pytest.mark.parametrize("mutation", ("missing_observer", "missing_specializations", "missing_origin", "duplicate_origin", "unknown_origin", "unbound_compiled"))
def test_join_refuses_incomplete_unit_code_bindings(code, monkeypatch, mutation):
    from experiments.numerics_rows.join import join
    from tests.test_numerics_rows_receipts import joined_inputs
    pop, digest, manifests, receipts = joined_inputs(monkeypatch)
    assert join(pop, digest, manifests, receipts)["verdict"] == "PASS"
    unit = receipts[0][1]["units"][0]
    if mutation == "missing_observer": del unit["dependency_code"]
    elif mutation == "missing_specializations": del unit["triton_specializations"]
    else:
        unit["arm"]["profiler_kernels"] = [{"name": "kernel"}]
        origin = {"profiler_name": "kernel", "origin": "image", "path": "/image/library.so", "sha256": "a" * 64, "symbol": "kernel"}
        unit["kernel_origins"] = [origin]
        if mutation == "missing_origin": unit["kernel_origins"] = []
        elif mutation == "duplicate_origin": unit["kernel_origins"].append(copy.deepcopy(origin))
        elif mutation == "unknown_origin": origin["origin"] = "unverified cache"
        else: origin.update(origin="compiled", identity={"entry": "kernel"})
    with pytest.raises(Invalid, match="code|kernel|compiled"):
        join(pop, digest, manifests, receipts)


def test_image_origin_hashes_the_executable_mapped_file_at_its_image_path(code, tmp_path, monkeypatch):
    path = Path(torch._C.__file__).resolve()
    inventory = tmp_path / "image.json"
    common.write_json(inventory, {"image_digest": "image", "libraries": [{"path": str(path), "sha256": common.sha256(path)}]})
    monkeypatch.setattr(code, "image_symbols", lambda p: [{"symbol": "_Zfill", "demangled": CANARY[1]}])
    audit = code.CodeAudit("image", tmp_path, inventory=inventory)
    result, = audit.image_libraries()
    assert result["path"] == str(path) and result["sha256"] == common.sha256(path)
    changed = json.loads(inventory.read_text())
    changed["libraries"][0]["sha256"] = "0" * 64
    common.write_json(inventory, changed)
    with pytest.raises(Invalid, match="loaded native bytes changed"):
        audit.image_libraries()


@pytest.mark.parametrize("field", ("cubin_sha256", "entry", "target", "constants", "launch", "case", "unit_id", "duplicate", "missing"))
def test_join_refuses_changed_or_missing_specialization_bindings(code, monkeypatch, field):
    from experiments.numerics_rows.join import join
    from tests.test_numerics_rows_receipts import joined_inputs
    pop, digest, manifests, receipts = joined_inputs(monkeypatch)
    unit = receipts[0][1]["units"][0]
    identity = {"cubin_sha256": "a" * 64, "entry": "compiled_entry", "target": {"backend": "cuda", "arch": 121, "warp_size": 32},
                "constants": [[[0], 64]], "launch": {"grid": [1, 1, 1], "metadata": [], "packed_metadata": [4, 0]}}
    record = code.binding(copy.deepcopy(unit["unit_id"]), "sweep:0", identity)
    unit["triton_specializations"] = [record]
    unit["dependency_code"]["p2_verified"] = True
    unit["arm"]["profiler_kernels"] = [{"name": "compiled_entry"}]
    unit["kernel_origins"] = [{"profiler_name": "compiled_entry", "origin": "compiled",
                               "code": {k: v for k, v in identity.items() if k != "launch"}}]
    assert join(pop, digest, manifests, receipts)["verdict"] == "PASS"
    if field in identity: identity[field] = "different"
    elif field == "case": record["case"] = "different"
    elif field == "unit_id": record["unit_id"] = ["another unit"]
    elif field == "duplicate": unit["triton_specializations"].append(copy.deepcopy(record))
    else: unit["triton_specializations"] = []
    with pytest.raises(Invalid, match="compiled|specialization|kernel|Triton"):
        join(pop, digest, manifests, receipts)


@pytest.mark.parametrize("mutation", ("shared_name", "unaudited_only", "unreadable_roster", "unreadable_file"))
def test_profile_refuses_unaudited_mapped_device_owners(profiled, code, tmp_path, monkeypatch, mutation):
    from tests.test_numerics_rows_receipts import elf_bytes
    from tessera._dev import native_identity
    import os
    run, names, _, audit = profiled
    # Exercise real mapped-owner discovery, controlling only /proc and cuobjdump.
    if hasattr(type(audit), "unaudited_libraries"):
        monkeypatch.setattr(audit, "unaudited_libraries", type(audit).unaudited_libraries.__get__(audit))
    path = tmp_path / "other.so"
    path.write_bytes(elf_bytes())  # A host-only launcher must not block attestation.
    stat = path.stat()
    original = Path.read_text
    active = False
    def read_maps(self, *args, **kwargs):
        if str(self) == "/proc/self/maps":
            return f"1000-2000 r-xp 00000000 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} {stat.st_ino} {path}\n" if active else ""
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_maps)
    monkeypatch.setattr(native_identity, "mapped_file_device", lambda fd: (os.major(stat.st_dev), os.minor(stat.st_dev)))
    monkeypatch.setattr(code, "image_symbols", lambda p: [{"symbol": "_Zother", "demangled": CANARY[2]}])
    assert not run()["unattested"]
    active = True
    assert not run()["unattested"]
    path.write_bytes(elf_bytes(extra_sections=[(".nv_fatbin", 1, 2, b"device code", 11)]))
    if mutation == "unaudited_only":
        names.append("only_in_other_file")
        monkeypatch.setattr(code, "image_symbols", lambda p: [{"symbol": "only_in_other_file", "demangled": "only_in_other_file"}])
    elif mutation == "unreadable_roster":
        def unreadable(path): raise code.Unattested("roster unavailable")
        monkeypatch.setattr(code, "image_symbols", unreadable)
    elif mutation == "unreadable_file":
        path.write_bytes(b"unreadable ELF sections")
    unit = run()
    assert unit["unattested"]
    if mutation in ("shared_name", "unaudited_only"):
        assert any(any(c["origin"] == "unaudited" for c in r["candidates"]) for r in unit["unresolved_kernel_origins"])


EMPTY_ROSTERS = Path(__file__).parent.parent / "experiments" / "numerics_rows" / "empty-rosters.json"
FATBIN_ELF = dict(extra_sections=[(".nv_fatbin", 1, 2, b"device code", 11)])
EMPTY_SYMBOLS = "\nFatbin elf code:\n================\narch = sm_75\ncode version = [1,8]\nhost = linux\ncompile_size = 64bit\ncompressed\n\nsymbols:\n"


def replay_mapped(profiled, tmp_path, monkeypatch, files, calls=None, after_first=None):
    """Map each {name: (file bytes, -symbols listing)} and run the real audit; cuobjdump replays the listing of the
    bytes it is given to read (the verified descriptor or a path); every other call runs."""
    from tessera._dev import native_identity
    import hashlib, os, subprocess
    run, _, _, audit = profiled
    monkeypatch.setattr(audit, "unaudited_libraries", type(audit).unaudited_libraries.__get__(audit))
    paths = {}
    for name, (data, _) in files.items():
        paths[name] = tmp_path / name
        paths[name].write_bytes(data)
    stats = {name: path.stat() for name, path in paths.items()}
    maps = "".join(f"1000-2000 r-xp 00000000 {os.major(s.st_dev):x}:{os.minor(s.st_dev):x} {s.st_ino} {paths[n]}\n"
                   for n, s in stats.items())
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: maps if str(self) == "/proc/self/maps" else original(self, *a, **k))
    device = next(iter(stats.values())).st_dev
    monkeypatch.setattr(native_identity, "mapped_file_device", lambda fd: (os.major(device), os.minor(device)))
    by_content = {hashlib.sha256(data).hexdigest(): listing for data, listing in files.values()}
    real, pending = subprocess.run, [after_first] if after_first else []
    def cuobjdump(argv, *args, **kwargs):
        listing = by_content.get(hashlib.sha256(Path(argv[-1]).read_bytes()).hexdigest()) if argv[0] == "cuobjdump" else None
        if listing is None:
            return real(argv, *args, **kwargs)
        if calls is not None:
            calls.append(argv[-1])
        while pending:
            pending.pop()(paths)
        return subprocess.CompletedProcess(argv, 0, listing if kwargs.get("text") else listing.encode(), "" if kwargs.get("text") else b"")
    monkeypatch.setattr(subprocess, "run", cuobjdump)
    unit = run()
    return unit, unit["dependency_code"].get("proved_empty_rosters", []), paths


def reviewed(code, tmp_path, monkeypatch, data):
    """A reviewed-proof file that lists exactly `data`, in the real file's form."""
    import hashlib
    proofs = tmp_path / "empty-rosters.json"
    entry = {"elf": 1, "ptx": 0, "sass_functions": 0, "image_paths": ["/image/lib.so"],
             "listings_sha256": {"symbols": "a" * 64, "lelf": "b" * 64, "lptx": "c" * 64}}
    proofs.write_text(json.dumps({"files": {hashlib.sha256(data).hexdigest(): entry}}))
    monkeypatch.setattr(code, "EMPTY_ROSTERS", proofs, raising=False)
    return proofs


def test_empty_roster_with_a_reviewed_proof_is_accepted_and_recorded(profiled, code, tmp_path, monkeypatch):
    import hashlib
    from tests.test_numerics_rows_receipts import elf_bytes
    data = elf_bytes(**FATBIN_ELF)
    proofs = reviewed(code, tmp_path, monkeypatch, data)
    unit, recorded, paths = replay_mapped(profiled, tmp_path, monkeypatch, {"mapped.so": (data, EMPTY_SYMBOLS)})
    assert not unit["unattested"], unit["unattested"]
    assert recorded == [{"path": str(paths["mapped.so"]), "sha256": hashlib.sha256(data).hexdigest(), "elf_count": 1, "ptx_count": 0,
                         "evidence_sha256": hashlib.sha256(proofs.read_bytes()).hexdigest(),
                         "symbols_sha256": "a" * 64, "lelf_sha256": "b" * 64, "lptx_sha256": "c" * 64}], unit["unattested"]


def test_empty_roster_without_a_reviewed_proof_stays_refused(profiled, tmp_path, monkeypatch):
    """Any other empty roster refuses, whatever its listings say: the bytes are not a reviewed entry."""
    import hashlib
    from tests.test_numerics_rows_receipts import elf_bytes
    data = elf_bytes(**FATBIN_ELF)
    unit, recorded, paths = replay_mapped(profiled, tmp_path, monkeypatch, {"mapped.so": (data, EMPTY_SYMBOLS)})
    assert recorded == [], unit["unattested"]
    assert unit["unattested"][-1] == (f"mapped file or device-code roster unavailable: {paths['mapped.so']}: Unattested: mapped "
                                      f"device-code roster is empty and sha256 {hashlib.sha256(data).hexdigest()} has no reviewed proof"), unit["unattested"]


def test_reviewed_empty_rosters_are_the_proved_files():
    """The evidence: five IMG-N files proved by Flux jobs to hold one ELF with no function, no PTX, no SASS function.
    libcudnn_ext.so.9.26.0 (22 sm_121 PTX) is not among them."""
    assert EMPTY_ROSTERS.is_file(), "REVIEWED_EMPTY_ROSTERS_MISSING"
    proofs = json.loads(EMPTY_ROSTERS.read_text())
    assert sorted(h[:8] for h in proofs["files"]) == ["3053d32e", "58d0f3c5", "5e2be143", "7bceb090", "af44d604"]
    assert not any(h.startswith("785ba5ca") for h in proofs["files"])
    assert proofs["evidence_jobs"] == {"roster_diag": "fHmSFFMKeP", "fatbin_diag": "fHorQAsQk7", "sass_functions": "fHsHu8S4jy", "listings": "fHrdtcL1GK"}
    assert all((e["elf"], e["ptx"], e["sass_functions"]) == (1, 0, 0) for e in proofs["files"].values())
    cufft = next(e for h, e in proofs["files"].items() if h.startswith("58d0f3c5"))
    assert cufft["listings_sha256"]["symbols"] == "c5aeee84b93be31cd9fadcc0d559b2c16609bc42b3eb03ee675e71110833c52d"


def test_roster_lists_the_verified_inode_not_the_path(profiled, code, tmp_path, monkeypatch):
    """cuobjdump reads /proc/<pid>/fd/<n> of the checked open file: a path swapped after the check cannot reach it."""
    import os
    from tests.test_numerics_rows_receipts import elf_bytes
    data, calls = elf_bytes(**FATBIN_ELF), []
    reviewed(code, tmp_path, monkeypatch, data)
    replay_mapped(profiled, tmp_path, monkeypatch, {"mapped.so": (data, EMPTY_SYMBOLS)}, calls)
    assert calls and all(c.startswith(f"/proc/{os.getpid()}/fd/") for c in calls), f"LISTED_BY_PATH: {calls}"


@pytest.mark.parametrize("change", ("rewritten", "grown"))
def test_empty_roster_refuses_bytes_that_change_while_read(profiled, code, tmp_path, monkeypatch, change):
    """The file changes in place (same inode) after the first listing: the hash, the snapshot listing and the mapped
    inode no longer hold the same bytes, so the reviewed proof is not used. A grown file is found even when each
    read returns at most the original length."""
    import os
    from tests.test_numerics_rows_receipts import elf_bytes
    data = elf_bytes(**FATBIN_ELF)
    reviewed(code, tmp_path, monkeypatch, data)
    if change == "grown":
        read = os.read
        monkeypatch.setattr(os, "read", lambda fd, n: read(fd, min(n, len(data))))
    def rewrite(paths):
        with open(paths["mapped.so"], "r+b" if change == "rewritten" else "ab") as stream:
            stream.write(b"\0" * 8)
    unit, recorded, paths = replay_mapped(profiled, tmp_path, monkeypatch, {"mapped.so": (data, EMPTY_SYMBOLS)}, after_first=rewrite)
    assert recorded == [], unit["unattested"]
    assert unit["unattested"][-1].startswith(f"mapped file or device-code roster unavailable: {paths['mapped.so']}: Unattested: "
                                             "mapped device-code roster changed while it was read"), unit["unattested"]


def test_roster_proof_survives_another_files_failure(profiled, code, tmp_path, monkeypatch):
    from tests.test_numerics_rows_receipts import elf_bytes
    data = elf_bytes(**FATBIN_ELF)
    reviewed(code, tmp_path, monkeypatch, data)
    unit, recorded, paths = replay_mapped(profiled, tmp_path, monkeypatch, {
        "a.so": (data, EMPTY_SYMBOLS), "b.so": (b"unreadable ELF sections", "")})
    assert [r["path"] for r in recorded] == [str(paths["a.so"])], unit["unattested"]
    assert unit["unattested"][-1] == (f"mapped file or device-code roster unavailable: {paths['b.so']}: "
                                      "ValueError: unsupported ELF encoding"), unit["unattested"]


def test_roster_reads_the_symbol_type_field(code, monkeypatch):
    """A data symbol whose name holds STT_FUNC, or a bare STT_FUNC line, is not a function; a failed listing names its
    flag and exit code, also when its stderr is not UTF-8."""
    import subprocess
    listing = EMPTY_SYMBOLS + "STT_OBJECT STB_LOCAL STV_DEFAULT foo.STT_FUNC\nSTT_FUNC\n"
    monkeypatch.setattr(code.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, listing, ""))
    monkeypatch.setattr(code.subprocess, "check_output", lambda argv, input, **kwargs: input)  # c++filt: names unchanged
    assert code.image_symbols("/image/lib.so") == [], "DATA_SYMBOL_NAMED_STT_FUNC_COUNTED"
    monkeypatch.setattr(code.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 7, "", b"err\xff\n".decode(errors="replace")))
    with pytest.raises(code.Unattested, match=r"^image CUDA symbol roster unavailable: /image/lib\.so: cuobjdump -symbols exit 7: err"):
        code.image_symbols("/image/lib.so")


def test_case_status_keeps_the_roster_cause_before_a_later_effect():
    from experiments.numerics_rows import dense_phases
    assert hasattr(dense_phases, "case_status"), "CASE_STATUS_PREFERS_ROSTER_CAUSE"
    roster = "mapped file or device-code roster unavailable: /x.so: Unattested: mapped device-code roster is empty"
    unit = {"findings": [], "invalid": [], "cuda_errors": [],
            "unattested": [roster, "profiled kernel origin is missing or ambiguous", "NOT_EXERCISED: lane"]}
    assert dense_phases.case_status(unit) == {"findings": [], "invalid": [], "unattested": [roster], "cuda_errors": []}
    unit["unattested"] = ["profiled kernel origin is missing or ambiguous", "NOT_EXERCISED: lane"]
    assert dense_phases.case_status(unit)["unattested"] == ["NOT_EXERCISED: lane"]


@pytest.mark.parametrize("mutation", ("unreadable_roster", "unreadable_file"))
def test_roster_refusal_names_its_failing_step(profiled, code, tmp_path, monkeypatch, mutation):
    """One reason line per failure, with the error class and the tool's first error line."""
    from tests.test_numerics_rows_receipts import elf_bytes
    data = elf_bytes(**FATBIN_ELF) if mutation == "unreadable_roster" else b"unreadable ELF sections"
    if mutation == "unreadable_roster":
        def unreadable(path): raise code.Unattested("image CUDA symbol roster unavailable: first line\nsecond line")
        monkeypatch.setattr(code, "image_symbols", unreadable)
    unit, _, paths = replay_mapped(profiled, tmp_path, monkeypatch, {"mapped.so": (data, {})})
    cause = {"unreadable_roster": "Unattested: image CUDA symbol roster unavailable: first line",
             "unreadable_file": "ValueError: unsupported ELF encoding"}[mutation]
    assert unit["unattested"][-1] == f"mapped file or device-code roster unavailable: {paths['mapped.so']}: {cause}", unit["unattested"]


def test_library_audit_runs_once_per_mapped_key(code, tmp_path, monkeypatch):
    audit = code.CodeAudit("image", tmp_path)
    audit.saved = (object, None, None)
    image = [{"origin": "image", "path": "/lib/a.so", "sha256": "a" * 64, "symbols": [{"symbol": "k", "demangled": "k"}]}]
    calls, key = [], ["k1"]
    monkeypatch.setattr(audit, "image_libraries", lambda: calls.append(key[0]) or [dict(r) for r in image])
    monkeypatch.setattr(audit, "unaudited_libraries", lambda libraries: [], raising=False)
    monkeypatch.setattr(code, "mapped_executables", lambda: [key[0]], raising=False)
    def case():
        unit = {"unattested": [], "triton_specializations": []}
        audit.bind_profile(unit, ["k"])
        assert not unit["unattested"] and unit["kernel_origins"][0]["sha256"] == image[0]["sha256"], unit
        return unit["dependency_code"].get("library_audit", {})
    first = [case() for _ in range(3)]
    assert len(calls) == 1, "LIBRARY_AUDIT_PER_CASE"
    assert [r.get("audited_at_case") for r in first] == [True, False, False]
    key[0] = "k2"
    remapped = case()
    assert calls == ["k1", "k2"] and remapped["audited_at_case"] and remapped["mapped_key_sha256"] != first[0]["mapped_key_sha256"]
    assert remapped["audit_sha256"] == first[0]["audit_sha256"]
    image[0]["sha256"], key[0] = "b" * 64, "k3"
    assert case()["audit_sha256"] != first[0]["audit_sha256"]


def test_mapped_key_sees_a_write_to_the_same_inode(code, tmp_path):
    assert hasattr(code, "mapped_executables"), "MAPPED_KEY_ABSENT"
    path = tmp_path / "lib.so"
    path.write_bytes(b"one")
    maps = f"1000-2000 r-xp 00000000 00:01 {path.stat().st_ino} {path}\n1000-2000 r-xp 00000000 00:01 9 /gone.so\n"
    before = code.mapped_executables(maps)
    path.write_bytes(b"two!")
    after = code.mapped_executables(maps)
    entries = {e[0]: e for e in map(json.loads, after)}
    assert before != after and entries[str(path)][3][:3] == [4, path.stat().st_ctime_ns, path.stat().st_ino]
    assert entries["/gone.so"] == ["/gone.so", "00:01", "9", None]
