"""Independent code-owner rosters and a model of the pinned Triton launch API."""
from collections import namedtuple
from contextlib import nullcontext
from pathlib import Path
import json
import copy
import importlib
from types import SimpleNamespace

import pytest

from experiments.numerics_rows import common, row
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
    import torch
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
def test_profile_refuses_a_kernel_without_a_unique_code_owner(profiled, mutation):
    run, names, image, _ = profiled
    positive = run()
    assert not positive["unattested"] and not positive["findings"]
    if mutation == "unknown": names.append("unknown_generated_kernel")
    elif mutation == "image_missing": image.pop()
    else: image.append({**copy.deepcopy(image[0]), "path": "/other/image.so"})
    assert "profiled kernel origin is missing or ambiguous" in run()["unattested"]


@pytest.mark.parametrize("dev", ("0", "1"))
def test_profile_checks_image_file_identity_before_accepting_origin(profiled, monkeypatch, dev):
    run, _, _, audit = profiled
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    assert not run()["unattested"]
    def changed():
        raise Invalid("image library identity: loaded bytes changed")
    monkeypatch.setattr(audit, "image_libraries", changed)
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
    import torch
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
