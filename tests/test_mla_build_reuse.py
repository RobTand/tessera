"""Retained native build identity must be checked before compiler or dlopen."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip('torch')
from tessera.serving import mla_prefill as mla


@pytest.fixture
def native_build(tmp_path, monkeypatch):
    source = tmp_path / 'kernel.cu'
    source.write_bytes(b'closed test kernel')
    nvcc = tmp_path / 'nvcc'
    nvcc.write_bytes(b'test compiler')
    include = tmp_path / 'include'
    header = include / 'flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_mg.cuh'
    header.parent.mkdir(parents=True)
    header.write_bytes(b'test header')
    env = ModuleType('flashinfer.jit.env')
    env.FLASHINFER_INCLUDE_DIR = include
    env.CCCL_INCLUDE_DIRS = []
    for name in ('flashinfer', 'flashinfer.jit'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, 'flashinfer.jit.env', env)
    monkeypatch.setattr(mla.ext, 'native_source_path', lambda _: str(source))
    monkeypatch.setattr(mla.ext, 'toolchain_report', lambda _: {'nvcc':str(nvcc), 'complete':True})
    monkeypatch.setattr(mla.ext, '_nvcc_for_build', lambda: str(nvcc))
    monkeypatch.setattr(mla.ext, '_resolve_ninja', lambda: 'ninja')
    monkeypatch.setattr(mla.subprocess, 'check_output', lambda *a, **kw: 'Cuda compilation tools V13.0.88')
    monkeypatch.setattr(mla.torch.cuda, 'get_device_capability', lambda: (12, 1))
    monkeypatch.delenv('TESSERA_CENSUS_RUNTIME_IMAGE', raising=False)
    calls = []

    def compile_library(**kw):
        calls.append('compile')
        root = Path(kw['build_directory'])
        root.mkdir(parents=True, exist_ok=True)
        path = root / (kw['name'] + '.so')
        path.write_bytes(b'fake linked DSO')
        (root / 'build.ninja').write_text('fake selected build recipe '+kw['name'])
        return str(path)

    def native_library(path):
        calls.append('dlopen')

        def identity(pointer):
            values = dict(abi=1,cudacc_major=13,cudacc_minor=0,cudacc_build=88,
                          declared_fast_math=1,block_threads=384,heads=32,
                          tile_entries=64,model=3,groups=2)
            for name, value in values.items():
                setattr(pointer._obj, name, value)
            return 0

        return SimpleNamespace(tessera_mla_prefill_identity=identity,
                               tessera_mla_prefill_launch=lambda *args: 0)

    monkeypatch.setattr(mla, 'load', compile_library)
    monkeypatch.setattr(mla.ctypes, 'CDLL', native_library)
    return SimpleNamespace(root=tmp_path/'build', calls=calls, source=source,
                           header=header, nvcc=nvcc)


def test_repeat_runtime_load_reuses_retained_dso_without_compiler(native_build):
    first = mla.MlaPrefillLibrary(native_build.root)
    second = mla.MlaPrefillLibrary(native_build.root)
    assert first.path == second.path
    assert native_build.calls == ['compile', 'dlopen', 'dlopen']


def test_tampered_retained_dso_refuses_before_native_load(native_build):
    first = mla.MlaPrefillLibrary(native_build.root)
    Path(first.path).write_bytes(b'FAKE LINKED DSO')
    native_build.calls.clear()
    with pytest.raises(RuntimeError, match='retained MLA'):
        mla.MlaPrefillLibrary(native_build.root)
    assert native_build.calls == []


@pytest.mark.parametrize('changed', ['header', 'nvcc', 'image'])
def test_changed_build_inputs_refused_before_loader(native_build, monkeypatch, changed):
    mla.MlaPrefillLibrary(native_build.root)
    if changed == 'image':
        monkeypatch.setenv('TESSERA_CENSUS_RUNTIME_IMAGE', 'different declared image')
    else:
        getattr(native_build, changed).write_bytes(b'different input bytes')
    native_build.calls.clear()
    with pytest.raises(RuntimeError, match='input drift'):
        mla.MlaPrefillLibrary(native_build.root)
    assert native_build.calls == []


def test_missing_retained_selection_never_compiles(native_build):
    with pytest.raises(RuntimeError, match='missing retained'):
        mla.MlaPrefillLibrary(native_build.root, require_retained=True)
    assert native_build.calls == []


def test_default_cannot_adopt_experimental_build(native_build):
    mla.MlaPrefillLibrary(native_build.root, p0_buffers=True)
    native_build.calls.clear()
    with pytest.raises(RuntimeError, match='missing retained'):
        mla.MlaPrefillLibrary(native_build.root, require_retained=True)
    assert native_build.calls == []


def test_selections_in_one_directory_retain_their_own_recipe(native_build):
    baseline = mla.MlaPrefillLibrary(native_build.root)
    candidate = mla.MlaPrefillLibrary(native_build.root, p0_buffers=True)
    reused = mla.MlaPrefillLibrary(native_build.root, require_retained=True)
    assert reused.path == baseline.path != candidate.path
    assert native_build.calls == ['compile','dlopen','compile','dlopen','dlopen']


def test_bound_retained_manifest_refuses_changed_evidence(native_build):
    import hashlib
    built = mla.MlaPrefillLibrary(native_build.root)
    path = built.build.manifest_path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mla.MlaPrefillLibrary(native_build.root, require_retained=True,
                         retained_manifest_sha256=digest)
    path.write_text(path.read_text()+' ')
    native_build.calls.clear()
    with pytest.raises(RuntimeError, match='manifest SHA-256 mismatch'):
        mla.MlaPrefillLibrary(native_build.root, require_retained=True,
                             retained_manifest_sha256=digest)
    assert native_build.calls == []


@pytest.mark.parametrize('selector', ['mutation', 'p0_buffers', 'p0_wrong_pass'])
@pytest.mark.parametrize('value', ['0', 1])
def test_non_boolean_build_selectors_refused(native_build, selector, value):
    with pytest.raises(ValueError, match='exact booleans'):
        mla.MlaPrefillBuild(native_build.root, **{selector:value})
    assert native_build.calls == []


@pytest.mark.parametrize('selectors', [dict(p0_wrong_pass=True), dict(mutation=True,p0_buffers=True)])
def test_ambiguous_mutation_selection_refused(native_build, selectors):
    with pytest.raises(ValueError):
        mla.MlaPrefillBuild(native_build.root, **selectors)
    assert native_build.calls == []
