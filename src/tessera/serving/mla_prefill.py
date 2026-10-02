"""Build/load the recovered single-shape kernel using Tessera's toolchain resolver.

Two entry points, one build recipe:

* :class:`MlaPrefillBuild` is the build owner. Every caller -- a CPU-only
  container build and a later numerical run -- resolves the same source, flags,
  toolchain, includes, build id and retained DSO through it, so a CPU-built
  artifact is the one the run selects.
* :class:`MlaPrefillLibrary` is the runtime path: it keeps the CUDA-device/SM121
  gate and binds the retained DSO with ctypes.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import torch
from torch.utils.cpp_extension import load
from tessera.serving import ext

FLAGS = ['-std=c++17','--threads','1','-use_fast_math','-DTESSERA_MLA_DECLARED_FAST_MATH=1','-O3',
         '-gencode=arch=compute_121a,code=sm_121a','-static-global-template-stub=false',
         '--expt-relaxed-constexpr','-DFLASHINFER_ENABLE_F16','-DFLASHINFER_ENABLE_BF16',
         '-DFLASHINFER_ENABLE_FP8_E4M3','-DFLASHINFER_ENABLE_FP8_E5M2','-DFLASHINFER_ENABLE_FP8_E8M0',
         '-DFLASHINFER_ENABLE_FP4_E2M1','-DNDEBUG']

#: Experiment-only build selectors. NOT serving flags: the default build has
#: TESSERA_MLA_P0_BUFFERS=0 and is the shipped L0 kernel. Each selection is
#: folded into the build identity, so two selections never share a cached .so.
MUTATION_FLAG = '-DTESSERA_MLA_MASK_GATE_DROP_LAST=1'
P0_BUFFERS_FLAG = '-DTESSERA_MLA_P0_BUFFERS=1'
P0_WRONG_PASS_FLAG = '-DTESSERA_MLA_P0_WRONG_PASS=1'

#: The loader's exact compiler gate. Printing a version is not enforcement.
REQUIRED_NVCC = re.compile(r'V13\.0\.88\b')
MODULE_PREFIX = 'tessera_mla_prefill_'


def select_flags(*, mutation=False, p0_buffers=False, p0_wrong_pass=False) -> list:
    """The build flags for one selection. The wrong-pass mutant is defined only
    as a coupling of the pass-buffer schedule, so it requires p0_buffers."""
    if any(type(value) is not bool for value in (mutation, p0_buffers, p0_wrong_pass)):
        raise ValueError('MLA build selectors must be exact booleans')
    if mutation and (p0_buffers or p0_wrong_pass):
        raise ValueError('mask-drop and pass-buffer experiments are separate selections')
    if p0_wrong_pass and not p0_buffers:
        raise ValueError('p0_wrong_pass is an experiment mutant of p0_buffers; set both')
    flags = list(FLAGS)
    if mutation:
        flags.append(MUTATION_FLAG)
    if p0_buffers:
        flags.append(P0_BUFFERS_FLAG)
    if p0_wrong_pass:
        flags.append(P0_WRONG_PASS_FLAG)
    return flags


def resolve_toolchain():
    """The exact nvcc the build uses, with the loader's version gate enforced."""
    report = ext.toolchain_report(torch)
    nvcc = ext._nvcc_for_build()
    if not nvcc or not report['complete']:
        raise RuntimeError('requires the complete resolved CUDA/ninja toolchain')
    nvcc = str(Path(nvcc).resolve())
    version = subprocess.check_output([nvcc, '--version'], text=True)
    if not REQUIRED_NVCC.search(version):
        raise RuntimeError('requires nvcc13.0.88: ' + version)
    ext._resolve_ninja()
    return nvcc, version


class Identity(ctypes.Structure):
    _fields_ = [(name,ctypes.c_int) for name in ['abi','cudacc_major','cudacc_minor','cudacc_build',
                'declared_fast_math','block_threads','heads','tile_entries','model','groups']] + [
                ('stock_smem',ctypes.c_longlong),('l0_smem',ctypes.c_longlong)]

class MlaPrefillBuild:
    """The one build owner.

    Produces (and retains) the DSO ``MlaPrefillLibrary`` will load, using the
    canonical FlashInfer/CCCL includes and the toolchain resolver. It does not
    check for a CUDA device: a build may happen on a CPU box. The device gate
    lives in the runtime class, the only path that runs the kernel.
    """

    def __init__(self, build_directory, *, mutation=False, p0_buffers=False,
                 p0_wrong_pass=False, require_retained=False):
        from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR, CCCL_INCLUDE_DIRS
        self.flags = select_flags(mutation=mutation, p0_buffers=p0_buffers,
                                  p0_wrong_pass=p0_wrong_pass)
        if build_directory is None:
            from torch.utils.cpp_extension import _get_build_directory
            build_directory = _get_build_directory('tessera_mla_prefill', verbose=False)
        self.build_directory = str(Path(build_directory).resolve())
        self.nvcc, self.nvcc_version = resolve_toolchain()
        self.source = Path(ext.native_source_path(MODULE_PREFIX))
        self.source_sha256 = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.build_id = hashlib.sha256((self.source_sha256 + repr(self.flags)).encode()).hexdigest()
        self.name = f'{MODULE_PREFIX}{self.build_id[:16]}'
        self.library_path = Path(self.build_directory) / f'{self.name}.so'
        includes = [str(Path(p).resolve()) for p in CCCL_INCLUDE_DIRS] + [
            str(Path(FLASHINFER_INCLUDE_DIR).resolve())]
        header = Path(FLASHINFER_INCLUDE_DIR) / (
            'flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_mg.cuh')
        self.inputs = {
            'source_sha256': self.source_sha256, 'flags': list(self.flags),
            'nvcc_version': self.nvcc_version, 'nvcc_sha256': _sha256(Path(self.nvcc)),
            'include_roots': includes, 'stock_prefill_header_sha256': _sha256(header),
            # The experiment wrapper binds these roots inside its immutable
            # image. This declaration is provenance, not a host attestation.
            'declared_image': os.environ.get('TESSERA_CENSUS_RUNTIME_IMAGE'),
            'torch_version': torch.__version__,
            'mutation': mutation, 'p0_buffers': p0_buffers, 'p0_wrong_pass': p0_wrong_pass,
        }
        if self.manifest_path.exists():
            self.manifest = self._read_retained()
            self.path = str(self.library_path)
            return
        if require_retained:
            raise RuntimeError(f'missing retained MLA build: {self.manifest_path}')
        Path(self.build_directory).mkdir(parents=True, exist_ok=True)
        self.path = load(name=self.name, sources=[str(self.source)], build_directory=self.build_directory,
                         extra_include_paths=includes,
                         extra_cuda_cflags=self.flags, extra_cflags=['-O3'], is_python_module=False, verbose=True)
        if Path(self.path).resolve() != self.library_path.resolve():
            raise RuntimeError('JIT returned a library outside the declared identity')
        ninja = Path(self.build_directory) / 'build.ninja'
        self.manifest = {
            'schema': 'tessera.mla_prefill.build.v1',
            **self.inputs,
            'source': str(self.source), 'source_sha256': self.source_sha256,
            'flags': list(self.flags),
            'fast_math_evidence': 'explicit build declaration, not a compiler predicate',
            'nvcc': self.nvcc, 'nvcc_version': self.nvcc_version,
            'nvcc_sha256': self.inputs['nvcc_sha256'],
            'ninja_sha256': hashlib.sha256(ninja.read_bytes()).hexdigest(),
            'build_id': self.build_id, 'name': self.name,
            'library': str(self.library_path),
            'library_bytes': self.library_path.stat().st_size,
            'library_sha256': hashlib.sha256(self.library_path.read_bytes()).hexdigest(),
            'stock_prefill_header_sha256': self.inputs['stock_prefill_header_sha256'],
        }
        self._retain_manifest()

    @property
    def manifest_path(self) -> Path:
        return Path(self.build_directory) / f'{self.name}.manifest.json'

    def _retain_manifest(self):
        """Publish after the DSO is complete; never overwrite retained evidence."""
        path = self.manifest_path
        with path.open('x') as handle:
            handle.write(json.dumps(self.manifest, indent=2, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def _read_retained(self):
        """Refuse drift BEFORE loading or rebuilding an existing executable."""
        with self.manifest_path.open('rb') as handle:
            raw = handle.read(65537)
        if len(raw) > 65536:
            raise RuntimeError('retained MLA manifest exceeds 64KiB')
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or manifest.get('schema') != 'tessera.mla_prefill.build.v1':
            raise RuntimeError('unrecognized retained MLA build manifest')
        for key, value in self.inputs.items():
            if manifest.get(key) != value:
                raise RuntimeError(f'retained MLA build input drift: {key}')
        if (manifest.get('build_id'), manifest.get('name'), manifest.get('library')) != (
                self.build_id, self.name, str(self.library_path)):
            raise RuntimeError('retained MLA build path/selection mismatch')
        if (self.library_path.stat().st_size != manifest.get('library_bytes')
                or _sha256(self.library_path) != manifest.get('library_sha256')
                or _sha256(Path(self.build_directory) / 'build.ninja') != manifest.get('ninja_sha256')):
            raise RuntimeError('retained MLA executable/build recipe drift')
        return manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MlaPrefillLibrary:
    def __init__(self, build_directory, *, mutation=False, p0_buffers=False,
                 p0_wrong_pass=False, require_retained=False):
        # Runtime gate: the kernel is compiled for a specific device, so it is
        # only run where that device is present.
        if torch.cuda.get_device_capability() != (12,1):
            raise RuntimeError('requires SM121')
        self.build = MlaPrefillBuild(build_directory, mutation=mutation,
                                     p0_buffers=p0_buffers, p0_wrong_pass=p0_wrong_pass,
                                     require_retained=require_retained)
        self.source = self.build.source
        self.source_sha256 = self.build.source_sha256
        self.flags = self.build.flags
        self.build_id = self.build.build_id
        self.path = self.build.path
        self.build_manifest = self.build.manifest
        self.handle = ctypes.CDLL(str(self.build.library_path))
        self.handle.tessera_mla_prefill_identity.argtypes=[ctypes.POINTER(Identity)]
        self.handle.tessera_mla_prefill_identity.restype=ctypes.c_int
        identity=Identity();self.handle.tessera_mla_prefill_identity(ctypes.byref(identity))
        self.identity={name:getattr(identity,name) for name,_ in Identity._fields_}
        expected={'abi':1,'cudacc_major':13,'cudacc_minor':0,'cudacc_build':88,'declared_fast_math':1,
                  'block_threads':384,'heads':32,'tile_entries':64,'model':3,'groups':2}
        if any(self.identity[k]!=v for k,v in expected.items()):raise RuntimeError(self.identity)
        self.launch=self.handle.tessera_mla_prefill_launch
        self.launch.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*5+[ctypes.c_float]*2+[ctypes.c_int]*3+[
                             ctypes.c_ulonglong]*3+[ctypes.c_void_p]
        self.launch.restype=ctypes.c_int

    def call(self,variant,q,kv,indices,out,lse):
        # The qualification harness supplies contiguous, resident tensors.
        rc=self.launch(variant,q.data_ptr(),kv.data_ptr(),indices.data_ptr(),out.data_ptr(),lse.data_ptr(),
                       1.0/16.0,1.0,q.shape[0],indices.shape[1],64,
                       kv.stride(0)*kv.element_size(),lse.stride(0),656,
                       torch.cuda.current_stream(q.device).cuda_stream)
        if rc:raise RuntimeError(f'kernel variant{variant} refused/failed rc{rc}')
