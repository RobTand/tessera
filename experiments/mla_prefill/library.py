"""Build/load the recovered single-shape kernel using Tessera's toolchain resolver."""
from __future__ import annotations
import ctypes
import hashlib
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

class Identity(ctypes.Structure):
    _fields_ = [(name,ctypes.c_int) for name in ['abi','cudacc_major','cudacc_minor','cudacc_build',
                'declared_fast_math','block_threads','heads','tile_entries','model','groups']] + [
                ('stock_smem',ctypes.c_longlong),('l0_smem',ctypes.c_longlong)]

class Library:
    def __init__(self, build_directory, *, mutation=False):
        from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR, CCCL_INCLUDE_DIRS
        report = ext.toolchain_report(torch)
        nvcc = report['nvcc']
        version = subprocess.check_output([nvcc,'--version'],text=True)
        if not re.search(r'V13\.0\.88\b',version):
            raise RuntimeError('requires nvcc13.0.88: '+version)
        if torch.cuda.get_device_capability() != (12,1):
            raise RuntimeError('requires SM121')
        ext._resolve_ninja()
        self.source = Path(__file__).resolve().parents[2]/'src/tessera/serving/csrc/mla_prefill_mg.cu'
        self.source_sha256 = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.flags = FLAGS + (['-DTESSERA_MLA_MASK_GATE_DROP_LAST=1'] if mutation else [])
        self.build_id = hashlib.sha256((self.source_sha256 + repr(self.flags)).encode()).hexdigest()
        name = 'tessera_mla_prefill_' + self.build_id[:16]
        Path(build_directory).mkdir(parents=True,exist_ok=True)
        self.path = load(name=name,sources=[str(self.source)],build_directory=str(build_directory),
                         extra_include_paths=[str(p) for p in CCCL_INCLUDE_DIRS]+[str(FLASHINFER_INCLUDE_DIR)],
                         extra_cuda_cflags=self.flags,extra_cflags=['-O3'],is_python_module=False,verbose=True)
        ninja = Path(build_directory)/'build.ninja'
        self.build_manifest = {'source_sha256': self.source_sha256, 'flags': self.flags,
            'fast_math_evidence': 'explicit build declaration, not a compiler predicate',
            'nvcc_version': version, 'nvcc_sha256': hashlib.sha256(Path(nvcc).read_bytes()).hexdigest(),
            'ninja_sha256': hashlib.sha256(ninja.read_bytes()).hexdigest(),
            'library_sha256': hashlib.sha256(Path(self.path).read_bytes()).hexdigest(),
            'stock_prefill_header_sha256': hashlib.sha256((Path(FLASHINFER_INCLUDE_DIR)/
                'flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_mg.cuh').read_bytes()).hexdigest(),
            'mutation': mutation}
        self.handle = ctypes.CDLL(self.path)
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
