"""Experimental code-artifact callback guards; real CUDA load is GPU-only proof."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import pytest

from prismabuild import client
from _routed_gate_sdk_fixture import fixture
from test_routed_gate_staged_store import ROOT


def callback_module():
    path=ROOT/'experiments/t8r_speed/pb_staged_store.py'
    spec=importlib.util.spec_from_file_location('pb_native_control',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def owner_fixture(tmp_path,monkeypatch,*,expected=None,exec_hook=None):
    reader,staged,opened,sdk=fixture(tmp_path,origin='/forbidden-origin/module.so',offset=0)
    module=callback_module();calls=[]
    source=ROOT/'src/tessera/serving/csrc/routed_fused_window.cu'
    def build(name,source_name,compile_fn):
        calls.append((name,source_name))
        return compile_fn(str(source),'/normal-owner-build','sm_121',False)
    rf=SimpleNamespace(build_library=build,_ext=SimpleNamespace(cache_info=lambda:SimpleNamespace(currsize=0)))
    def make(spec):
        lib=ModuleType(spec.name);lib.__file__=spec.origin;lib.__spec__=spec
        return lib
    monkeypatch.setattr(module.importlib.util,'module_from_spec',make)
    monkeypatch.setattr(module.importlib.machinery.ExtensionFileLoader,'exec_module',
                        lambda self,lib: exec_hook(lib,staged) if exec_hook else None)
    owner=module.NativeCallback(reader,'/forbidden-origin/module.so',rf,tmp_path/'retained',
        expected_sha256=expected or hashlib.sha256(b'owned-wire').hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return owner,reader,staged,rf,build,calls


def test_retained_binary_uses_existing_owner_and_stays_held_until_fence(tmp_path,monkeypatch):
    owner,reader,staged,rf,original,calls=owner_fixture(tmp_path,monkeypatch)
    lib=rf.build_library(owner.MODULE,owner.MODULE,lambda *a:pytest.fail('must not JIT rebuild'))
    owner.bind(lib)
    assert calls==[(owner.MODULE,owner.MODULE)]
    assert lib.__file__==f'/proc/self/fd/{owner.fd}'
    fence=[]
    def fenced():
        assert os.fstat(owner.fd).st_size==10
        assert not reader.closed
        fence.append(True)
    owner.finish(fenced)
    assert fence==[True] and rf.build_library is original
    assert owner.record['before_load_sha256']==owner.record['after_load_sha256']==owner.record['after_profile_sha256']
    with pytest.raises(OSError):os.fstat(owner.fd)
    reader.close()


def test_wrong_native_digest_refuses_and_closes_without_callback_install(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='digest'):
        owner_fixture(tmp_path,monkeypatch,expected='0'*64)


def test_foreign_already_loaded_module_is_not_adopted(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'tessera_routed_fused_mma_e4m3',ModuleType('foreign'))
    with pytest.raises(ValueError,match='already loaded'):
        owner_fixture(tmp_path,monkeypatch)


@pytest.mark.parametrize('fault',['foreign_origin','file_change'])
def test_load_mutation_refuses_and_restores_callback(tmp_path,monkeypatch,fault):
    def corrupt(lib,staged):
        if fault=='foreign_origin':lib.__file__='/foreign/module.so'
        else:staged.write_bytes(b'other-wire')
    owner,reader,staged,rf,original,calls=owner_fixture(tmp_path,monkeypatch,exec_hook=corrupt)
    with pytest.raises(ValueError):
        rf.build_library(owner.MODULE,owner.MODULE,lambda *a:pytest.fail('JIT'))
    with pytest.raises(ValueError):owner.finish(lambda:None)
    assert owner.closed and rf.build_library is original
    with pytest.raises(OSError):os.fstat(owner.fd)
    reader.close()


def test_profile_mutation_refuses_after_fence_and_restores_callback(tmp_path,monkeypatch):
    owner,reader,staged,rf,original,calls=owner_fixture(tmp_path,monkeypatch)
    rf.build_library(owner.MODULE,owner.MODULE,lambda *a:pytest.fail('JIT'))
    staged.write_bytes(b'other-wire')
    fences=[]
    with pytest.raises(ValueError):owner.finish(lambda:fences.append(True))
    assert fences==[True] and owner.closed and rf.build_library is original
    reader.close()


def test_other_library_cannot_trigger_fallback_compile(tmp_path,monkeypatch):
    owner,reader,staged,rf,original,calls=owner_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='foreign'):
        rf.build_library('other','other',lambda *a:pytest.fail('JIT'))
    assert calls==[]
    owner.finish(lambda:None);reader.close()


