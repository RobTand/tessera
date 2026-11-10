"""Controlled native-owner fixture shared by reader and acceptor tests."""
import hashlib
import importlib.util
import json
from types import ModuleType, SimpleNamespace

from routed_gate_staged_store_support import ROOT


def callback_module():
    path=ROOT/'experiments/t8r_speed/declared_inputs.py'
    spec=importlib.util.spec_from_file_location('declared_native_control',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def owner_fixture(tmp_path,monkeypatch,*,expected=None,exec_hook=None,owner_args=None):
    module=callback_module();calls=[]
    staged=tmp_path/'original.so';staged.write_bytes(b'owned-wire')
    native_path=str(staged)
    manifest=tmp_path/'declared.json'
    manifest.write_text(json.dumps({'schema':'tessera.declared_inputs.v1','mount_prefix':str(tmp_path),'entries':[{'path':native_path,'offset':0,'bytes':10,
        'sha256':hashlib.sha256(b'owned-wire').hexdigest()}],'entry_count':1,'total_bytes':10}))
    reader=module.DeclaredInputs(manifest)
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
    owner=module.NativeCallback(reader,native_path,rf,tmp_path/'retained',
        expected_sha256=expected or hashlib.sha256(b'owned-wire').hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), **(owner_args or {}))
    return owner,reader,staged,rf,build,calls


