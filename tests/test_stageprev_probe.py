"""Closed #793 preparation controls. CPU fixtures do not qualify CUDA numerics."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def probe():
    spec = importlib.util.spec_from_file_location('stageprev_probe', ROOT/'experiments/t8r_speed/stageprev_probe.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def options():
    expected = json.loads((ROOT / "experiments/configs/stageprev_793_expected.json").read_text())
    argv = expected["launch"]["AB_BENCH_ARGS"]
    return SimpleNamespace(artifact=argv[argv.index("--artifact") + 1],
        groups='experts.R1024.L10,experts.R1088.L11,experts.R832.L42', ms='1,2048', no_graph=True,
        ncu=False, hash_only=True, input_manifest='sealed.json', profile_native_file='declared.so',
        single_routing_file=None, warmup=0, iters=0, power_s=0, routing=None)


def test_exact_original_three_group_numeric_mode_admits():
    probe().require_options(options(), stubbed=False)


@pytest.mark.parametrize('field,value', [('groups','experts.R1024.L10'), ('groups','all'),
    ('artifact','unqualified-current-model'), ('ms','512,2048'), ('no_graph',False), ('ncu',True),
    ('hash_only',False), ('input_manifest',None), ('profile_native_file',None), ('iters',30), ('power_s',3),
    ('routing','unbound-origin-directory')])
def test_numeric_mode_refuses_menu_or_timing_or_unbound_native(field, value):
    args=options(); setattr(args,field,value)
    with pytest.raises(ValueError,match='stageprev'):
        probe().require_options(args,stubbed=False)


def test_numeric_mode_refuses_stubbed_vllm():
    with pytest.raises(ValueError,match='stageprev'):
        probe().require_options(options(),stubbed=True)


@pytest.mark.parametrize('layer',[10,11,42])
def test_explicit_module_roster_binds_without_foreign_layer_alias(layer):
    spec=importlib.util.spec_from_file_location('pb_staged_store',ROOT/'experiments/t8r_speed/pb_staged_store.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reader=module.StagedInputs.__new__(module.StagedInputs)
    name=f'model.language_model.layers.{layer}.mlp.experts'
    roles=[{'expert':e,'role':r,'tensor':f'{name}.{e}.{r}.weight'}
           for e in range(288) for r in ('gate_proj','up_proj','down_proj')]
    reader.bind_roles('/unused',roles,module=name)
    assert len(reader.roles)==864
    roles[-1]=dict(roles[-1],tensor=f'{name}.0.gate_proj.weight')
    with pytest.raises(ValueError,match='roster'):
        reader.bind_roles('/unused',roles,module=name)


@pytest.mark.parametrize("failure", [False, True])
def test_observer_restores_actual_frozen_class_on_success_and_failure(tmp_path, monkeypatch, failure):
    """The observer sees both stages at the load-bound uniform launch, as the
    real uniform forward issues them, and reads run tables from the class
    projections."""
    torch = pytest.importorskip("torch")
    from dataclasses import MISSING, fields
    from tessera.routed_fused import FusedRoutedWindowMoE, _UniformProjection, _UniformWindowKernel
    runs = {"gate": torch.tensor([[4,0,1,0,0,0,0,0]]), "up": torch.tensor([[4,0,1,0,0,1,0,64]]),
            "down": torch.tensor([[4,0,1,0,0,0,0,0]])}
    kernel = _UniformWindowKernel(**{f.name: None for f in fields(_UniformWindowKernel)}
        | {role: _UniformProjection(None, None, None, table, None) for role, table in runs.items()})
    values = {f.name: None for f in fields(FusedRoutedWindowMoE) if f.default is MISSING}
    native = FusedRoutedWindowMoE(**values | {"library": "e4m3mma"}, uniform=kernel)
    def launch(self,mode,*args,**kwargs):
        kwargs["out"].fill_(mode+1)
        if failure:raise RuntimeError("owned launch failure")
    monkeypatch.setattr(_UniformWindowKernel,"launch",launch)
    monkeypatch.setattr(FusedRoutedWindowMoE,"resident_bytes",lambda self:64)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    def fn(*unused):
        for mode in (0,2):
            out=torch.empty(1,2,dtype=torch.bfloat16)
            native.uniform.launch(mode,out=out)
        return out.clone()
    fn.native_adapter=native
    xa = (torch.ones(1,2,dtype=torch.bfloat16), torch.zeros(1,8,dtype=torch.int32),
          torch.full((1,8),1/8,dtype=torch.float32))
    if failure:
        with pytest.raises(RuntimeError,match="owned launch failure"):
            probe().observe(fn,xa,tmp_path/"words")
    else:
        result=probe().observe(fn,xa,tmp_path/"words")
        assert set(result["outputs"])=={"gate_up","down_routes","out"}
        assert result["repeat_equal"] is True
        assert set(result["inputs"])=={"x","ids","weights"}
        assert result["inputs"]["x"]["shape"]==[1,2]
        assert result["run_tables"]=={"gate_proj":[[4,0,1,0,0,0,0,0]],
                                      "up_proj":[[4,0,1,0,0,1,0,64]],
                                      "down_proj":[[4,0,1,0,0,0,0,0]]}
        assert result["native_extra_resident_bytes"]==64
    assert _UniformWindowKernel.launch is launch


def test_observer_refuses_a_multi_class_owner(tmp_path):
    """The class dispatcher has no uniform launch to observe; refuse by name."""
    pytest.importorskip("torch")
    from dataclasses import MISSING, fields
    from tessera.routed_fused import FusedRoutedWindowMoE
    values = {f.name: None for f in fields(FusedRoutedWindowMoE) if f.default is MISSING}
    native = FusedRoutedWindowMoE(**values | {"library": "e4m3mma"})
    fn = lambda *unused: None
    fn.native_adapter = native
    with pytest.raises(ValueError, match="uniform"):
        probe().observe(fn, (None, None, None), tmp_path/"words")

