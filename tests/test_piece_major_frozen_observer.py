"""Exercise the PM observer against the actual frozen production adapter."""
import ast
from dataclasses import fields, MISSING
from pathlib import Path

import pytest


def observer():
    path = Path(__file__).resolve().parents[1] / 'experiments/t8r_speed/bench_t8r.py'
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == 'numeric_outputs')
    scope = {'bits': lambda t: t.detach().contiguous().view(__import__('torch').uint8).cpu()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['numeric_outputs']


@pytest.mark.parametrize('fault', [None, 'foreign', 'missing', 'duplicate', 'raised', 'family', 'classes'])
def test_actual_frozen_observer_captures_and_restores(monkeypatch, fault):
    """The observer sees every routed mode at the load-bound uniform launch.

    The forward reaches ``_UniformWindowKernel.launch`` directly, as the real
    uniform forward does; ``gate_up`` reaches it through the real
    ``FusedRoutedWindowMoE._launch``."""
    torch = pytest.importorskip('torch')
    from tessera import routed_fused as rf
    owner = rf.FusedRoutedWindowMoE
    kernel_type = rf._UniformWindowKernel

    def kernel():
        return kernel_type(**{f.name: None for f in fields(kernel_type)})
    values = {f.name: None for f in fields(owner) if f.default is MISSING}
    values['library'] = 'e4m3' if fault == 'family' else 'e4m3mma'
    native = owner(**values, uniform=None if fault == 'classes' else kernel())
    other = kernel()

    def cpu_launch(instance, mode, *args, **kwargs):
        kwargs['out'].fill_(mode + 1)
    monkeypatch.setattr(kernel_type, 'launch', cpu_launch)

    def gate_up(instance, *unused):
        out = torch.empty(2, 4, dtype=torch.bfloat16)
        instance._launch(1, None, None, None, a_row_mode=0, mul_weight=False, limit=0.0, out=out)
        return out
    monkeypatch.setattr(owner, 'gate_up', gate_up)

    def forward(*unused):
        if fault == 'foreign':
            other.launch(0, out=torch.empty(2, 4, dtype=torch.bfloat16))
        act = torch.empty(2, 4, dtype=torch.bfloat16)
        native.uniform.launch(0, out=act)
        if fault == 'raised':
            raise RuntimeError('injected forward failure')
        if fault == 'duplicate':
            native.uniform.launch(0, out=act)
        if fault != 'missing':
            native.uniform.launch(2, out=act)
        return act.clone()
    forward.native = native
    forward.gate_up = lambda *xa: native.gate_up(*xa)
    try:
        if fault in ('missing', 'duplicate', 'family', 'classes'):
            with pytest.raises(ValueError):
                observer()(forward, ())
        elif fault == 'raised':
            with pytest.raises(RuntimeError, match='injected'):
                observer()(forward, ())
        else:
            result = observer()(forward, ())
            assert set(result) == {'forward', 'mode0', 'mode1', 'mode2'}
            assert torch.equal(result['mode1'], torch.full((2, 4), 2, dtype=torch.bfloat16).view(torch.uint8))
    finally:
        assert kernel_type.launch is cpu_launch
        assert 'launch' not in vars(other)


@pytest.mark.parametrize('probe', ['bench', 'stageprev'])
def test_both_observers_capture_the_real_uniform_launches(tmp_path, probe):
    """The two numeric observers see every routed mode of a real uniform owner."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('the native routed lane is a CUDA path')
    import importlib.util
    import test_routed_fused_window as fixture

    stacks = fixture._stacks('e4m3', hidden=256, inter=128, experts=5, seed=880, cut=True)
    native = fixture._fused(fixture._bundles('e4m3', stacks))
    assert native.uniform is not None and native.library == 'e4m3mma'
    ids, weights = fixture._routes(64, 3, 881, experts=5)
    x = torch.randn((64, 256), generator=torch.Generator().manual_seed(882)).bfloat16().cuda()

    def fn(a, i, w):
        return native(a, i, w, swiglu_limit=10.0, apply_router_weight_on_input=False)
    fn.native = fn.native_adapter = native
    fn.gate_up = native.gate_up
    if probe == 'bench':
        result = observer()(fn, (x, ids, weights))
        assert set(result) == {'forward', 'mode0', 'mode1', 'mode2'}
        assert torch.equal(result['forward'], fn(x, ids, weights).view(torch.uint8).cpu())
    else:
        path = Path(__file__).resolve().parents[1] / 'experiments/t8r_speed/stageprev_probe.py'
        spec = importlib.util.spec_from_file_location('stageprev_probe', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.observe(fn, (x, ids, weights), tmp_path / 'words')
        assert result['repeat_equal'] is True
        assert set(result['outputs']) == {'gate_up', 'down_routes', 'out'}
        assert set(result['run_tables']) == {'gate_proj', 'up_proj', 'down_proj'}
