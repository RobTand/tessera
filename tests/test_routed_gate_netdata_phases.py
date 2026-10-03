"""The existing telemetry owner must retain every arm's measured power window."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace


def collect_fixture(tmp_path, population):
    source = Path(__file__).resolve().parents[1] / 'experiments/t8r_speed/routed_gate_netdata.py'
    functions = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in ('retain_query','main')]
    owner = SimpleNamespace(collect=lambda address, a, z, points: {'window': [a, z]},
                            _fetch=lambda *a: {'url': 'inert', 'doc': {}},
                            _stats=lambda *a: {})
    util = SimpleNamespace(spec_from_file_location=lambda *a: SimpleNamespace(
                               loader=SimpleNamespace(exec_module=lambda module: None)),
                           module_from_spec=lambda spec: owner)
    scope = dict(Path=Path, json=json, math=math,
                 sys=SimpleNamespace(argv=['collector', str(tmp_path)]),
                 importlib=SimpleNamespace(util=util), __file__=str(source))
    (tmp_path / 'action-window.txt').write_text('0 40')
    timing = tmp_path / 'timing'
    timing.mkdir()
    (timing / 'bench_t8r.json').write_text(json.dumps(population))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), scope)
    scope['main']()
    return json.loads((tmp_path / 'netdata.json').read_text())


def test_every_comparison_arm_keeps_both_box_windows(tmp_path):
    report = collect_fixture(tmp_path, {'results': [{
        'group': 'experts.R1024.L10', 'cells': {
            '0:legacy:M1': {'power': {'window_unix': [10, 20]}},
            '1:piece_major:M1': {'power': {'window_unix': [21, 31]}},
        }}]})
    windows = {tuple(phase['window_unix']) for phase in report['phases'].values()}
    assert {(10, 20), (21, 31)} <= windows
    assert all(set(phase['boxes']) == {'sparky', 'sparklina'} for phase in report['phases'].values())
    assert report['energy_status'].startswith('HOLD')



def test_repeatability_retains_conditioning_events_and_raw_supplemental_queries(tmp_path):
    report = collect_fixture(tmp_path, {'results': [{
        'M': 2048, 'conditioning': {'window_unix': [1,61]},
        'cells': {'0:legacy:M2048': {'wall_window_unix': [62,62.5]},
                  '1:piece_major:M2048': {'wall_window_unix': [63,63.5]}}}]})
    assert 'conditioning:0' in report['phases']
    assert 'events:0:0:legacy:M2048' in report['phases']
    assert 'events:0:1:piece_major:M2048' in report['phases']
    for phase in report['phases'].values():
        for box in phase['boxes'].values():
            for context in ('system.io','nvidia_smi.gpu_clock_freq','nvidia_smi.gpu_temperature'):
                assert 'raw_response' in box[context]
                assert box[context]['coverage'] is None  # Inert unit input cannot prove coverage.
    assert report['energy_status'].startswith('HOLD')
