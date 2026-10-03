"""The existing measurement owner reports the conventional sample median."""
import ast
from pathlib import Path
import statistics

import pytest


def summarizer():
    path = Path(__file__).resolve().parents[1]/'experiments/t8r_speed/bench_t8r.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize')
    scope={'statistics':statistics}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['summarize']


@pytest.mark.parametrize('samples,expected', [([3,1,2],2),([4,1,3,2],2.5),([2,1],1.5)])
def test_true_odd_and_even_sample_median(samples,expected):
    result=summarizer()(samples)
    assert result['median_ms']==expected
    assert result['n']==len(samples)
    assert result['min_ms']==min(samples)


def test_timing_samples_remain_available_in_execution_order():
    raw=[4.0,1.0,3.0,2.0]
    result=summarizer()(raw)
    assert result['raw_samples_ms']==raw
    assert result['raw_samples_ms'] is not raw



def polling_owner():
    import threading
    import time
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1]/'experiments/t8r_speed/bench_t8r.py'
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='PowerSampler')
    threads = []
    def thread(**kwargs):
        item = threading.Thread(**kwargs)
        threads.append(item)
        return item
    scope = {'threading': SimpleNamespace(Event=threading.Event, Thread=thread), 'time': time}
    exec(compile(ast.Module(body=[owner], type_ignores=[]), str(path), 'exec'), scope)
    sampler = scope['PowerSampler'].__new__(scope['PowerSampler'])
    sampler.source = 'unit-fixture-not-device-evidence'
    sampler.observation = lambda **kw: {'unix': time.time(), 'sm_clock_mhz': 2000,
                                       'temperature_c': 60, 'throttle_reason_mask': 0}
    return sampler, threads


def test_repeatability_polling_is_explicitly_clock_unqualified_and_keeps_outliers():
    sampler, threads = polling_owner()
    raw = [20.0]*29+[2000.0]
    result, observations = sampler.observe_while(lambda: raw)
    assert result is raw
    assert len(observations['observations']) >= 2
    assert observations['requested_poll_interval_s'] == 0.1
    assert observations['clock_status'].startswith('unqualified_reported_host_polled')
    assert observations['thermal_policy'].endswith('no_result_based_filtering')
    assert not any(thread.is_alive() for thread in threads)
    assert summarizer()(result)['raw_samples_ms'] == raw


def test_repeatability_polling_owner_stops_on_work_failure():
    sampler, threads = polling_owner()
    def failed():
        raise RuntimeError('injected work failure')
    with pytest.raises(RuntimeError, match='injected work failure'):
        sampler.observe_while(failed)
    assert threads and not any(thread.is_alive() for thread in threads)


def test_repeatability_conditioning_uses_both_resident_arms_without_profiling():
    import time
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1]/'experiments/t8r_speed/bench_t8r.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name=='condition_repeatability')
    order = []
    arms = {arm: (lambda *xa, arm=arm: order.append(arm),) for arm in ('legacy', 'piece_major')}
    sampler, _ = polling_owner()
    scope = {'time': time, 'torch': SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    result = scope['condition_repeatability'](arms, (), sampler, 0.001)
    assert order and order == ['legacy','piece_major']*(len(order)//2)
    assert result['calls_by_arm']['legacy'] == result['calls_by_arm']['piece_major']
    assert result['seconds'] >= 0.001
    assert result['scope'].endswith('not thermal equilibration proof')



@pytest.mark.parametrize('failure_at', [1, 2])
def test_repeatability_retains_events_on_boundary_observation_failure(failure_at):
    sampler, threads = polling_owner()
    count = 0
    def observation(**kwargs):
        nonlocal count
        count += 1
        if count == failure_at:
            raise RuntimeError('injected boundary observer failure')
        return {'unix': 1.0, 'temperature_c': 60}
    sampler.observation = observation
    raw = [1.0]*30
    result, captured = sampler.observe_while(lambda: raw)
    assert result is raw
    assert any('injected boundary observer failure' in row.get('observation_error','')
               for row in captured['observations'])
    assert captured['clock_status'].startswith('unqualified')
    assert not any(thread.is_alive() for thread in threads)
