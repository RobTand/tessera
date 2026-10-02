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
