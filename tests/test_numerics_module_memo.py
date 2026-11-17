"""Check ordered aliases, table ownership and streamed role data."""
import itertools
import json

import pytest

from experiments.numerics_rows.batch_evaluator import BatchEvaluator
from experiments.numerics_rows.scale_index import ModuleMemo, evaluate, joint_coverage, make_plan, passes, scale_values


@pytest.mark.parametrize('vectorized', [False, True])
@pytest.mark.parametrize('arity', [1, 2])
@pytest.mark.parametrize('n', [1, 3, 8])
@pytest.mark.parametrize('g,u', [(0.0, 0.0), (1.0, 1.0), (1.0, -1.0)])
def test_module_aliases_match_exhaustive_fraction(arity, n, g, u, vectorized):
    coding = passes(n, -2, 1, compact=True)['coding']
    arrays = [scale_values(e) for e in coding]
    cases = [{'read': list(read), 'scale': 'row', 'g': g, 'u': u,
              'a_scale': 1.0, 'g_scale': 1.0, 'u_scale': 1.0}
             for read in itertools.product(range(n), repeat=arity)]
    class Table:
        limit = 10.0
        def __call__(self, value): return value
    table, expected = Table(), []
    for case in cases:
        own = [evaluate(case, case['read'], values, table) for values in arrays]
        collision = next((wrong for wrong in itertools.product(range(n), repeat=arity)
                          if list(wrong) != case['read'] and
                          all(evaluate(case, wrong, values, table) == value for values, value in zip(arrays, own))), None)
        if collision is not None: expected.append({'read': case['read'], 'aliases': list(collision)})
    memo = ModuleMemo(table, vectorized=vectorized)
    for case in cases:
        evaluator = memo.evaluator([case], coding, {})
        result = joint_coverage([case['read']], coding, evaluator, n, memo=memo)
        assert result['uncovered_reads'] == [e for e in expected if e['read'] == case['read']]
    assert memo.report()['group_misses'] == 1
    if g == 0 and len(cases) > 1: assert memo.report()['matches']['hits'] > 0


def test_module_evaluation_reuses_only_equal_operands():
    memo = ModuleMemo(None)
    coding = [{'values': [1.0, 2.0]}]
    for g in (1.0, 1.0, 2.0, -0.0, 0.0):
        case = {'read': [0], 'scale': 'row', 'g': g, 'a_scale': 1.0, 'g_scale': 7.0}
        evaluator = memo.evaluator([case], coding, {})
        assert evaluator((0,), (0,), [1.0]) == evaluate(case, (0,), [1.0])
    assert memo.report()['evaluate']['hits'] == 1
    assert memo.report()['evaluate']['misses'] == 4


def test_module_groups_bind_every_descriptor_field():
    memo = ModuleMemo(None)
    coding = passes(3, -2, 1, compact=True)['coding']
    memo.groups(coding, [scale_values(e) for e in coding])
    changed = [{**e, 'low': -1} for e in coding]
    memo.groups(changed, [scale_values(e) for e in changed])
    assert memo.report()['group_misses'] == 2


def test_streamed_manifest_keeps_role_data(tmp_path):
    from experiments.numerics_rows.module_io import RoleFile, write_manifest
    role = {'separation': {'masks': [1, 9]}, 'scale_index': [{'read_cases': [{'read': [1, 2]}]}]}
    path = tmp_path / 'case_manifest.json'
    write_manifest(path, {'roles': [RoleFile(tmp_path / 'role.pt', role)], 'generation_finished': True})
    assert json.loads(path.read_text()) == {'roles': [role], 'generation_finished': True}


def test_module_memo_refuses_another_table():
    with pytest.raises(ValueError, match='another measured D table'):
        make_plan('row', 2, [], memo=ModuleMemo(object()), table=object())


def test_module_memo_uses_batch_by_default():
    memo = ModuleMemo(None)
    assert isinstance(memo.evaluator([], [], None), BatchEvaluator)
