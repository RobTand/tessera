"""Replay the v68 stub receipts without expanding their measured scope."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tessera.serving.contract import CENSUS_PHASE_REGIMES, PAYLOAD_FAMILY_BY_ROUTE, load_serving_contract, validate_serving_contract

ROOT = Path(__file__).resolve().parents[1]
SCOPE = json.loads((ROOT / 'experiments/results/glm53_requal_v68_scope.json').read_text())
SPEC = importlib.util.spec_from_file_location('requal_census', ROOT / 'tools/tessera_route_census.py')
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def inputs(key):
    row = SCOPE['receipts'][key]
    receipt = json.loads((ROOT / row['receipt']).read_text())
    config = json.loads((ROOT / row['config']).read_text())
    groups = config['quantization_config']['config_groups']
    rungs = {receipt['declared_name_mapping'].get(t, t): TOOL.declared_rung(g['scheme'])
             for g in groups.values() for t in g['targets']}
    return row, receipt, config, rungs


def agreement(receipt, rungs, contract):
    return TOOL.all_structure_agreement(
        receipt['records'], cells=contract['lane_eligibility']['cells'],
        phase_regimes=CENSUS_PHASE_REGIMES, platform='sm_121', declared_rungs=rungs,
        record_owners=receipt['record_owner'], families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        runtime_image=receipt['runtime']['image'], execution_mode='eager')


@pytest.mark.parametrize('key,count', [('e2m1', 3), ('bf16', 9)])
def test_receipt_identity_and_complete_replay(key, count):
    row, receipt, config, rungs = inputs(key)
    assert row['sha256'] == {'e2m1': '4b81ea19e436f16cac836e0fa4ca09bc84190822f786b0a4a1e5b3a2bb3f033c', 'bf16': 'bb409f5fb9182f86d257f26f488038e96afe810a2d8fb9fda2f918b2b2de951f'}[key]
    assert hashlib.sha256((ROOT / row['receipt']).read_bytes()).hexdigest() == row['sha256']
    assert hashlib.sha256((ROOT / row['config']).read_bytes()).hexdigest() == row['config_sha256']
    assert receipt['checkpoint_sidecars']['config.json'] == row['config_sha256']
    assert receipt['versions']['tessera_commit'] == SCOPE['commit']
    assert receipt['verdict'] == 'served' and receipt['problems'] == []
    assert config['text_config']['num_hidden_layers'] == (4 if key == 'e2m1' else 8)
    assert receipt['env']['TESSERA_RESEARCH_GLM53_NOPE'] == ('1' if key == 'e2m1' else '0')
    contract = load_serving_contract()
    block, problems = agreement(receipt, rungs, contract)
    assert problems == [] and block['agrees'] is True
    cells = {c['id']: c for c in contract['lane_eligibility']['cells']}
    joined = set()
    for per in block['structures'].values():
        for phase, result in per['phases'].items():
            assert result['covered_by_cell'] == result['modules'] > 0
            assert result['unattested'] == 0
            joined.update(result['cells'])
    assert len(joined) == 4
    assert all(len(records) == count for records in receipt['records'].values())
    for cell_id in joined:
        cell = cells[cell_id]
        assert cell['runtime']['image'] == receipt['runtime']['image']
        assert cell['runtime']['vllm'] == receipt['versions']['vllm']
        assert cell['runtime']['torch'] == receipt['versions']['torch']
        assert cell['evidence']['grade'] == 'route_only'
        assert cell['requires_serve_flags'] == ['TESSERA_SERVE_MODE=resident']
    without = copy.deepcopy(contract)
    without['lane_eligibility']['cells'] = [c for c in without['lane_eligibility']['cells'] if c['id'] not in joined]
    empty, _ = agreement(receipt, rungs, without)
    assert all(r['covered_by_cell'] == 0 for p in empty['structures'].values() for r in p['phases'].values())
    damaged = copy.deepcopy(receipt)
    first = next(iter(damaged['records']['decode'].values()))
    first['decoder'] = 'native_span2_grouped' if key == 'e2m1' else 'native_routed_fused_window_folded'
    assert agreement(damaged, rungs, contract)[1]
    wrong_image = copy.deepcopy(receipt)
    wrong_image['runtime']['image'] = 'example.invalid/other@sha256:' + '0' * 64
    unseen, _ = agreement(wrong_image, rungs, contract)
    assert all(r['covered_by_cell'] == 0 for p in unseen['structures'].values() for r in p['phases'].values())


@pytest.mark.parametrize('field,value', [('regime', 'prefill'), ('executes', []), ('rungs_q256', [640]),
                                         ('executes', [{'symbol': 'torch.mm', 'decoder': 'torch_window'}]),
                                         ('execution_modes', ['eager', 'compiled'])])
def test_malformed_or_unmeasured_scope_refuses(field, value):
    contract = load_serving_contract()
    cell = next(c for c in contract['lane_eligibility']['cells'] if c['id'] == 'tessera_e2m1_k2_dense_sm121_decode_resident')
    if field == 'execution_modes':
        cell['runtime'][field] = value
    elif field == 'executes' and value:
        cell[field] += value
    else:
        cell[field] = value
    with pytest.raises(ValueError):
        validate_serving_contract(contract)


def test_img_n_cells_refuse_old_folded_records_on_the_same_image():
    _, receipt, _, rungs = inputs('bf16')
    old = json.loads((ROOT / 'experiments/results/glm53_u1_stub_b_nightly_tp1_eager_census.json').read_text())
    assert old['runtime']['image'] == receipt['runtime']['image']
    records = {phase: {name: record for name, record in rows.items()
                        if record['policy'].startswith('TESSERA_BF16:')}
               for phase, rows in old['records'].items()}
    assert all(records.values())
    old['records'] = records
    assert agreement(old, rungs, load_serving_contract())[1]
