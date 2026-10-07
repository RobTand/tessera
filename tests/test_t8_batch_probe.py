"""The batch probe preserves explicit scope, source bytes and measured populations.

The tests use CPU control flow. They do not qualify GPU execution.
"""

import json
import os
import sys
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[1]
SRC = CHECKOUT / "src"
EXPERIMENTS = CHECKOUT / "experiments"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "t8_census"))


STAGE_WRAPPER = EXPERIMENTS / "t8_census" / "ab_stage.sh"


@pytest.fixture
def torch_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")


@pytest.mark.usefixtures("torch_runtime")
class TestStagedRungScope:
    def test_rungs_are_explicit_never_historical(self):
        import ab_batched_best_form as ab
        parser = ab.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["/tmp/out"])
        args = parser.parse_args(["/tmp/out", "--rungs", "E4M3:768,E2M1:640",
                                  "--hessian", "/tmp/h.json",
                                  "--producer-authority", "/tmp/a.py"])
        assert args.rungs == "E4M3:768,E2M1:640"

    def test_rung_spec_parses_grid_and_rung(self):
        import ab_batched_best_form as ab
        assert ab.parse_rungs("768", grid="E4M3") == [("E4M3", 768)]
        assert ab.parse_rungs("E4M3:768,E2M1:640", grid=None) == [("E4M3", 768), ("E2M1", 640)]
        with pytest.raises(SystemExit):
            ab.parse_rungs("768", grid=None)
        with pytest.raises(SystemExit):
            ab.parse_rungs("E9M9:768", grid=None)


@pytest.mark.usefixtures("torch_runtime")
class TestBudgetGuard:
    def test_whole_action_budget_stops_with_named_unmeasured(self):
        import ab_batched_best_form as ab
        stages = [
            {"stage": "rung0/E4M3:768/seq_front", "units": 12, "s_per_unit_prior": 15.2},
            {"stage": "rung0/E4M3:768/bat_best", "units": 12, "s_per_unit_prior": 15.2},
            {"stage": "rung0/E4M3:768/bat_front", "units": 4, "s_per_unit_prior": 15.2},
            {"stage": "rung0/E4M3:768/seq_best", "units": 4, "s_per_unit_prior": 15.2},
            {"stage": "rung1/E2M1:640/seq_front", "units": 4, "s_per_unit_prior": 15.2},
            {"stage": "rung1/E2M1:640/bat_best", "units": 4, "s_per_unit_prior": 15.2},
        ]
        plan = ab.plan_stages(stages, budget_s=2700.0, elapsed_s=120.0)
        assert [d["decision"] for d in plan["stages"]].count("skip") == 0
        assert plan["projected_total_s"] <= 2700.0
        assert not plan["unmeasured"]
        # A tight budget drops later stages into `unmeasured`, by name, and
        # never silently extends.
        tight = ab.plan_stages(stages, budget_s=600.0, elapsed_s=120.0)
        assert tight["stages"][0]["decision"] == "run"
        assert any(d["decision"] == "skip" for d in tight["stages"])
        assert tight["unmeasured"] == [d["stage"] for d in tight["stages"] if d["decision"] == "skip"]
        assert all("projected_s" in d for d in tight["stages"])

    def test_extrapolations_name_both_stacks_distinctly(self):
        import ab_batched_best_form as ab
        basis = {"grid": "E4M3", "q256": 768, "layer": 3, "batch": 4,
                 "shape_costs": {"2048x4096": 15.2, "4096x2048": 15.2},
                 "shape_population": {"2048x4096": 576, "4096x2048": 288}}
        ext = ab.extrapolations(15.2, projections_measured=("gate_proj", "up_proj", "down_proj"),
                                basis=basis)
        assert ext["projection_equivalent_288_unit_stack_estimate_min"] == round(15.2 * 288 / 60, 3)
        complete = ext["complete_stack"]["complete_864_unit_stack_estimate_min"]
        assert complete == round(15.2 * 864 / 60, 3)
        assert complete != ext["projection_equivalent_288_unit_stack_estimate_min"]
        assert ext["basis"] == basis
        assert "not a measured full export" in ext["complete_stack"]["kind"]

    def test_incomplete_projection_keeps_complete_stack_unmeasured(self):
        import ab_batched_best_form as ab
        ext = ab.extrapolations(15.2, projections_measured=("gate_proj",),
                                basis={"grid": "E4M3", "q256": 768, "layer": 3, "batch": 4})
        assert "unmeasured" in ext["complete_stack"]
        assert "gate_proj" in ext["complete_stack"]["unmeasured"]
        assert "complete_864_unit_stack_estimate_min" not in ext["complete_stack"]

    def test_truncated_prefix_is_weighted_by_full_shape_population(self):
        import ab_batched_best_form as ab
        basis = {"shape_costs": {"gate_up": 10.0, "down": 30.0},
                 "shape_population": {"gate_up": 576, "down": 288}}
        ext = ab.extrapolations(15.2, projections_measured=("gate_proj", "up_proj", "down_proj"),
                                basis=basis)
        assert ext["complete_stack"]["complete_864_unit_stack_estimate_min"] == 240.0
        assert ext["projection_equivalent_288_unit_stack_estimate_min"] == 80.0

    def test_missing_shape_cost_never_claims_a_complete_stack(self):
        import ab_batched_best_form as ab
        ext = ab.extrapolations(10.0, projections_measured=("gate_proj", "up_proj", "down_proj"),
                                basis={"shape_costs": {"gate_up": 10.0},
                                       "shape_population": {"gate_up": 576, "down": 288}})
        assert "unmeasured" in ext["complete_stack"]
        assert "down" in ext["complete_stack"]["unmeasured"]
        assert "complete_864_unit_stack_estimate_min" not in ext["complete_stack"]






@pytest.mark.usefixtures("torch_runtime")
class TestPowerReceipt:
    def test_work_per_joule_derives_observed_joules(self):
        import ab_batched_best_form as ab
        info = ab.leg_evidence(s_per_unit=15.2, units=12, wall_s=182.4,
                               power={"joules": 21888.0, "mean_w": 120.0, "samples": 100})
        assert info["joules_observed"] == 21888.0
        assert info["units_per_kJ"] == round(12 * 1000.0 / 21888.0, 3)

    def test_no_joules_no_work_per_joule(self):
        import ab_batched_best_form as ab
        info = ab.leg_evidence(s_per_unit=15.2, units=12, wall_s=182.4,
                               power={"samples": 0})
        assert info["units_per_kJ"] is None
        assert info["joules_observed"] is None


@pytest.mark.usefixtures("torch_runtime")
class TestSourceBinding:
    def test_shards_for_reads_only_the_shards_the_units_live_on(self, tmp_path):
        import profile_unit_encode as pue
        index = {"weight_map": {
            "model.language_model.layers.3.mlp.experts.112.gate_proj.weight": "model-00002-of-00019.safetensors",
            "model.language_model.layers.3.mlp.experts.112.up_proj.weight": "model-00002-of-00019.safetensors",
            "model.language_model.layers.3.mlp.experts.113.down_proj.weight": "model-00005-of-00019.safetensors",
        }}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        shards = pue.shards_for(tmp_path, [
            "model.language_model.layers.3.mlp.experts.112.gate_proj.weight",
            "model.language_model.layers.3.mlp.experts.112.up_proj.weight",
            "model.language_model.layers.3.mlp.experts.113.down_proj.weight",
        ])
        assert shards == ["model-00002-of-00019.safetensors", "model-00005-of-00019.safetensors"]

    def test_shards_for_refuses_units_outside_the_index(self, tmp_path):
        import profile_unit_encode as pue
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        with pytest.raises(KeyError):
            pue.shards_for(tmp_path, ["model.language_model.layers.3.mlp.experts.0.gate_proj.weight"])




def test_exhausted_deadline_never_starts_correctness_packet(tmp_path):
    import subprocess
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ticks = tmp_path / "ticks"
    date = fake_bin / "date"
    date.write_text(f"#!{sys.executable}\n" +
        "import pathlib, sys\n" +
        f"p=pathlib.Path({str(ticks)!r})\n" +
        "if sys.argv[1:] == ['+%s']:\n" +
        "    n=int(p.read_text()) if p.exists() else 0\n" +
        "    p.write_text(str(n+1)); print(100 if n == 0 else 101)\n" +
        "else: print('2026-10-05T00:00:00Z')\n")
    date.chmod(0o755)
    called = tmp_path / "correctness-called"
    python = fake_bin / "python"
    python.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(called)!r}).write_text('called')\n")
    python.chmod(0o755)
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ['PATH'], PYTHON=str(python),
               TESSERA_PRODUCER_PYTHON=str(python), TESSERA_PRODUCER_SOURCE=str(tmp_path))
    done = subprocess.run(['bash', str(STAGE_WRAPPER), str(tmp_path/'out'), '1', '--', 'fixture-test', '--'],
                          env=env, capture_output=True, text=True, timeout=30)
    assert done.returncode == 124, done.stdout + done.stderr
    assert not called.exists(), "an exhausted budget was passed as GNU timeout0 and started correctness"


@pytest.mark.parametrize("mode", ["skipped_anchor", "tail", "shared_split", "timed_mismatch", "anchor_mismatch", "effective_width_mismatch", "profiles"])
def test_unanchored_shape_cannot_enter_timed_population(tmp_path, monkeypatch, mode, torch_runtime):
    sys.path.insert(0, str(CHECKOUT / 'experiments' / 't8_census'))
    import ab_batched_best_form as ab
    from tessera import export_serving, export
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    stack = 'model.language_model.layers.3.mlp.experts'
    population = 2 if mode == 'skipped_anchor' else 5 if mode == 'tail' else 8
    units = [{'stack': stack, 'rows': 2, 'cols': 2, 'projection': 'gate_proj',
              'tensor': f'{stack}.{i}.gate_proj.weight'} for i in range(population)]
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {u['tensor']: 'source.safetensors' for u in units}}))
    monkeypatch.setattr(export_serving, 'authenticate_producer_python', lambda: {'qualified': True})
    monkeypatch.setattr(ab.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(ab.torch.cuda, 'get_device_name', lambda _: 'CPU control-flow fixture')
    monkeypatch.setattr(ab, 'load_producer_authority', lambda _: (None, None))
    monkeypatch.setattr(export.ActivationSource, 'from_capture', lambda *a, **k: None)
    monkeypatch.setattr(ab, 'quantizable', lambda _: (None, None, None, {}))
    monkeypatch.setattr(ab, 'expert_stacks', lambda _: {stack: [0, 1]})
    monkeypatch.setattr(ab, 'plan_expert_stack', lambda *a, **k: {'units': units})
    monkeypatch.setattr(ab, 'grid_for', lambda _: 'grid')
    monkeypatch.setattr(ab, 'served_recipe', lambda *a: None)
    monkeypatch.setattr(ab, 'bind_source', lambda *a: {'shards': [], 'digest_s': 0, 'cache': {},
        'cached_shards': [], 'hashed_shards': [], 'receipt': {}, 'identity': {'files': {'source.safetensors': 'a'*64}}})
    clock = [0.0]
    monkeypatch.setattr(ab.time, 'perf_counter', lambda: clock[0])
    if mode == 'profiles':
        monkeypatch.setattr(ab, 'COARSE_PRIOR_S_PER_UNIT', 0.05)
    events = []
    anchors = []
    def anchor(positions, *a, **k):
        anchors.extend(positions)
        events.append(('anchor', list(positions)))
        if mode == 'profiles': clock[0] += 1
        return {'digests': {units[i]['tensor']: ('b' if mode == 'anchor_mismatch' else 'a')*64 for i in positions}, 'units': len(positions),
                'wall_s': 1, 's_per_unit': 0.5, 'start_utc': 'fixture', 'end_utc': 'fixture', 'power': {}}
    monkeypatch.setattr(ab, 'anchor_units', anchor)
    calls = []
    def run(positions, *a, **k):
        calls.append(positions)
        events.append(('batch', a[-1]))
        if mode == 'profiles': clock[0] += 1
        digests = {units[i]['tensor']: 'a'*64 for i in positions}
        if mode == 'timed_mismatch' and a[-1].endswith('-b000'):
            digests[units[positions[0]]['tensor']] = 'b'*64
        widths = ([2, 2] if positions[0] == 0 else [1, 3]) if mode == 'shared_split' else [len(positions)]
        if mode == 'effective_width_mismatch' and a[-1].endswith('-b000'): widths = [1, 3]
        return {'positions': positions, 'units': len(positions), 'key': '2x2', 'widths_observed': widths,
                'digests': digests, 'wall_s': 1, 's_per_unit': 0.5, 'power': {},
                'start_utc': 'fixture', 'end_utc': 'fixture', 'workload': 'encode+frame+verify'}
    monkeypatch.setattr(ab, 'run_owner_batch', run)
    class CountFixture:
        def __enter__(self): events.append(('count', None)); return self
        def __exit__(self, *args): return False
        def record(self): return {'fixture': 'CPU control flow, no device profiling'}
    def profile_fixture(fn, out, label):
        events.append(('profile', label))
        result = fn()
        if mode == 'profiles': clock[0] += 5
        return {'fixture': 'CPU control flow, no CUDA profile', 'label': label}
    monkeypatch.setattr(ab, 'Count', CountFixture)
    monkeypatch.setattr(ab, 'cuda_profile', profile_fixture)
    batch, budget = ('2', '35') if mode == 'skipped_anchor' else ('4', '30' if mode == 'profiles' else '1000')
    monkeypatch.setattr(sys, 'argv', ['probe', str(tmp_path / 'out'), '--src', str(source), '--rungs', 'E4M3:768',
        '--hessian', 'h', '--producer-authority', 'a', '--batch', batch, '--budget-s', budget, *([] if mode == 'profiles' else ['--no-profile'])])
    result = ab.main()
    packet = json.loads((tmp_path / 'out' / 'ab_batched_best_form.json').read_text())
    row = packet['rungs']['E4M3:768']
    if mode == 'skipped_anchor':
        assert not calls, 'skipped warm+anchor cohort still executed a timed batch'
        assert not row['batches']
        assert result != 0, 'unqualified population must not be reported as accepted'
    elif mode in ('timed_mismatch', 'anchor_mismatch', 'effective_width_mismatch'):
        assert result == 4, 'timed digests were discarded instead of compared'
        assert not row['batches']
    else:
        if mode == 'profiles':
            assert row['batches'], 'duplicate profiles starved the admitted timed population'
            first_timed = next(i for i, event in enumerate(events) if event[0] == 'batch' and '-b000' in event[1])
            assert all(i < first_timed for i, event in enumerate(events) if event[0] == 'profile')
            assert len(row['captures']) == 1, 'identical shape/effective-width cohorts need one before/after profile'
        assert set(anchors) == set(range(population)), 'tail/shared schedule members were not independently anchored'
        assert row['batches'] and all('digests' in b for b in row['batches']), 'timed digests must be retained'




def test_actual_cpu_batch_and_anchor_keep_reads_outside_both_timers(tmp_path, monkeypatch, torch_runtime):
    import ab_batched_best_form as ab
    import test_export_serving as fixture
    from tessera import export as encoder
    from tessera import export_serving as owner
    from tessera.structure import STRUCTURE_ROUTED_MOE
    source = fixture._write(tmp_path, fixture._checkpoint(), fixture._config())
    tensors = fixture._checkpoint()
    (source / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {name: 'model.safetensors' for name in tensors}}))
    _shards, _dense, _packed, routed = owner.quantizable(source)
    stack = fixture.STACK
    grid, rung = owner.grid_for('E4M3'), 896
    plan = owner.plan_expert_stack(stack, owner.expert_stacks(routed)[stack], grid, rung, config=fixture._config())
    units = owner.expert_work_units(stack, plan)
    keys = [(u['stack'], u['rows'], u['cols']) for u in units]
    positions = next(p for p in owner.plan_joined_encodes(keys, 2) if len(p) == 2)
    recipe = owner.served_recipe(grid, rung, STRUCTURE_ROUTED_MOE)
    clock = [0.0]
    real_read, real_batch, real_one = ab.read_tensor, ab.fresh_joined_encode, encoder.encode_linear_planes
    reads = []
    def read_once(*args):
        value = real_read(*args)
        reads.append(args[1]); clock[0] += 100
        return value
    def batch_once(*args, **kwargs):
        value = real_batch(*args, **kwargs)
        clock[0] += 7
        return value
    def one_once(*args, **kwargs):
        value = real_one(*args, **kwargs)
        clock[0] += 7 / len(positions)
        return value
    class NoDevicePower:
        def start(self): pass
        def stop(self, **kwargs): return {'samples': 0, 'fixture': 'CPU, no GPU power'}
    monkeypatch.setattr(ab, 'read_tensor', read_once)
    monkeypatch.setattr(ab, 'fresh_joined_encode', batch_once)
    monkeypatch.setattr(encoder, 'encode_linear_planes', one_once)
    monkeypatch.setattr(ab, 'Power', NoDevicePower)
    monkeypatch.setattr(ab.time, 'perf_counter', lambda: clock[0])
    before_batch = clock[0]
    batch = ab.run_owner_batch(positions, units, {stack: plan}, recipe, None, source, tmp_path, 'cpu-batch', device='cpu')
    batch_work = clock[0] - before_batch - 100 * len(positions)
    before_anchor = clock[0]
    anchor = ab.anchor_units(positions, units, source, grid, rung, recipe, None, tmp_path, 'cpu-anchor', device='cpu')
    anchor_work = clock[0] - before_anchor - 100 * len(positions)
    assert batch['digests'] == anchor['digests']
    # Batch verification may invoke sequential references internally. Its
    # work cost need not equal the anchor: neither timer may price source IO.
    assert batch['wall_s'] == batch_work
    assert anchor['wall_s'] == anchor_work == 7
    assert len(reads) == 2 * len(positions)
    assert batch['widths_observed'] == [len(positions)]
    assert batch['power']['samples'] == anchor['power']['samples'] == 0




def test_probe_actual_plan_uses_the_exporter_worklist(tmp_path, monkeypatch, torch_runtime):
    import ab_batched_best_form as ab
    import test_export_serving as fixture
    from tessera import export, serving_parts
    source = fixture._write(tmp_path, fixture._checkpoint(), fixture._config())
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        name: 'model.safetensors' for name in fixture._checkpoint()}}))
    from tessera import export_serving
    monkeypatch.setattr(export_serving, 'authenticate_producer_python', lambda: {'fixture': 'CPU control-flow only'})
    monkeypatch.setattr(ab.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(ab.torch.cuda, 'get_device_name', lambda _: 'CPU control-flow fixture')
    monkeypatch.setattr(ab, 'load_producer_authority', lambda _: (None, None))
    monkeypatch.setattr(export.ActivationSource, 'from_capture', lambda *a, **k: None)
    identity = serving_parts.source_part_identity(source)
    monkeypatch.setattr(ab, 'bind_source', lambda *a: {'shards': list(identity['files']), 'digest_s': 0,
        'cache': {}, 'cached_shards': [], 'hashed_shards': list(identity['files']), 'receipt': {}, 'identity': identity})
    monkeypatch.setattr(ab.time, 'perf_counter', lambda: 0)
    monkeypatch.setattr(sys, 'argv', ['probe', str(tmp_path/'out'), '--src', str(source), '--layer', '1',
        '--rungs', 'E4M3:896', '--hessian', 'fixture', '--producer-authority', 'fixture',
        '--batch', '2', '--budget-s', '1', '--no-profile'])
    assert ab.main() == 3
    row = json.loads((tmp_path/'out'/'ab_batched_best_form.json').read_text())['rungs']['E4M3:896']
    assert row['membership']['units'] == fixture.EXPERTS * 3
    assert row['schedule']['units'] == fixture.EXPERTS * 3
    assert not row['batches']
