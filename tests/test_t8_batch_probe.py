"""The t8 ABBA batch probe's own contract: staged scope, budget truth, binding.

This file pins what the PROBE owns.  The producer-interpreter authentication
it calls is core-owned (``tessera.export_serving.authenticate_producer_python``,
maintained by the producer-selector core sibling): here it is pinned only as a
call contract -- the probe calls it, with no arguments, before anything
encodes -- never re-tested; its refusals and receipt are the core sibling's
tests.  Everything here is CPU-only: the probe's GPU legs are exercised by the
probe itself on the bounded announced stage, never by these tests.

Pinned here:
* the probe files' import arrangement: Tessera comes from the selected
  interpreter's installed package, never by inserting a checkout ``src`` tree
  ahead of it, and the authentication happens before the first encode;
* the staged rung scope: rungs are explicit dynamic args naming the restored
  plan's rungs (GA L3 E4M3@768, L4 BF16@960, L5 BF16@1088, L6 BF16@1152; GB
  L3 E2M1@640, L4 E2M1@768); the historical 1280/1408 default is gone and
  must not come back;
* truthful measurement scope: the whole-action budget guard stops with an
  exact partial receipt (unmeasured stages named, never silently dropped);
* the two stack extrapolations, distinct, whole only when all three
  projections were measured;
* the power receipt: work/J derives OBSERVED joules (the leg's own timestamped
  NVML integral), never a fabricated mean-times-time summary;
* the source binding: exactly the shards the selected units live on, through
  the source's own stable digest cache -- never a second cache.
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

#: The restored plan rungs the probe's staged launch names (campaign d19,
#: tessera#689).  Pinned here because the probe's help refuses to guess them.
RESTORED_RUNG_TEXT = ("GA layer3 E4M3@768, layer4 BF16@960, layer5 BF16@1088, "
                      "layer6 BF16@1152; GB layer3 E2M1@640, layer4 E2M1@768")

PROBE = EXPERIMENTS / "t8_census" / "ab_batched_best_form.py"
UNIT_PROFILE = EXPERIMENTS / "t8_census" / "profile_unit_encode.py"
STAGE_WRAPPER = EXPERIMENTS / "t8_census" / "ab_stage.sh"


@pytest.fixture
def torch_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")


class TestAuthCallContract:
    """The core sibling owns the function; the probe owns the call."""

    def test_probe_authenticates_before_any_encode(self):
        text = PROBE.read_text()
        assert "authenticate_producer_python()" in text, (
            "the probe must authenticate the selected interpreter's installed "
            "tessera via the core-owned call before anything encodes")
        assert "if auth is None:" in text, (
            "an unauthenticated run (TESSERA_PRODUCER_PYTHON unset) encodes nothing")
        assert text.index("authenticate_producer_python()") < text.index("def run_owner_batch")

    def test_probe_drives_the_common_owner_on_original_membership(self):
        """The timed work is the common fresh encode/finish owner; no second
        implementation, no derived source, no renamed keys."""
        text = PROBE.read_text()
        assert "encode_linears_planes(" not in text, (
            "the probe never calls the joined encoder itself; the owner "
            "(fresh_joined_encode) is its only caller")
        for owner_piece in ("fresh_joined_encode", "plan_joined_encodes",
                            "plan_expert_stack", "expert_stacks", "quantizable"):
            assert owner_piece in text, f"the probe must drive the owner piece {owner_piece}"
        assert "encode_linear_planes(" in text, (
            "the seq anchor is the exporter's own per-unit finish")



    def test_auth_call_takes_no_arguments(self, torch_runtime):
        """Env-selected source, no flags: the core owns the semantics.

        PREREQUIREMENT REGRESSION (retained red until the core sibling's
        commit lands, PB action 67150cddfc57...): this imports the core-owned
        function; its absence fails HERE and only here, never the probe's own
        contract tests below.
        """
        import inspect
        from tessera.export_serving import authenticate_producer_python
        params = [p for p in inspect.signature(authenticate_producer_python).parameters.values()
                  if p.default is inspect.Parameter.empty]
        assert params == [], (
            "the probe's call passes nothing; required parameters would make the "
            f"probe own authentication semantics it does not own: {params}")


class TestProbeImportArrangement:
    def test_probe_files_never_insert_a_src_tree(self):
        """The probe imports Tessera from the interpreter's install, period."""
        for path in (PROBE, UNIT_PROFILE):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if "sys.path" in line:
                    assert '"src"' not in line and "'src'" not in line, (
                        f"{path.name}:{lineno}: {line.strip()}")


@pytest.mark.usefixtures("torch_runtime")
class TestStagedRungScope:
    def test_rungs_are_explicit_never_historical(self):
        import ab_batched_best_form as ab
        parser = ab.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["/tmp/out"])
        help_text = " ".join(parser.format_help().split())
        assert "768" in help_text and "640" in help_text
        assert RESTORED_RUNG_TEXT in help_text
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

    def test_default_cache_is_the_existing_stable_cache(self):
        """One cache: the stubs source's own source-digests, never a second."""
        import profile_unit_encode as pue
        assert pue.DEFAULT_DIGEST_CACHE == pue.SRC.parent / "source-digests"


class TestStagedAction:
    """The one bounded action: core's CUDA correctness packet, then the probe."""

    def test_wrapper_runs_correctness_before_the_probe_under_one_deadline(self):
        text = STAGE_WRAPPER.read_text()
        assert text.index("pytest") < text.index("ab_batched_best_form.py"), (
            "the correctness packet runs first: code/native proof must survive a "
            "deadline stop that lands in the timed legs")
        assert "DEADLINE" in text and "budget" in text.lower()

    def test_wrapper_cap_is_enforced_on_both_phases(self):
        text = STAGE_WRAPPER.read_text()
        assert "timeout" in text, "each phase is bounded by what is left of the one deadline"

    def test_budget_is_an_integer_within_cap_before_any_phase(self):
        """Garbage or over-cap ceilings are refusals, validated before phase 1."""
        text = STAGE_WRAPPER.read_text()
        assert "BUDGET must be an integer 1..2700" in text
        assert "exit 2" in text
        assert text.index("BUDGET must be an integer 1..2700") < text.index('/usr/bin/timeout "$CORRECTNESS_BOUND"'), (
            "the ceiling is validated before the correctness packet, not after a phase "
            "has already run")

    def test_only_exhaustion_stops_before_the_probe(self):
        """No arbitrary slack heuristic: <=0 is the one wrapper-side stop."""
        text = STAGE_WRAPPER.read_text()
        assert "-le 0" in text
        assert "-le 30" not in text, (
            "the removed arbitrary '30s left' heuristic must not return; the probe's "
            "own --budget-s guard controls the actual work")


class TestProvisionerQualifyGitroot:
    """Qualify builds and authenticates from the external --source-ref checkout,
    never from this script's parentless PB-snapshot root (whose HEAD is not the
    branch commit an --expected-commit names)."""

    PROVISIONER = CHECKOUT / "tools" / "provision_producer_env.py"

    def test_qualify_requires_source_ref_and_binds_gitroot_there(self):
        text = self.PROVISIONER.read_text()
        assert "--phase qualify requires --source-ref" in text
        assert "build_root = Path(args.source_ref).resolve()" in text
        assert 'git_in(build_root, "rev-parse", "HEAD")' in text, (
            "the qualified HEAD is read from the --source-ref checkout")
        assert 'git_in(build_root, "status", "--porcelain", "--untracked-files=no")' in text, (
            "the clean-committed-payload check runs against the --source-ref tree")
        assert '"git", "-C", str(build_root), "archive", "HEAD"' in text, (
            "the wheel is archived from the --source-ref checkout's exact HEAD")
        assert 'TESSERA_PRODUCER_SOURCE=str(build_root / "src" / "tessera")' in text, (
            "the authentication binds the SAME ref the wheel was built from")
        assert "TESSERA_PRODUCER_PYTHON=str(python)" in text, (
            "the newly created interpreter is explicitly selected for its actual authentication")
        assert "producer authentication returned no selected-producer receipt" in text
        assert text.index("record.write_text") > text.index("receipt = json.loads(auth.stdout"), (
            "a qualification record is published only after a non-null authentication")

    def test_qualify_checks_genuine_ancestry(self):
        text = self.PROVISIONER.read_text()
        assert "b770727c50eef822132518bdc4fd6efe84359c9e" in text, (
            "the qualified tree must descend from the campaign's b770 source ancestor")
        assert '"merge-base", "--is-ancestor"' in text

    def test_acquire_records_its_snapshot_kind(self):
        text = self.PROVISIONER.read_text()
        assert "PB sealed snapshot (acquire prototype)" in text, (
            "acquire wheels are prototype bindings of the sealed snapshot tree, and the "
            "record says so; final qualification is a NEW prefix built from the frozen "
            "--source-ref commit")


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
