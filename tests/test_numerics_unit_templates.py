"""Check reviewed unit templates without a Flux client or numerical work."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

import pytest

from experiments.numerics_rows import phase_instance, prepare_split
from experiments.numerics_rows.common import Invalid


def templates(root, canary=False, instance_lists=None, module="module-A", paths=None):
    root.mkdir()
    source = Path(prepare_split.__file__).parent
    for name in ("phase_instance.py", "feeder-settings.json"):
        shutil.copyfile(source / name, root / {"phase_instance.py": "instance.py"}.get(name, name))
    if instance_lists is not None:
        settings = json.loads((root / "feeder-settings.json").read_text())
        settings["template_instances"] = instance_lists
        (root / "feeder-settings.json").write_text(json.dumps(settings))
    config = {"module": module, "tool_commit": "a" * 40, "canary": canary,
              "paths": paths or {key: str(root / key) for key in prepare_split.INPUTS},
              "staged_population": paths["population"] if paths else str(root / "population.json")}
    prepare_split.write_templates(root, config, root / "repository.git")
    prepare_split.bind_templates(root, config)
    return config, [root / "templates" / unit["key"] / unit["phase"] for unit in config["templates"]]


def payload_arguments(directory, attempt, tools):
    """Expand the generated shell. Collect arguments at the Python worker boundary."""
    tools.mkdir(exist_ok=True)
    python = tools / "python3"
    collected = tools / "arguments.json"
    collected.unlink(missing_ok=True)
    python.write_text(f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
if sys.argv[1] == '-c' and sys.argv[5:6] == ['experiments.numerics_rows.row']:
    Path({str(collected)!r}).write_text(json.dumps(sys.argv[6:]))
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
""")
    python.chmod(0o700)
    result = subprocess.run(["bash", directory / "payload.sh", attempt], capture_output=True, text=True,
                            env={"PATH": str(tools) + os.pathsep + os.defpath})
    assert result.returncode == 0, "generated payload did not reach the Python worker"
    return json.loads(collected.read_text())


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob("*")}


def refuse(script, arguments, text, root):
    before = snapshot(root)
    result = subprocess.run(["bash", str(script), *map(str, arguments)], text=True, capture_output=True)
    assert result.returncode == 3, result.stderr
    assert text in result.stderr
    assert snapshot(root) == before


def prepared(attempt):
    directory = attempt / "prepare"
    directory.mkdir()
    raw = json.dumps({"schema": "tessera.dense_row_phase.v2", "phase": "prepare", "completion": True}).encode()
    (directory / "phase.json").write_bytes(raw)
    (directory / "completion.sha256").write_text(hashlib.sha256(raw).hexdigest() + "  phase.json\n")


def job_world(root, fault=""):
    tools, scratch, calls = root / "tools", root / "scratch", root / "calls"
    tools.mkdir()
    scratch.mkdir()
    scripts = {
        "git": '[[ "$*" != *rev-parse* ]] || printf "%s\\n" ' + "a" * 40,
        "python3": f'''if [[ "$1" == */cdi_env.py ]]; then shift 6; exec "$@"; fi
if [[ "$*" == *--verify* ]]; then step=verify; elif [[ "$2" == check ]]; then step=instance; else exit 0; fi
printf '%s\\n' "$step" >> {shlex.quote(str(calls))}
[[ "$step" != {shlex.quote(fault)} ]] || {{ echo "refused $step" >&2; exit 3; }}
[[ "$step" != instance ]] || printf '{{"checked":true}}\\n'
''',
        "podman": '[[ "$1" != run ]] || exit 9',
    }
    for name, body in scripts.items():
        path = tools / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o700)
    return {"PATH": str(tools) + os.pathsep + os.defpath, "FLUX_SCRATCH": str(scratch), "FLUX_JOB_ID": "unit-job"}, calls


def test_generated_job_setup(tmp_path, monkeypatch):
    from experiments.numerics_rows import counted_feeder
    from experiments.numerics_rows.common import write_json
    from tests.test_numerics_feeder_graph import make_graph

    canary, bookings, configs = make_graph(tmp_path, monkeypatch, modules=("module-A",))
    root = tmp_path / "module-A"
    config = configs["module-A"]
    repository = root / "repository.git"
    repository.mkdir()
    files = ["tests/test_numerics_fixture.py"]
    (repository / "tests").mkdir()
    (repository / files[0]).touch()
    tool = Path(prepare_split.__file__)
    local_module = repository / "experiments/numerics_rows" / tool.name
    local_module.parent.mkdir(parents=True)
    shutil.copyfile(tool, local_module)
    for arguments in (("init", "-q"), ("add", "."), ("-c", "user.name=Fixture", "-c",
                      "user.email=fixture@example.com", "commit", "-qm", "fixture")):
        assert subprocess.run(["git", *arguments], cwd=repository, capture_output=True).returncode == 0
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    for unit in config["templates"]:
        shutil.rmtree(root / "templates" / unit["key"] / unit["phase"])
    cpu_check = root / "cpu-check.json"
    arm = dict(source=revision, exit=0, failures=0, errors=0, skips=0, selected=1, passed=True)
    cpu_check.write_text(json.dumps(dict(verdict="PASS", after=[arm, arm], test_files=files)))
    population = Path(config["paths"]["population"])
    population.write_text("{}")
    digests = {}
    for key, path in config["paths"].items():
        if key not in ("source", "manifest"):
            Path(path).touch()
            digests[key] = prepare_split.sha256(path)
    write_json(canary / "campaign.json", dict(canary=True, module="module-A", tool_commit=revision,
               paths=config["paths"], digests=digests))
    for phase in ("prepare", "capture", "compare"):
        (canary / phase / "results").mkdir(parents=True)
        (canary / phase / "tool-exit.txt").write_text("0\n")
    canary_receipt = canary / "compare/results/numerics_row.json"
    write_json(canary_receipt, dict(canary=True, tool={"commit": revision},
               tessera={"commit": prepare_split.SOURCE}, rule={"sha256": prepare_split.RULE_SHA, "h1_sha256": prepare_split.H1_SHA},
               row=dict(outcome="PASS", exit_status=0, finalization_finished=True), units=[{"outcome": "PASS"}]))
    meter = dict.fromkeys(("wall_minutes", "gpu_busy_minutes", "peak_device_memory_mib", "peak_power_watts",
                          "mean_sample_power_watts", "whole_job_peak_memory_bytes", "sampled_seconds", "busy_sampled_seconds"), 1)
    write_json(canary / "capture/results/numerics_row.json",
               {"row": {"gpu_minutes": dict(meter, samples=[{"compute_process_count": 1}] * 2, errors=[])}})
    config.update(tool_commit=revision, source_revision=prepare_split.SOURCE, rule_sha256=prepare_split.RULE_SHA,
                  freeze_sha256=prepare_split.RULE_SHA, cpu_check=str(cpu_check), digests=digests,
                  canary_directory=str(canary), canary_evidence={"path": str(canary_receipt), "sha256": prepare_split.sha256(canary_receipt)})
    prepare_split.write_templates(root, config, repository)
    prepare_split.bind_templates(root, config)
    freeze_sha = json.loads((root / "campaign.json").read_text())["freeze_sha256"]
    broker = {"broker.starttime": "123", "local-uri": "local://test"}
    monkeypatch.setattr(counted_feeder, "observe", lambda name: dict(name=name, broker_starttime=broker["broker.starttime"],
                        uri=broker["local-uri"], config_sha256=hashlib.sha256(b"{}").hexdigest()))
    counted_feeder.run(canary, bookings)
    unit = next(u for u in config["templates"] if u["phase"] == "compare")
    directory = root / "templates" / unit["key"] / "compare"
    attempt, = counted_feeder.attempt_jobs(root, unit["key"])
    out = attempt / "compare"
    shutil.rmtree(out)
    scratch, tools = tmp_path / "scratch", tmp_path / "tools"
    scratch.mkdir()
    tools.mkdir()
    job_id = (attempt / "compare-jobid").read_text().strip()
    task = scratch / job_id / "work"
    for name, code in {
        "git": f"""if [[ "$PWD" == {shlex.quote(config["paths"]["source"])} ]]; then printf '%s\\n' {prepare_split.SOURCE}
elif [[ "${{FIXTURE_WRONG_HEAD:-}}" && "$3" == rev-parse ]]; then printf 'other\\n'
else exec {shutil.which("git")} "$@"; fi""",
        "grep": f"""if [[ "${{FIXTURE_SIGNAL:-}}" ]]; then kill -s "$FIXTURE_SIGNAL" 0; fi
exec {shutil.which("grep")} "$@" """,
        "podman": """if [[ "$1" == image ]]; then
    test -d "$HOME" -a -d "$TMPDIR" -a -s "$CONTAINERS_CONF" -a -s "$CONTAINERS_STORAGE_CONF" || exit 18
    printf 'observation boundary\\n' >&2
    exit 17
fi""",
        "python3": f"""#!{sys.executable}
import json, os, runpy, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, {str(tool.parents[2])!r})
from experiments.numerics_rows import common, phase_instance
if sys.argv[1] == '-m':
    script = Path(os.environ['PYTHONPATH']) / 'experiments/numerics_rows/prepare_split.py'
    sys.argv = [str(script), *sys.argv[3:]]
    original = common.sha256
    pinned = {{{str(prepare_split.RULE)!r}: {prepare_split.RULE_SHA!r}, {str(prepare_split.FREEZE)!r}: {freeze_sha!r}}}
    with patch.object(common, 'sha256', lambda path: pinned.get(str(path)) or original(path)):
        runpy.run_path(script, run_name='__main__')
elif sys.argv[2] == 'complete':
    phase_instance.complete(Path(sys.argv[3]))
else:
    assert sys.argv[2] == 'check'
    phase_instance.INSTANCE = Path({str(canary / "fixture-instance")!r})
    def query(command, *args):
        if args[0] == 'job':
            return args[-1]
        return {json.dumps(broker)!r} if 'attr.get' in args[-1] else '{{}}'
    phase_instance.query = query
    print(json.dumps(phase_instance.check(Path(sys.argv[3]), sys.argv[4])))
""",
    }.items():
        path = tools / name
        path.write_text(code if code.startswith("#!") else "#!/bin/bash\n" + code + "\n")
        path.chmod(0o700)

    env = {"PATH": str(tools) + os.pathsep + os.defpath, "FLUX_SCRATCH": str(scratch), "FLUX_JOB_ID": job_id}
    (canary / "fixture-instance").write_text(str(tools))
    Path(config["paths"]["source"]).mkdir()

    def run(environment=env, *, script=directory / "job.sh", job=attempt):
        return subprocess.run([script, job], env=environment, capture_output=True, text=True, timeout=30, start_new_session=True,
                              preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))

    accepted = run()
    assert accepted.returncode == 17 and "observation boundary" in accepted.stderr
    assert (out / "affinity.txt").read_text().startswith("Cpus_allowed_list:")
    assert json.loads((out / "instance.json").read_text()) == json.loads((attempt / "compare-instance.json").read_text())
    assert (out / "tool-exit.txt").read_text() == "17\n"
    assert not task.parent.exists()
    shutil.rmtree(out)

    from experiments.numerics_rows import row
    from experiments.numerics_rows.phase_lifecycle import handled_signals

    for unit in config["templates"]:
        phase, key = unit["phase"], unit["key"]
        job, = counted_feeder.attempt_jobs(root, key)
        phase_out = job / phase
        saved = phase_out.rename(job / "fixture-output") if phase_out.exists() else None
        job_id = (job / f"{phase}-jobid").read_text().strip()
        for number in (signal.SIGTERM, signal.SIGINT):
            script = root / "templates" / key / phase / "job.sh"
            stopped = run(dict(env, FLUX_JOB_ID=job_id, FIXTURE_SIGNAL=number.name), script=script, job=job)
            assert stopped.returncode == 128 + number
            assert (phase_out / "tool-exit.txt").read_text().strip() == str(128 + number)
            assert not (scratch / job_id).exists()
            worker_unit = row.new_unit({"cell": "dense", "role": "unit"})
            if number == signal.SIGINT:
                worker_unit["findings"].append("exact mismatch")
            with pytest.raises(SystemExit) as worker:
                with handled_signals({"row": {}, "units": [worker_unit]}, phase_out):
                    signal.raise_signal(number)
            result = json.loads((phase_out / "numerics_row.json").read_text())["row"]
            assert worker.value.code == result["exit_status"] == 128 + number
            assert result["stop_cause"] == number.name and not result["finalization_finished"]
            assert result["outcome"] == ("FAIL" if number == signal.SIGINT else "STOPPED")
            shutil.rmtree(phase_out)
        if saved is not None:
            saved.rename(phase_out)

    for name in ("FLUX_SCRATCH", "FLUX_JOB_ID"):
        rejected = run({key: value for key, value in env.items() if key != name})
        assert rejected.returncode != 0 and name in rejected.stderr and not out.exists()
    blocked = tmp_path / "blocked-scratch"
    blocked.touch()
    rejected = run(dict(env, FLUX_SCRATCH=str(blocked)))
    assert rejected.returncode != 0 and "mkdir:" in rejected.stderr and not out.exists()
    moved_repository = repository.rename(repository.with_name("moved.git"))
    rejected = run()
    assert rejected.returncode != 0 and "does not exist" in rejected.stderr and not out.exists()
    moved_repository.rename(repository)
    rejected = run(dict(env, FIXTURE_WRONG_HEAD="1"))
    assert rejected.returncode == 1 and "observation boundary" not in rejected.stderr and not out.exists()
    population.write_text("[]")
    rejected = run()
    assert rejected.returncode != 0 and "split row bound input changed: population" in rejected.stderr
    assert not out.exists()
    payload = directory / "payload.sh"
    payload.write_text(payload.read_text() + "\n")
    rejected = run()
    assert rejected.returncode != 0 and "payload.sh: FAILED" in rejected.stdout
    assert not out.exists()
    moved = directory.rename(directory.with_name("moved"))
    rejected = run(script=moved / "job.sh")
    assert rejected.returncode != 0 and "No such file or directory" in rejected.stderr and not out.exists()


@pytest.mark.parametrize("fault", ["verify", "instance", ""])
def test_job_checks_documents_before_claim(tmp_path, fault):
    assert (Path(prepare_split.__file__).parent / "templates/job.sh").read_text().index("--verify @ROOT@") < (Path(prepare_split.__file__).parent / "templates/job.sh").read_text().index('mkdir "$attempt/@PHASE@"'), "feature absent: document checks precede the phase claim"
    _, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == "prepare")
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    environment, calls = job_world(tmp_path, fault)
    before = snapshot(tmp_path / "module")
    result = subprocess.run([str(directory / "job.sh"), str(attempt)], env=environment, text=True, capture_output=True)
    assert result.returncode == (3 if fault else 9)
    assert calls.read_text().splitlines() == (["verify"] if fault == "verify" else ["verify", "instance"])
    assert not any((tmp_path / "scratch").iterdir())
    if fault:
        assert "refused " + fault in result.stderr
        assert snapshot(tmp_path / "module") == before
    else:
        assert json.loads((attempt / "prepare/instance.json").read_text()) == {"checked": True}


def test_every_template_uses_cdi_replay_or_gpu_device(tmp_path):
    _, directories = templates(tmp_path / "module")
    for directory in directories:
        launch, = [line for line in (directory / "job.sh").read_text().splitlines() if line.startswith("timeout ")]
        replay = "/experiments/numerics_rows/cdi_env.py" in launch
        device = "--device nvidia.com/gpu=all" in launch
        if directory.name == "capture":
            assert device and not replay, "capture must request the CDI GPU device"
        else:
            assert replay and not device, f"{directory.name} lacks CPU CDI replay"


def test_unit_payloads_select_their_tp_groups(tmp_path, monkeypatch):
    from experiments.numerics_rows import row

    config, directories = templates(tmp_path / "module", canary=True)
    groups = [(None, None, {"size": size}) for size in (1, 2)]
    parse_args = row.argparse.ArgumentParser.parse_args

    class PayloadParsed(Exception):
        pass

    def check_payload(parser):
        args = parse_args(parser, payload[payload.index("experiments.numerics_rows.row") + 1:])
        try:
            selected = row.selected_groups(groups, args.tp_size, args.tp_rank)
        except row.Invalid as error:
            raise AssertionError(f"TP_PAYLOAD_REFUSED: {error}") from error
        assert selected == [groups[unit["size"] - 1]], "TP_PAYLOAD_WRONG_GROUP"
        raise PayloadParsed

    monkeypatch.setattr(row.argparse.ArgumentParser, "parse_args", check_payload)
    for unit, directory in zip(config["templates"], directories):
        if "size" in unit:
            payload = shlex.split((directory / "payload.sh").read_text().split("exec ", 1)[1])
            with pytest.raises(PayloadParsed):
                row.main()


def gate_worker(root, monkeypatch):
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from experiments.numerics_rows import cases, common, dependency_code, launch_state, row, sass_slice, scope, serving_build
    from tessera import routed_fused
    from tessera.fused import pack_fused
    from tessera.serving import ext
    from tessera.unit_artifact import parse_unit_metadata
    from tests.test_numerics_rows_receipts import population_cases

    sha, write = common.sha256, common.write_json
    paths = {key: root / key for key in (*prepare_split.INPUTS, "rule", "freeze")}
    contract, population, _ = population_cases()
    entry = population["modules"][0]
    name = entry["name"] = "model.layers.0.mlp.down_proj"
    unit = (Path(__file__).parent / "data/legacy/e4m3-1024-window-channel-256c.tessera").read_bytes()
    blob = pack_fused([("down_proj", parse_unit_metadata(unit).rows, unit)])
    scheme = cases.projection_scheme(dict(family="TESSERA_FP8", grid="E4M3", body="WINDOW", plane="CHANNEL"), "down_proj", blob)
    entry.update(root=str(root), family=scheme["family"], roles=scheme["roles"], q256=scheme["q256"])
    write(root / "config.json", {"quantization_config": {"config_groups": {"dense": {"scheme": scheme, "targets": [name]}}}})
    tensor = name + ".wire_bytes"
    write(root / "model.safetensors.index.json", {"weight_map": {tensor: "wire.safetensors"}})
    save_file({tensor: torch.frombuffer(bytearray(blob), dtype=torch.uint8)}, root / "wire.safetensors")
    wire = root / "reader.wire"
    wire.write_bytes(blob)
    runtime = common.IMAGES["N"]
    for cell in contract["lane_eligibility"]["cells"]:
        cell["runtime"]["image"] = runtime["image_digest"]
        cell["rungs_q256"] = [scheme["q256"]]
    draft = root / "draft.json"
    write(draft, dict(population, image="N", modules=[entry], wire_stamps_read=True))
    write(paths["population"], dict(json.loads(draft.read_text()), reviewed=True,
        review=dict(signed_by="lead", signed_utc="then", source_file=str(draft), source_flux_job="inventory", source_sha256=sha(draft), spec="test")))

    source = paths["source"]
    contract_path = source / "src/tessera/serving/runtime_contract.json"
    kernel = source / common.KERNEL
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"retained kernel fixture\n")
    write(contract_path, contract)
    for args in (("init", "-q"), ("add", "src"), ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.com",
                 "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture")):
        status = subprocess.run(["git", *args], cwd=source, capture_output=True).returncode
        assert status == 0
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    binary = dict(sha256="a" * 64)
    library = dict(key="e4m3", name=ext.ROUTED_FUSED_E4M3_MODULE_NAME, library=binary, probe=binary,
                   probe_source_sha256="probe", cflags=routed_fused._cflags("sm_121", True))
    proof = dict(library_so_sha256=binary["sha256"], library_code_sha256="code", library_code_sections=[], cubin_sha256s=["cubin"])
    write(paths["build"], dict(commit=commit, runtime=runtime, libraries=[library]))
    write(paths["build_provenance"], dict(commit=commit, build_sha256=sha(paths["build"]), libraries=[dict(
        library, native_source_path=str(kernel), native_source_sha256=sha(kernel))]))
    for module in (common, serving_build):
        monkeypatch.setattr(module, "KERNEL_SHA256", sha(kernel))
    monkeypatch.setattr(ext, "native_source_path", lambda name: str(kernel))
    monkeypatch.setattr(row, "check_runtime", lambda image: dict(runtime))
    monkeypatch.setattr(common, "register_native_ops", lambda image: {"result": "PASS"})
    monkeypatch.setattr(dependency_code.CodeAudit, "install", lambda self: self)
    monkeypatch.setattr(sass_slice, "match_library", lambda *args: proof)

    state = launch_state.LaunchState()
    state.bind_math(torch)
    monkeypatch.setattr(launch_state, "LaunchState", lambda path=launch_state.LEVERS, **kwargs: type(state)(path))
    paths["rule"].write_text("rule fixture\n")
    write(paths["freeze"], dict(sha256=sha(paths["rule"]), frozen_utc="then", reviewer="lead",
          decisions={f"D{i}": "yes" for i in range(1, 10)}))
    write(paths["freeze_addendum"], dict(freeze_record_sha256=sha(paths["freeze"]), rule_sha256=sha(paths["rule"]),
          reviewer="lead", frozen_utc="then", launch_levers_sha256=state.levers_sha256))
    monkeypatch.setattr(row, "FREEZE_ADDENDUM", paths["freeze_addendum"])
    for key in ("rule", "freeze"):
        monkeypatch.setattr(prepare_split, key.upper(), paths[key])
    write(paths["h1"], dict(state.fields(), flux_job_id="H1", build_sha256=sha(paths["build"]), preconditions_confirmed=True,
          P1={"mismatches": 0}, P2={"result": "PASS"}, P3={"e4m3": proof}, libraries={"e4m3": {"P4": True, "P5": True}}))
    dom = root / "d-table"
    np.zeros(1, dtype="<f4").tofile(dom)
    samples = torch.tensor([0, *cases.sweep_values("e4m3")], dtype=torch.bfloat16)
    scales = (samples.float().abs() / 448).clamp_min(2**-24)
    quantizer = root / "quantizer.npz"
    np.savez(quantizer, input_bits=samples.view(torch.uint16).numpy()[:, None],
             code_bytes=(samples.float() / scales).to(torch.float8_e4m3fn).view(torch.uint8).numpy(), scale_bytes=scales.numpy().view(np.uint8))
    q1 = dict(tables=[dict(key="e4m3", library_so_sha256=binary["sha256"], probe_sha256=binary["sha256"], L_bits=0, dom_size=1,
        d_table_sha256=sha(dom), dom_sha256=sha(dom), dom_path=str(dom), d_table_path=str(dom), d_nondecreasing_0_L=True)],
        quantizers=[dict(family="e4m3", gs=1.0, path=str(quantizer), sha256=sha(quantizer), repeat_equal=True, smallest_normal_unsaturated_j=None)])
    write(paths["q1"], q1)
    _, weights, scale, _ = cases.decode_roles(blob, scheme, name)[0]
    scale = scale.float().reshape(-1).expand(scheme["rows"])
    checked = cases.column_coverage(weights.numpy(), scale.numpy(), "e4m3", cases.QuantizerTable(q1, "e4m3"))
    cuts = [dict(tp=cut, launch_geometry=geometry, m_configs=cases.configurations(**geometry, library="e4m3", sms=1, structure="dense"),
            roles=[dict(name="down_proj", expert=0, cols=geometry["cols"], cut_axis="column", cut_range=[0, geometry["cols"]],
                        matches_attested_wire=True, reader_path=str(wire), reader_sha256=sha(wire), separation=checked, gs=1.0, lut=[], scale_index=[])])
            for cut, geometry in scope.wire_geometries(entry)]
    write(paths["manifest"], dict(state.fields(), source_commit=commit, contract_sha256=sha(contract_path),
          population_sha256=sha(paths["population"]), q1_table_sha256=sha(paths["q1"]), selected_modules=[name],
          generation_finished=True, modules=[dict(entry, cuts=cuts, complete=checked["complete"], wire_sha256=sha(wire))]))
    config, directories = templates(root / "campaign", module=name, canary=True,
                                    paths={key: str(paths[key]) for key in prepare_split.INPUTS})
    paired = {u["phase"]: p for u, p in zip(config["templates"], directories) if u.get("size") == 1 and u["case_set"] == "gate"}
    attempt = root / "campaign/units" / paired["prepare"].parent.name / "attempts/1"
    monkeypatch.setattr(sys, "path", sys.path[:])
    check_output, tool_root = row.subprocess.check_output, Path(row.__file__).resolve().parents[2]

    def tool_git(command, **kw):  # the CPU check runs the tool from an exported tree, not a Git checkout
        if command[:1] != ["git"] or Path(kw.get("cwd", ".")).resolve() != tool_root:
            return check_output(command, **kw)
        return commit + "\n" if command[1:] == ["rev-parse", "HEAD"] else b"experiments/numerics_rows/row.py\0"
    monkeypatch.setattr(row.subprocess, "check_output", tool_git)

    def run(phase):
        out = attempt / phase
        out.mkdir(parents=True)
        write(out / "instance.json", {"name": "fv1"})
        command = (prepare_split.row_arguments(config, root / "campaign", source, paths["population"], config["paths"],
                    phase, unit={"case_set": "all"}, attempt=attempt) if phase == "compare"
                   else (paired[phase] / "payload.sh").read_text().split("exec ", 1)[1])
        fields = shlex.split(command)
        payload = fields[fields.index("experiments.numerics_rows.row") + 1:]
        monkeypatch.setattr(sys, "argv", ["row", *(arg.replace("$attempt", str(attempt)) for arg in payload)])
        row.main()

    return run, attempt, paths


def finish_gate_prepare(run, prepared):
    with pytest.raises(SystemExit) as exited:
        run("prepare")
    cause = exited.value.__cause__
    if isinstance(cause, TypeError) and "pstats.Stats" in str(cause):
        raise AssertionError("GATE_PREPARE_CANNOT_FINISH") from cause
    if exited.value.code != 0:
        raise exited.value
    return json.loads((prepared / "phase.json").read_text())


def test_generated_tp1_gate_prepare_finishes(tmp_path, monkeypatch):
    from experiments.numerics_rows.phase_artifact import records

    run, attempt, _ = gate_worker(tmp_path, monkeypatch)
    prepared = attempt / "prepare"
    document = finish_gate_prepare(run, prepared)
    assert document["completion"] is True and document["case_set"] == "gate"
    assert document["receipt"]["row"]["prepare_finished"] is True
    assert document["streams"]["cases"]["count"] > 0
    assert all(not record["file"].startswith("diagnostic-reference.") for record in records(prepared, "files", document))


def stub_capture_device(monkeypatch):
    from types import SimpleNamespace
    import torch
    from experiments.numerics_rows import gpu_time

    check_output = subprocess.check_output
    monkeypatch.setattr(subprocess, "check_output", lambda command, *args, **kwargs:
                        "fixture driver\n" if command[0] == "nvidia-smi" else check_output(command, *args, **kwargs))
    monkeypatch.setattr(gpu_time, "RowMeter", lambda: SimpleNamespace(finish=lambda: {}))
    monkeypatch.setattr(gpu_time, "preflight", lambda **kwargs: {"verdict": "FREE"})
    monkeypatch.setattr(gpu_time, "self_visibility", lambda torch: {"verdict": "VISIBLE"})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fixture device")


def test_generated_tp1_gate_capture_enters_device_capture(tmp_path, monkeypatch):
    from experiments.numerics_rows import dense_phases

    run, attempt, _ = gate_worker(tmp_path, monkeypatch)
    prepared = attempt / "prepare"
    assert finish_gate_prepare(run, prepared)["completion"] is True

    class CaptureEntered(SystemExit):
        pass

    def enter(args, receipt, phases, groups, *, library, compile_paths):
        assert [cut for _, _, cut, _ in groups] == [{"size": 1, "rank": 0}]
        assert args.prepare_artifact == [prepared] and args.case_set == "gate"
        raise CaptureEntered

    stub_capture_device(monkeypatch)
    monkeypatch.setattr(dense_phases, "device_capture", enter)
    with pytest.raises(CaptureEntered):
        run("capture")


def test_join_cli_refuses_finished_canary(tmp_path, monkeypatch):
    from experiments.numerics_rows.common import write_json

    _, _, paths = gate_worker(tmp_path, monkeypatch)
    manifest = json.loads(paths["manifest"].read_text())
    receipt, output = tmp_path / "canary.json", tmp_path / "join.json"
    runtime = {key: manifest[key] for key in ("launch_state_names", "launch_state_sha256", "launch_selectors", "start_environment_spec_sha256")}
    write_json(receipt, {"schema": "tessera.numerics_row.v2", "canary": True, "runtime": runtime,
        "inputs": {"launch_levers_sha256": manifest["launch_levers_sha256"]},
        "row": {"outcome": "PASS", "exit_status": 0, "utc_end": "then", "finalization_finished": True}})
    contract = paths["source"] / "src/tessera/serving/runtime_contract.json"
    result = subprocess.run([sys.executable, "-B", "-m", "experiments.numerics_rows.join",
        "--population", str(paths["population"]), "--contract", str(contract), "--manifest", str(paths["manifest"]),
        "--row", str(receipt), "--out", str(output)], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert result.stderr.splitlines()[-1:] == ["experiments.numerics_rows.common.Invalid: a canary cannot supply attestation"]
    assert result.returncode == 1 and not output.exists()


def test_templates_cover_units_phases_and_manifest(tmp_path):
    assert hasattr(prepare_split, "write_templates"), "feature absent: one reviewed template per unit and phase"
    config, directories = templates(tmp_path / "module")
    assert len(directories) == 9
    assert len(config["unit_keys"]) == len(set(config["unit_keys"])) == 4
    for size in (1, 2):
        for part in ("gate", "diagnostics"):
            key, = [k for k in config["unit_keys"] if k.startswith(f"tp{size}-{part}-")]
            for phase in ("prepare", "capture"):
                directory = tmp_path / "module/templates" / key / phase
                payload = shlex.split((directory / "payload.sh").read_text().split("exec ", 1)[1])
                assert payload[payload.index("--tp-size") + 1] == str(size)
                assert payload[payload.index("--case-set") + 1] == part
                if size == 1:
                    assert payload[payload.index("--tp-rank") + 1] == "0"
                else:
                    assert "--tp-rank" not in payload
                assert payload[payload.index("--out") + 1] == "$attempt"
                assert "--phase-local-output" in payload
    for directory in directories:
        manifest = (directory / "template.sha256").read_text()
        assert "  payload.sh\n" in manifest
        assert all("  " + str(tmp_path / "module" / name) + "\n" in manifest
                   for name in ("instance.py", "campaign.json", "feeder-settings.json"))
        assert "  job.sh\n" in manifest, "template manifest omits job.sh"
        assert "  table.json\n" not in manifest
        result = subprocess.run(["sha256sum", "-c", "template.sha256"], cwd=directory, capture_output=True)
        assert result.returncode == 0, result.stderr
    assert not (tmp_path / "module/units").exists()


@pytest.mark.parametrize("phase", ("prepare", "capture", "compare"))
@pytest.mark.parametrize("count", (0, 2))
def test_template_refuses_argument_count(tmp_path, phase, count):
    assert hasattr(prepare_split, "write_templates"), "feature absent: template requires exactly one argument"
    _, directories = templates(tmp_path / "module")
    script = next(p for p in directories if p.name == phase) / "job.sh"
    refuse(script, ["unused"] * count, "require one attempt argument", tmp_path)


@pytest.mark.parametrize("mutation", ("module", "unit", "zero", "symlink", "parent"))
@pytest.mark.parametrize("phase", ("prepare", "capture", "compare"))
def test_template_refuses_other_paths_and_attempt_ids(tmp_path, mutation, phase):
    assert hasattr(prepare_split, "write_templates"), "feature absent: template fixes module unit and canonical attempt"
    config, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == phase)
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    reason = "attempt path differs from fixed module and unit"
    if mutation == "module":
        attempt = tmp_path / "other/units" / directory.parent.name / "attempts/1"
    elif mutation == "unit":
        attempt = attempt.parents[2] / config["unit_keys"][1] / "attempts/1"
    elif mutation == "zero":
        attempt = attempt.with_name("0")
    elif mutation == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        attempt.rmdir()
        attempt.symlink_to(target, target_is_directory=True)
        reason = "attempt path is not canonical"
    else:
        attempt = str(attempt) + "/../1"
    refuse(directory / "job.sh", [attempt], reason, tmp_path)


@pytest.mark.parametrize("phase", ("prepare", "capture", "compare"))
def test_template_refuses_existing_phase_claim(tmp_path, phase):
    assert hasattr(prepare_split, "write_templates"), "feature absent: template claims phase with one atomic mkdir"
    _, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == phase)
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    if phase == "capture":
        prepared(attempt)
    (attempt / phase).mkdir()
    environment, _ = job_world(tmp_path)
    before = snapshot(tmp_path / "module")
    result = subprocess.run([str(directory / "job.sh"), str(attempt)], env=environment, text=True, capture_output=True)
    assert result.returncode == 3 and "phase claim already exists or failed" in result.stderr
    assert snapshot(tmp_path / "module") == before
    script = (directory / "job.sh").read_text()
    assert script.count('mkdir "$attempt/' + phase + '"') == 1
    assert 'mkdir -p "$attempt/' not in script


@pytest.mark.parametrize("state", ("missing", "incomplete", "no-hashes", "bad-hash"))
def test_capture_template_requires_complete_prepare(tmp_path, state):
    assert hasattr(prepare_split, "write_templates"), "feature absent: capture template checks prepare completion and hashes before writes"
    _, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == "capture")
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    if state != "missing":
        prepared(attempt)
        if state == "no-hashes":
            (attempt / "prepare/completion.sha256").unlink()
        elif state == "incomplete":
            (attempt / "prepare/phase.json").write_text('{"completion":false}')
        else:
            with (attempt / "prepare/phase.json").open("a") as stream:
                stream.write("\n")
    refuse(directory / "job.sh", [attempt], "prepare is not complete or hashes differ", tmp_path)


@pytest.mark.parametrize("canary", (False, True))
@pytest.mark.parametrize("prepare_instances", (None, ["fv1"], ["fv1e-sparky"], ["fv1", "fv1e-sparklina"]))
def test_every_template_uses_fleet_table_schema(tmp_path, canary, prepare_instances):
    assert "template_instances" in json.loads(Path(prepare_split.__file__).with_name("feeder-settings.json").read_text()), "feature absent: every template uses Fleet table schema"
    placement = {"canary": ["fv1"], "prepare": prepare_instances, "capture": ["fv1"], "compare": ["fv1e-sparklina"]}
    _, directories = templates(tmp_path / "module", canary, placement if prepare_instances is not None else None)
    for attempt_id, directory in enumerate(directories, start=4):
        assert (directory / "table.json").is_file(), "feature absent: every template uses Fleet table schema"
        phase = directory.name
        fields = json.loads((directory / "table.json").read_text())
        defaults = {"prepare": ["fv1", "fv1e-sparky", "fv1e-sparklina"], "compare": ["fv1"]}
        names = ["fv1"] if canary or phase == "capture" else (defaults[phase] if prepare_instances is None else placement[phase])
        instances = {}
        for name in names:
            hosts = {"fv1": ["sparky", "sparklina"], "fv1e-sparky": ["sparky"], "fv1e-sparklina": ["sparklina"]}[name]
            memory = {"fv1": 8, "fv1e-sparky": 9, "fv1e-sparklina": 9}[name] if phase == "prepare" else 32
            instances[name] = {"constraints": [{"hostlist": [host]} for host in hosts],
                               "shapes": [[{"type": "slot", "count": 1, "label": "task", "with":
                                           [{"type": "core", "count": 4 if canary and phase == "capture" else 1},
                                            {"type": "memory", "count": memory}]
                                           + ([{"type": "gpu", "count": 1}] if phase == "capture" else [])}]],
                               "max_duration_s": {"prepare": 28800 if "-gate-" in directory.parent.name else 18000,
                                                  "capture": 36000 if "-gate-" in directory.parent.name else 18000,
                                                  "compare": 86400}[phase]}
        assert fields == {"schema": 1, "instances": instances,
                          "job_name": f"tessera-1717-{directory.parent.name}-{phase}", "args": 1,
                          "max_attempts": 5, "urgency": 8,
                          "cwd": str(directory), "env_names": ["PATH", "HOME", "XDG_RUNTIME_DIR", "FLUX_F58_FORCE_ASCII", "FLUX_SCRATCH",
                                                                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}
        assert not {"policy.json", "submit-after-review.sh", "submit-reviewed.sh"} & {p.name for p in directory.iterdir()}
        assert (directory / "job.sh").stat().st_mode & 0o100
        attempt = tmp_path / "module/units" / directory.parent.name / f"attempts/{attempt_id}"
        attempt.mkdir(parents=True)
        if phase == "capture":
            prepared(attempt)
        result = subprocess.run([str(directory / "job.sh"), str(attempt)], text=True, capture_output=True,
                                env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"})
        assert result.returncode != 0 and "FLUX_SCRATCH" in result.stderr
        assert not (attempt / phase).exists()


def test_compare_template_lists_all_units_and_uses_fv1e(tmp_path):
    assert hasattr(prepare_split, "write_templates"), "feature absent: one module compare template selects every latest unit on fv1"
    config, directories = templates(tmp_path / "module")
    directory = directories[-1]
    payload = (directory / "payload.sh").read_text()
    command = shlex.split(payload.split("exec ", 1)[1])
    assert command.count("--prepare-artifact") == command.count("--capture-artifact") == 4
    assert command[command.index("--case-set") + 1] == "all"
    assert not {"--tp-size", "--tp-rank"} & set(command)
    for key in config["unit_keys"]:
        assert key in payload
    assert "compare-inputs.json" in payload and "find " not in payload
    assert list(json.loads((directory / "table.json").read_text())["instances"]) == ["fv1"]


def test_compare_payload_uses_saved_attempts(tmp_path):
    assert "compare-inputs.json" in inspect.getsource(prepare_split.bind_templates), "feature absent: compare payload reads saved input attempts"
    config, directories = templates(tmp_path / "module")
    directory = next(p for p in directories if p.name == "compare")
    attempt = tmp_path / "module/units" / directory.parent.name / "attempts/1"
    attempt.mkdir(parents=True)
    selected = {key: str(tmp_path / "module/units" / key / "attempts/1") for key in config["unit_keys"]}
    for path in selected.values():
        Path(path).mkdir(parents=True)
        Path(path).with_name("2").mkdir()
    (attempt / "compare-inputs.json").write_text(json.dumps(selected))
    prefix = (directory / "payload.sh").read_text().split("exec ", 1)[0]
    script = prefix + "printf '%s\\n' " + " ".join(f'"$pair_{index}"' for index in range(len(selected)))
    result = subprocess.run(["bash", "-c", script, "payload", str(attempt)], text=True, capture_output=True)
    assert result.returncode == 0 and result.stdout.splitlines() == list(selected.values())


@pytest.mark.parametrize('outcome,finalized', [('FAIL', True), ('STOPPED', True), ('FAIL', False)])
def test_compare_exit_three_keeps_checked_completion_hashes(tmp_path, outcome, finalized):
    assert 'r.get("finalization_finished")' in (Path(prepare_split.__file__).parent / 'templates/job.sh').read_text(), 'feature absent: finalized compare exit 3 has completion hashes'
    _, directories = templates(tmp_path / 'module')
    directory = next(p for p in directories if p.name == 'compare')
    job = tmp_path / 'compare-output'
    job.mkdir()
    (job / 'phase.json').write_text('{"completion":true}')
    receipt = job / 'numerics_row.json'
    receipt.write_text(json.dumps({'row': {'outcome': outcome, 'finalization_finished': finalized}}))
    tail = 'rc=0\n' + (directory / 'job.sh').read_text().split('rc=0\n', 1)[1]
    script = 'set -euo pipefail\ntimeout() { return 3; }\n'
    script += 'task_root=$PWD\ntask_scratch=$PWD\nname=stub\nphase_mounts=()\npopulation_mount=()\nclaimed=("$PWD")\n' + tail
    result = subprocess.run(['bash', '-c', script], cwd=job, text=True, capture_output=True)
    assert result.returncode == 3
    assert (job / 'completion.sha256').exists() is finalized
    if finalized:
        checked = subprocess.run(['sha256sum', '-c', 'completion.sha256'], cwd=job, capture_output=True)
        assert checked.returncode == 0
        receipt.write_text('{}')
        changed = subprocess.run(['sha256sum', '-c', 'completion.sha256'], cwd=job, capture_output=True)
        assert changed.returncode != 0


@pytest.mark.parametrize("phase", ("prepare", "capture"))
def test_phase_local_outputs_keep_partial_fail_evidence(tmp_path, phase):
    assert hasattr(prepare_split, "write_templates"), "feature absent: phase-local files retain partial FAIL and artifact paths"
    pytest.importorskip("torch")
    from experiments.numerics_rows import row
    from experiments.numerics_rows.gpu_time import PhaseTimes
    from tests.test_numerics_rows_phases import retry_phase_attempts
    dp, api, args, receipt, groups, attempt = retry_phase_attempts(tmp_path)
    document_path = attempt / phase / "phase.json"
    partial = json.loads(document_path.read_text())["receipt"]
    wanted = row.unit_identity(partial["units"][0])
    partial["units"][0]["findings"] = ["known mismatch"]
    partial["units"][0]["outcome"] = "FAIL"
    local_receipt = attempt / phase / "numerics_row.json"
    local_receipt.write_text(json.dumps(partial))
    document_path.unlink()
    args.prepare_artifact = [attempt.with_name("2") / "prepare" if path.parent == attempt else path for path in args.prepare_artifact]
    args.capture_artifact = [attempt.with_name("2") / "capture" if path.parent == attempt else path for path in args.capture_artifact]
    args.phase_root = args.out / "compare"
    args.out = args.phase_root
    args.out.mkdir()
    before = snapshot(tmp_path)
    dp.host_compare(args, receipt, PhaseTimes({}), groups)
    failed = next(unit for unit in receipt["units"] if row.unit_identity(unit) == wanted)
    assert receipt["row"]["outcome"] == failed["outcome"] == "FAIL"
    assert (args.out / "phase.json").is_file()
    assert not (args.out / "compare").exists()
    listing = next(item for item in receipt["attempts"]["listing"] if item["unit"] == attempt.parent.parent.name and item["attempt"] == 1)
    assert listing["phases"][phase + "_partial"] == hashlib.sha256(local_receipt.read_bytes()).hexdigest()
    prefix = str(args.out.relative_to(tmp_path)) + "/"
    assert all(name.startswith(prefix) for name, value in snapshot(tmp_path).items() if name not in before or before[name] != value)
    api.read(args.out, "compare", api.identities(receipt), dp.scopes(groups))


def test_prepare_takes_output_root_and_refuses_existing_root(tmp_path, monkeypatch):
    assert "output_root" in inspect.signature(prepare_split.prepare).parameters, "feature absent: explicit campaign output root with existing root refusal"
    assert inspect.signature(prepare_split.prepare).parameters["output_root"].default is None
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    monkeypatch.setattr(prepare_split, "FREEZE", freeze)
    monkeypatch.setattr(prepare_split.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    clones = []
    monkeypatch.setattr(prepare_split.subprocess, "run", lambda command, **kwargs: clones.append(command))
    output = tmp_path / "chosen root"
    module = "model.language_model.layers.2.mlp.gate_up_proj"
    assert prepare_split.prepare(tmp_path / "checks.json", module=module, output_root=output) == output
    config = json.loads((output / "campaign.json").read_text())
    assert config["module"] == module and config["canary"] is True
    assert len(config["templates"]) == 9 and len(clones) == 1
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        prepare_split.prepare(tmp_path / "checks.json", output_root=output)
    assert snapshot(tmp_path) == before and len(clones) == 1
    monkeypatch.undo()
    help_result = subprocess.run([sys.executable, "-m", "experiments.numerics_rows.prepare_split", "--help"],
                                 cwd=prepare_split.ROOT, capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--output-root" in help_result.stdout and "--module" in help_result.stdout


@pytest.mark.parametrize("canary", [True, False])
def test_phase_durations_from_settings_reach_table_and_job(tmp_path, canary):
    assert "phase_minutes" in json.loads(Path(prepare_split.__file__).with_name("feeder-settings.json").read_text()), "feature absent: phase and case set durations come from settings"
    config, directories = templates(tmp_path / "module", canary=canary)
    expected = {"prepare": {"gate": 480, "diagnostics": 300},
                "capture": {"gate": 600, "diagnostics": 300}, "compare": {"all": 1440}}
    for custom in (False, True):
        if custom:
            shutil.rmtree(tmp_path / "module/templates")
            expected = {phase: {part: minutes + 7 for part, minutes in parts.items()} for phase, parts in expected.items()}
            settings_path = tmp_path / "module/feeder-settings.json"
            settings = json.loads(settings_path.read_text())
            settings["phase_minutes"] = expected
            settings_path.write_text(json.dumps(settings))
            prepare_split.write_templates(tmp_path / "module", config, tmp_path / "module/repository.git")
        for unit, directory in zip(config["templates"], directories):
            seconds = expected[unit["phase"]][unit["case_set"]] * 60
            table = json.loads((directory / "table.json").read_text())
            assert all(instance["max_duration_s"] == seconds for instance in table["instances"].values())
            assert f"timeout --signal=TERM --kill-after=20s {seconds - 60}s " in (directory / "job.sh").read_text()


def cpu_campaign(tmp_path, monkeypatch):
    files = ["tests/test_numerics_alpha.py", "tests/test_numerics_beta.py"]
    rule, freeze, data, checks_path = [tmp_path / name for name in ("rule.md", "freeze.json", "population.json", "checks.json")]
    for path in (rule, freeze, data):
        path.write_text("{}")
    monkeypatch.setattr(prepare_split, "RULE", rule)
    monkeypatch.setattr(prepare_split, "RULE_SHA", prepare_split.sha256(rule))
    monkeypatch.setattr(prepare_split, "FREEZE", freeze)
    config = {"bound": True, "source_revision": prepare_split.SOURCE, "rule_sha256": prepare_split.RULE_SHA,
              "freeze_sha256": prepare_split.sha256(freeze), "tool_commit": "a" * 40, "cpu_check": str(checks_path),
              "canary": True, "paths": {"source": str(tmp_path), "population": str(data)},
              "digests": {"population": prepare_split.sha256(data)}}
    (tmp_path / "campaign.json").write_text(json.dumps(config))
    def git_output(command, **kwargs):
        if command[1] == "ls-tree":
            assert command == ["git", "ls-tree", "-r", "--name-only", config["tool_commit"], "--", "tests"]
            assert kwargs["cwd"] == prepare_split.ROOT
            return "\n".join(files + ["tests/test_other.py", "tests/nested/test_numerics_other.py"])
        assert command == ["git", "rev-parse", "HEAD"] and kwargs["cwd"] == str(tmp_path)
        return prepare_split.SOURCE
    monkeypatch.setattr(prepare_split.subprocess, "check_output", git_output)
    record = {"verdict": "PASS", "before": {"source": "b" * 40, "exit": 1, "failures": 4},
              "after": [dict(source=config["tool_commit"], exit=0, failures=0, errors=0, skips=0, selected=1234, passed=True)
                        for _ in range(2)], "test_files": files[::-1]}
    checks_path.write_text(json.dumps(record))
    return config, record, checks_path


def test_verify_accepts_v2_cpu_check(tmp_path, monkeypatch):
    assert hasattr(prepare_split, "check_cpu_check"), "feature absent: verify accepts the whole package v2 CPU check"
    config, record, checks_path = cpu_campaign(tmp_path, monkeypatch)
    assert "source" not in record
    assert prepare_split.verify(tmp_path) == config
    for arm in record["after"]:
        arm["selected"] = 4321
    checks_path.write_text(json.dumps(record))
    assert prepare_split.verify(tmp_path) == config


@pytest.mark.parametrize("arm_index", [0, 1])
@pytest.mark.parametrize("fault", ["source", "exit", "failures", "errors", "skips", "passed", "zero", "count", "missing-count", "bool-count",
                                   "verdict", "subset", "other-file", "one-arm", "three-arms"])
def test_verify_refuses_invalid_v2_cpu_check(tmp_path, monkeypatch, arm_index, fault):
    assert hasattr(prepare_split, "check_cpu_check"), "feature absent: verify refuses failed arms other sources and partial packages"
    config, record, checks_path = cpu_campaign(tmp_path, monkeypatch)
    assert prepare_split.verify(tmp_path) == config
    arm = record["after"][arm_index]
    if fault == "source":
        arm["source"] = "b" * 40
    elif fault in ("exit", "failures", "errors", "skips"):
        arm[fault] = 1
    elif fault == "passed":
        arm["passed"] = False
    elif fault == "zero":
        arm["selected"] = 0
    elif fault == "count":
        arm["selected"] += 1
    elif fault == "missing-count":
        del arm["selected"]
    elif fault == "bool-count":
        arm["selected"] = True
    elif fault == "verdict":
        record["verdict"] = "FAIL"
    elif fault == "subset":
        record["test_files"].pop()
    elif fault == "other-file":
        record["test_files"].append("tests/test_numerics_other.py")
    elif fault == "one-arm":
        record["after"].pop()
    else:
        record["after"].append(dict(arm))
    checks_path.write_text(json.dumps(record))
    with pytest.raises(Invalid, match="CPU check"):
        prepare_split.verify(tmp_path)
