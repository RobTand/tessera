"""Prepare F dense split jobs. No command in this module submits a job."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
import shlex
import shutil
import subprocess

from experiments.numerics_rows.common import Invalid, sha256, write_json
from experiments.numerics_rows.prepare_job import stage_population
from experiments.numerics_rows.start_env import container_options, worker_command

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "272507f18de39fc90f03c1588373c031a9e43717"
RULE = Path("/mnt/shared/flux-jobs/tessera-1717-rule/pass-rule-v2.md")
RULE_SHA = "ec8d4b3bbbb8df5f6c1be860bd20204f0f0a14ff3faa26eadc8c5e764fff5d53"
FREEZE = RULE.with_name("freeze-record-v2.json")
MODULE = "model.language_model.layers.32.mlp.shared_experts.down_proj"
INPUTS = ("source", "population", "manifest", "build", "build_provenance", "q1", "h1", "freeze_addendum")


def check_telemetry(path):
    """Require the capture worker's own measurements before job success."""
    meter = json.loads(Path(path).read_text())["row"]["gpu_minutes"]
    fields = ("wall_minutes", "gpu_busy_minutes", "peak_device_memory_mib", "peak_power_watts",
              "mean_sample_power_watts", "whole_job_peak_memory_bytes")
    if (any(type(meter.get(k)) not in (int, float) or not math.isfinite(meter[k]) or meter[k] < 0 for k in fields)
            or not meter["whole_job_peak_memory_bytes"] or not meter["peak_device_memory_mib"]
            or not meter.get("busy_sampled_seconds") or len(meter.get("samples", [])) < 2
            or not any(type(s.get("compute_process_count")) is int and s["compute_process_count"] >= 1 for s in meter.get("samples", []))):
        raise Invalid("capture telemetry is missing compute-process visibility, GPU busy, memory, power or whole-job memory evidence")
    active = [s for s in meter["samples"] if type(s.get("compute_process_count")) is int and s["compute_process_count"] > 0]
    count = max(s["compute_process_count"] for s in active)
    return {k: meter[k] for k in fields} | {"sampled_seconds": meter["sampled_seconds"],
        "busy_sampled_seconds": meter["busy_sampled_seconds"],
        "known_sample_coverage_fraction": meter["busy_sampled_seconds"] / meter["sampled_seconds"] if meter["sampled_seconds"] else None,
        "max_compute_process_count": count, "flag": "count_above_1" if count > 1 else None, "errors": meter["errors"]}


def check_canary(directory, revision, paths):
    directory = Path(directory)
    config = json.loads((directory / "campaign.json").read_text())
    phase_roots = [directory / phase for phase in ("prepare", "capture", "compare")]
    path = directory / "compare/results/numerics_row.json"
    if config.get("templates"):
        phase_roots = [max((directory / "units" / key / "attempts").iterdir(), key=lambda p: int(p.name)) / unit["phase"]
                       for unit in config["templates"] for key in unit.get("unit_keys", [unit["key"]])]
        path = phase_roots[-1] / "numerics_row.json"
    receipt = json.loads(path.read_text())
    if (not config["canary"] or config["tool_commit"] != revision or receipt.get("canary") is not True
            or receipt["tool"].get("commit") != revision or receipt["tessera"].get("commit") != SOURCE
            or receipt["rule"].get("sha256") != RULE_SHA or not receipt["row"].get("finalization_finished")
            or receipt["row"].get("exit_status") != 0 or receipt["row"].get("outcome") != "PASS"
            or not receipt["units"] or any(u["outcome"] != "PASS" for u in receipt["units"])):
        raise Invalid("counted row requires a finished PASS canary at the same tool, F and rule")
    if config["paths"]["source"] != str(paths["source"]) or any(
            config["digests"].get(k) != sha256(v) for k, v in paths.items() if k not in ("source", "manifest")):
        raise Invalid("counted row and canary must use the same F chain")
    for phase_root in phase_roots:
        if (phase_root / "tool-exit.txt").read_text().strip() != "0":
            raise Invalid("canary phase did not exit successfully: " + phase_root.name)
        if phase_root.name == "capture":
            check_telemetry(phase_root / ("numerics_row.json" if config.get("templates") else "results/numerics_row.json"))
    return {"path": str(path), "sha256": sha256(path)}


def row_arguments(config, root, source, population, paths, phase, *, unit=None, attempt=None):
    arguments = ["--image", "N", "--name", config["module"], "--phase", phase,
                 "--phase-instance", str(root / phase / "results/instance.json"),
                 "--out", str(root / phase / "results"), "--rule", str(RULE), "--freeze", str(FREEZE),
                 "--module", config["module"]]
    if attempt is not None:
        arguments[arguments.index("--out") + 1] = str(attempt)
        arguments[arguments.index("--phase-instance") + 1] = str(attempt / phase / "instance.json")
        arguments += ["--phase-local-output", "--case-set", unit["case_set"]]
        if unit.get("size"):
            arguments += ["--tp-size", str(unit["size"])]
    if config["canary"]:
        arguments += ["--canary"] + (["--tp-size", "2"] if attempt is None else [])
    for key, value in paths.items():
        arguments += ["--" + key.replace("_", "-"), str(population) if key == "population" else str(value)]
    if phase == "capture":
        arguments += ["--prepare-artifact", str(attempt / "prepare" if attempt is not None else root / "prepare/results/prepare")]
    if phase == "compare":
        for prepared, captured in config.get("phase_artifacts", [(root / "prepare/results/prepare", root / "capture/results/capture")]):
            arguments += ["--prepare-artifact", str(prepared), "--capture-artifact", str(captured)]
    return worker_command(source, "experiments.numerics_rows.row", arguments)


def bind(root, paths):
    root = Path(root)
    config = json.loads((root / "campaign.json").read_text())
    if set(paths) != set(INPUTS) or any(not str(p).startswith("/mnt/shared/") or not Path(p).exists() for p in paths.values()):
        raise Invalid("bind each F chain input from an existing shared path")
    source = Path(paths["source"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != SOURCE:
        raise Invalid("split row source is not F")
    if sha256(RULE) != RULE_SHA:
        raise Invalid("split row rule is not frozen v2")
    from experiments.numerics_rows.serving_build import check_retained_source
    check_retained_source(source, Path(paths["build"]), Path(paths["build_provenance"]))
    from experiments.numerics_rows.row import bind_freeze_addendum, check_rule_freeze
    from experiments.numerics_rows.launch_state import LEVERS
    freeze = bind_freeze_addendum(json.loads(FREEZE.read_text()), FREEZE, RULE, paths["freeze_addendum"])
    check_rule_freeze(RULE, freeze, canary=config["canary"], launch_levers_sha256=sha256(LEVERS))
    if not config["canary"]:
        config["canary_evidence"] = check_canary(config["canary_directory"], config["tool_commit"], paths)
    population, mount = stage_population(paths["population"], root)
    config.update(paths={k: str(v) for k, v in paths.items()}, digests={k: sha256(v) for k, v in paths.items() if k != "source"},
                  staged_population=str(population), population_mount=mount, bound=True)
    write_json(root / "campaign.json", config)
    bind_templates(root, config)
    return root


def bind_templates(root, config):
    paths, population = config["paths"], config["staged_population"]
    source = Path(paths["source"])
    for unit in config["templates"]:
        directory, phase = root / "templates" / unit["key"] / unit["phase"], unit["phase"]
        if "unit_keys" in unit:
            commands = [{"command": shlex.split(row_arguments(config, root, source, population, paths, phase,
                        unit=member, attempt=Path("__ATTEMPT__")))} for member in unit["members"]]
            write_json(directory / "commands.json", commands)
            command = worker_command(source, "experiments.numerics_rows.capture_group",
                                     [len(commands), directory / "commands.json", "--claimed"])
            (directory / "payload.sh").write_text(f'#!/bin/bash\nset -euo pipefail\nexec {command} "$@"\n')
            seal_template(directory)
            continue
        local = dict(config)
        prefix = ""
        if phase == "compare":
            pairs = []
            for index, key in enumerate(config["unit_keys"]):
                prefix += f"pair_{index}=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \"$attempt/compare-inputs.json\" {shlex.quote(key)})\n"
                pairs.append([Path(f"__PAIR_{index}__") / p for p in ("prepare", "capture")])
            local["phase_artifacts"] = pairs
        payload = row_arguments(local, root, source, population, paths, phase, unit=unit, attempt=Path("__ATTEMPT__"))
        payload = payload.replace("__ATTEMPT__", '"$attempt"')
        for index in range(len(config["unit_keys"])):
            payload = payload.replace(f"__PAIR_{index}__", f'"$pair_{index}"')
        (directory / "payload.sh").write_text(f'#!/bin/bash\nset -euo pipefail\nattempt=$1\n{prefix}exec {payload}\n')
        seal_template(directory)


def check_cpu_check(checks, revision, test_files):
    arms, files = checks.get("after"), checks.get("test_files")
    if (not isinstance(files, list) or any(not isinstance(name, str) for name in files)
            or not test_files or set(files) != set(test_files)):
        raise Invalid("CPU check test_files differs from the tool commit numerics package")
    if (checks.get("verdict") != "PASS" or not isinstance(arms, list) or len(arms) != 2
            or any(not isinstance(arm, dict) or arm.get("passed") is not True
                   or arm.get("source") != revision
                   or type(arm.get("selected")) is not int or arm["selected"] <= 0
                   or arm["selected"] != arms[0].get("selected")
                   or any(type(arm.get(field)) is not int or arm[field] != value
                          for field, value in (("exit", 0), ("failures", 0), ("errors", 0), ("skips", 0)))
                   for arm in arms)):
        raise Invalid("split row requires two passing CPU check arms at its tool commit and selected count")


def verify(root):
    config = json.loads((root / "campaign.json").read_text())
    if not config.get("bound") or config["source_revision"] != SOURCE or config["rule_sha256"] != RULE_SHA:
        raise Invalid("split row needs its F chain and lever freeze addendum")
    if sha256(RULE) != RULE_SHA or sha256(FREEZE) != config["freeze_sha256"]:
        raise Invalid("frozen v2 rule or record changed")
    checks = json.loads(Path(config["cpu_check"]).read_text())
    tree = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", config["tool_commit"], "--", "tests"],
                                   cwd=ROOT, text=True).splitlines()
    check_cpu_check(checks, config["tool_commit"], [name for name in tree if re.fullmatch(r"tests/test_numerics_[^/]+\.py", name)])
    for name, digest in config["digests"].items():
        if sha256(config["paths"][name]) != digest:
            raise Invalid("split row bound input changed: " + name)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config["paths"]["source"], text=True).strip() != SOURCE:
        raise Invalid("split row source moved away from F")
    if not config["canary"] and check_canary(config["canary_directory"], config["tool_commit"], config["paths"]) != config["canary_evidence"]:
        raise Invalid("counted row canary evidence changed")
    return config


def seal_template(directory):
    excluded = {"template.sha256", "job.sh", "table.json"}
    files = sorted(p for p in directory.iterdir() if p.name not in excluded)
    root = directory.parents[2]
    files += [root / name for name in ("instance.py", "campaign.json", "feeder-settings.json")]
    (directory / "template.sha256").write_text("".join(f"{sha256(p)}  {p.name if p.parent == directory else p}\n" for p in files))


def capture_templates(units, groups, table, *, module):
    """Fix group membership and shape before the generator writes a template."""
    from experiments.numerics_rows.launch_state import json_digest
    members = {unit["key"]: unit for unit in units}
    seen, keys, result = set(), set(members), []
    for group in groups:
        key, ordered = group["key"], group["unit_keys"]
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", key) or key in keys or key.startswith("compare-"):
            raise Invalid("capture group key is invalid or repeated")
        if not isinstance(ordered, list) or len(ordered) < 2 or any(item not in members for item in ordered):
            raise Invalid("capture group requires at least two known unit keys")
        if len(set(ordered)) != len(ordered) or seen.intersection(ordered):
            raise Invalid("unit key appears in two capture templates")
        if (any(type(table.get(field)) is not int or table[field] < 1 for field in ("cores_per_capture", "max_k"))
                or any(type(table.get(field)) not in (int, float) or not math.isfinite(table[field]) or table[field] <= 0
                       for field in ("anon_peak_gib_per_capture", "margin_gib"))):
            raise Invalid("capture group canary table is unset or invalid")
        k = len(ordered)
        if k > table["max_k"]:
            raise Invalid("capture group K exceeds canary admission")
        cores, memory = k * table['cores_per_capture'], math.ceil(k * table['anon_peak_gib_per_capture'] + table['margin_gib'])
        shape = f"slot=1/[core={cores};memory={memory};gpu=1]"
        resources = [{"type": "slot", "count": 1, "label": "task", "with":
                      [{"type": "core", "count": cores}, {"type": "memory", "count": memory}, {"type": "gpu", "count": 1}]}]
        result.append(dict(group, key="group-" + json_digest([module, ordered]), phase="capture",
                           members=[members[item] for item in ordered], shape=shape, resources=resources))
        seen.update(ordered)
        keys.add(key)
    return [dict(unit, phase=phase) for unit in units for phase in ("prepare", "capture")
            if phase != "capture" or unit["key"] not in seen] + result


def write_templates(root, config, repository):
    from experiments.numerics_rows.launch_state import json_digest
    settings = json.loads((root / "feeder-settings.json").read_text())
    cap = settings["max_attempts"]
    if type(cap) is not int or not 1 <= cap <= 999999:
        raise Invalid("max_attempts must be an integer from 1 through 999999")
    units = [{"key": f"tp{size}-{part}-" + json_digest([config["module"], size, part]), "size": size, "case_set": part}
             for size in (1, 2) for part in ("gate", "diagnostics")]
    config["unit_keys"] = [unit["key"] for unit in units]
    config["templates"] = capture_templates(units, config.get("capture_groups", []), settings["capture_group_canary"], module=config["module"])
    config["templates"].append({"key": "compare-" + json_digest([config["module"], "compare"]), "phase": "compare", "case_set": "all"})
    write_json(root / "campaign.json", config)
    options = shlex.join(container_options("$task_scratch")).replace("'$task_scratch", '\"$task_scratch').replace(":rw'", ':rw\"').replace(":ro'", ':ro\"')
    for unit in config["templates"]:
        phase, key = unit["phase"], unit["key"]
        directory = root / "templates" / key / phase
        directory.mkdir(parents=True)
        parts = [member["case_set"] for member in unit["members"]] if "members" in unit else [unit["case_set"]]
        minutes = max(settings["phase_minutes"][phase][part] for part in parts)
        if type(minutes) is not int or minutes <= 1:
            raise Invalid("phase duration must be an integer greater than one minute")
        ordered = unit.get("unit_keys", [key])
        instances = {}
        for name in settings["template_instances"]["canary" if config["canary"] else phase]:
            hosts = ["sparky", "sparklina"] if name == "fv1" else [name.removeprefix("fv1e-")]
            memory = settings["fv1_prepare_memory_gib" if name == "fv1" else "fv1e_prepare_memory_gib"] if phase == "prepare" else 32
            resources = unit.get("resources", [{"type": "slot", "count": 1, "label": "task", "with":
                                 [{"type": "core", "count": 4 if config["canary"] and phase == "capture" else 1},
                                  {"type": "memory", "count": memory}] + ([{"type": "gpu", "count": 1}] if phase == "capture" else [])}])
            instances[name] = {"constraints": [{"hostlist": [host]} for host in hosts],
                               "shapes": [resources], "max_duration_s": minutes * 60}
        table = {"schema": 1, "instances": instances,
                 "job_name": f"tessera-1717-{key}-{phase}", "args": len(ordered),
                 "max_attempts": cap, "urgency": settings["urgency"], "cwd": str(directory),
                 "env_names": ["PATH", "HOME", "XDG_RUNTIME_DIR", "FLUX_F58_FORCE_ASCII", "FLUX_SCRATCH",
                               "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]}
        write_json(directory / "table.json", table)
        prerequisites = ""
        if phase == "capture":
            prerequisites = ('python3 ' + shlex.quote(str(root / "instance.py")) + ' complete "$attempt/prepare" || refuse \'prepare is not complete or hashes differ\'\n'
                             '(cd "$attempt/prepare" && sha256sum -c completion.sha256) || refuse \'prepare is not complete or hashes differ\'')
        values = {"ROOT": shlex.quote(str(root)), "ROOT_RAW": str(root), "TEMPLATE": shlex.quote(str(directory)),
                  "REPOSITORY": shlex.quote(str(repository)), "REVISION": config["tool_commit"], "PHASE": phase,
                  "COUNT": str(len(ordered)), "COUNT_MESSAGE": "require one attempt argument" if len(ordered) == 1 else f"require {len(ordered)} attempt arguments",
                  "PATTERNS": shlex.join(["^" + re.escape(str(root / "units" / item / "attempts")) + "/([1-9][0-9]*)$" for item in ordered]),
                  "ABSENT": 'test ! -e "$attempt/capture" && test ! -L "$attempt/capture" || refuse \'phase claim already exists or failed\'' if len(ordered) > 1 else "",
                  "PREREQUISITES": prerequisites,
                  "SECONDS": str(minutes * 60 - 60), "OPTIONS": options,
                  "SCRATCH_SUBDIR": "" if phase == "capture" else "/work",
                  "CPU_LAUNCH": '' if phase == "capture" else 'python3 "$task_scratch/code/experiments/numerics_rows/cdi_env.py" --scratch "$task_root" --record "$task_root/cdi.json" -- ',
                  "GPU": "--device nvidia.com/gpu=all" if phase == "capture" else "",
                  "PREFLIGHT": 'PYTHONPATH="$task_scratch/code" python3 -m experiments.numerics_rows.gpu_time --preflight gpu-preflight-host.json' if phase == "capture" else "",
                  "TELEMETRY": 'PYTHONPATH="$task_scratch/code" python3 -m experiments.numerics_rows.prepare_split --check-telemetry numerics_row.json > telemetry-check.json' if phase == "capture" else ""}
        script = (Path(__file__).parent / "templates" / "job.sh").read_text()
        for marker, value in values.items():
            script = script.replace("@" + marker + "@", value)
        (directory / "job.sh").write_text(script)
        (directory / "job.sh").chmod(0o700)
        seal_template(directory)
    write_json(root / "campaign.json", config)


def prepare(cpu_check, *, module=MODULE, canary=True, canary_directory=None, output_root=None):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "CANARY" if canary else module.replace("model.language_model.layers.", "L").replace(".mlp.", "-").replace(".", "-")
    out = Path(output_root) if output_root is not None else Path(f"/mnt/shared/flux-jobs/tessera-1717-SPLIT-{label}-N-{revision[:8]}-{stamp}")
    out.mkdir()
    out = out.resolve()
    repository = Path(canary_directory) / "repository.git" if canary_directory else out / "repository.git"
    if not canary_directory:
        subprocess.run(["git", "clone", "--bare", "--no-hardlinks", str(ROOT), str(repository)], check=True)
    write_json(out / "campaign.json", {"tool_commit": revision, "source_revision": SOURCE, "rule": str(RULE),
        "rule_sha256": RULE_SHA, "freeze": str(FREEZE), "freeze_sha256": sha256(FREEZE),
        "module": module, "cuts": [{"size": 1, "rank": 0}, {"size": 2, "rank": 0}, {"size": 2, "rank": 1}],
        "canary": canary, "bound": False, "bookings_provisional": not canary,
        "canary_directory": str(canary_directory) if canary_directory else None,
        "paths": dict.fromkeys(INPUTS), "cpu_check": str(cpu_check)})
    shutil.copyfile(Path(__file__).with_name('phase_instance.py'), out / 'instance.py')
    shutil.copyfile(ROOT / 'experiments/numerics_rows/feeder-settings.json', out / 'feeder-settings.json')
    write_templates(out, json.loads((out / "campaign.json").read_text()), repository)
    (out / "bind-inputs.py").write_text(f'''import argparse, json, sys
from pathlib import Path
import subprocess
if not Path({str(out / 'binding-code')!r}).exists():
    subprocess.run(["git", "clone", "--shared", {str(repository)!r}, {str(out / 'binding-code')!r}], check=True)
subprocess.run(["git", "-C", {str(out / 'binding-code')!r}, "checkout", "--detach", {revision!r}], check=True)
sys.path.insert(0, {str(out / 'binding-code')!r})
from experiments.numerics_rows.prepare_split import bind
p=argparse.ArgumentParser()
p.add_argument("paths", help="JSON object with the eight F chain paths")
a=p.parse_args()
bind({str(out)!r}, json.load(open(a.paths)))
''')
    (out / "README.md").write_text(f"""# Split dense jobs

Prepared only. Tool {revision}; Tessera {SOURCE}.
Module: {module}. Canary: {canary}.
Bind the eight F chain inputs with bind-inputs.py.
Require a PASS CPU check at this tool commit: {cpu_check}.
Require Fleet review before submission.
Read each Flux verdict, exit status and logs.
Only compare can finish the row.
""")
    print(out)
    return out


def prepare_counted(cpu_check, population, canary_directory):
    from experiments.numerics_rows.population import load_signed_population
    signed, _ = load_signed_population(population)
    paths = {entry["name"]: str(prepare(cpu_check, module=entry["name"], canary=False,
                                      canary_directory=canary_directory)) for entry in signed["modules"]}
    write_json(Path(canary_directory) / "counted-rows.json", paths)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--check-telemetry", type=Path)
    parser.add_argument("--cpu-check", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--module", default=MODULE)
    parser.add_argument("--counted-population", type=Path)
    parser.add_argument("--canary-directory", type=Path)
    args = parser.parse_args()
    if args.check_telemetry:
        print(json.dumps(check_telemetry(args.check_telemetry), allow_nan=False))
    elif args.verify:
        verify(args.verify)
    elif args.cpu_check and args.counted_population and args.canary_directory and args.output_root is None:
        prepare_counted(args.cpu_check, args.counted_population, args.canary_directory)
    elif args.cpu_check and not (args.counted_population or args.canary_directory):
        prepare(args.cpu_check, module=args.module, output_root=args.output_root)
    else:
        parser.error("prepare requires --cpu-check")
