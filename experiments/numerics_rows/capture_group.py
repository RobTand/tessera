"""Run K capture processes within one admitted GPU job. Submit nothing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from experiments.numerics_rows.common import Invalid, write_json


def capture_group(k, units):
    if type(k) is not int or k < 1 or len(units) != k:
        raise Invalid("K must equal the number of capture unit commands")
    outputs = [str(Path(u["out"]).resolve()) for u in units]
    if len(set(outputs)) != k or any(Path(p).exists() for p in outputs):
        raise Invalid("each capture process requires a new output directory")
    for item in units:
        command = item["command"]
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
            raise Invalid("each command must select the capture phase")
        if any(command.count(flag) != 1 or command[command.index(flag) + 1:command.index(flag) + 2] != [value]
               for flag, value in (("--phase", "capture"), ("--out", item["out"]))) or "--capture-group-proof" in command:
            raise Invalid("each command must select the capture phase")
    from experiments.numerics_rows.gpu_time import preflight
    proof = {"parent_pid": os.getpid(), "out": outputs, "preflight": preflight()}
    if proof["preflight"]["verdict"] != "FREE":
        raise Invalid("capture group requires a free GPU before all processes")
    def run_unit(item):
        target = Path(item["out"])
        target.mkdir(parents=True)
        write_json(target / "slot.json", proof)
        command = item["command"] + ["--capture-group-proof", str(target / "slot.json")]
        with (target / "stdout").open("w") as output, (target / "stderr").open("w") as errors:
            status = subprocess.run(command, stdout=output, stderr=errors).returncode
        write_json(target / "exit.json", {"exit_status": status})
        return status
    with ThreadPoolExecutor(max_workers=k) as workers:
        statuses = list(workers.map(run_unit, units))
    return max(statuses, key=lambda value: value != 0)


def slot_preflight(path, out):
    proof = json.loads(Path(path).read_text())
    if proof["parent_pid"] != os.getppid() or str(Path(out).resolve()) not in proof["out"] or proof["preflight"]["verdict"] != "FREE":
        raise Invalid("capture process has no live parent slot preflight")
    return proof["preflight"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("k", type=int)
    parser.add_argument("units", type=Path)
    options = parser.parse_args()
    raise SystemExit(capture_group(options.k, json.loads(options.units.read_text())))
