"""Run K capture processes within one admitted GPU job. Submit nothing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
from threading import Lock

from experiments.numerics_rows.common import Invalid, write_json


def capture_group(k, units, *, claimed=False):
    if type(k) is not int or k < 1 or len(units) != k:
        raise Invalid("K must equal the number of capture unit commands")
    outputs = [str((Path(u["out"]) / "capture" if claimed else Path(u["out"])).resolve()) for u in units]
    if len(set(outputs)) != k or any(not Path(p).is_dir() if claimed else Path(p).exists() for p in outputs):
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
    children, lock, stopped = {}, Lock(), []
    def stop(signum, _frame):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        with lock:
            stopped.append(signum)
            for process in children.values():
                if process.poll() is None:
                    process.terminate()
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)} if claimed else {}
    def run_unit(item):
        target = Path(item["out"]) / "capture" if claimed else Path(item["out"])
        if not claimed:
            target.mkdir(parents=True)
        write_json(target / "slot.json", proof)
        command = item["command"] + ["--capture-group-proof", str(target / "slot.json")]
        with (target / "stdout").open("w") as output, (target / "stderr").open("w") as errors:
            if claimed:
                with lock:
                    process = None if stopped else subprocess.Popen(command, stdout=output, stderr=errors)
                    if process is not None:
                        children[item["out"]] = process
                status = process.wait() if process is not None else 128 + stopped[0]
            else:
                status = subprocess.run(command, stdout=output, stderr=errors).returncode
        write_json(target / "exit.json", {"exit_status": status})
        return status
    try:
        with ThreadPoolExecutor(max_workers=k) as workers:
            statuses = list(workers.map(run_unit, units))
        return 128 + stopped[0] if stopped else max(statuses, key=lambda value: value != 0)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def slot_preflight(path, out):
    proof = json.loads(Path(path).read_text())
    if proof["parent_pid"] != os.getppid() or str(Path(out).resolve()) not in proof["out"] or proof["preflight"]["verdict"] != "FREE":
        raise Invalid("capture process has no live parent slot preflight")
    return proof["preflight"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("k", type=int)
    parser.add_argument("units", type=Path)
    parser.add_argument("--claimed", nargs="+", help="Use the claimed capture directories for these ordered attempts")
    options = parser.parse_args()
    commands = json.loads(options.units.read_text())
    if options.claimed:
        if len(options.claimed) != options.k or len(commands) != options.k:
            parser.error("K must equal the number of attempts and commands")
        commands = [{"out": _attempt, "command": [_argument.replace("__ATTEMPT__", _attempt) for _argument in _command["command"]]}
                    for _attempt, _command in zip(options.claimed, commands, strict=True)]
    raise SystemExit(capture_group(options.k, commands, claimed=bool(options.claimed)))
