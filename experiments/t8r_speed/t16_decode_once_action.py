"""Supervise the T16 screen or tests with one memory guard."""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GIB = 1 << 30
# These peaks come from the prior full-shape screen, not this new job.
PRIOR = {"host_rss_bytes": 2090377216, "cuda_allocated_bytes": 2732211200,
         "cuda_reserved_bytes": 2977955840, "cgroup_peak_bytes": 4718301184,
         "receipt": "/mnt/shared/prismabuild-fleet/cas/actions/v3/35/3585668db32cca0aa2eae7ae11873df430d38a5e83017a0ca7a06d8afec0520e.json"}


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def mem_available():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024


def start_estimate(cpu):
    scratch = 12576 * 4096 * 2
    # Account for the separate same-wire control and the larger activation/output.
    added = 2 * scratch + (4096 - 2048) * (4096 + 12576) * 2
    baseline = max(PRIOR["host_rss_bytes"] + PRIOR["cuda_reserved_bytes"], PRIOR["cgroup_peak_bytes"])
    need = PRIOR["host_rss_bytes"] if cpu else baseline + added
    return {"kind": "first_run_estimate", "prior": PRIOR,
            "new_shared_scratch_bytes": 0 if cpu else scratch,
            "experiment_control_bytes": 0 if cpu else scratch,
            "incremental_activation_output_bytes": 0 if cpu else added - 2 * scratch,
            "baseline_bytes": PRIOR["host_rss_bytes"] if cpu else baseline,
            "margin_bytes": 3 * GIB, "required_bytes": need + 3 * GIB,
            "limit": "The prior receipt does not isolate driver overhead. This estimate is not a measured peak."}


def cgroup_memory():
    try:
        group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / group.lstrip("/")
        values = {}
        for name in ("memory.current", "memory.peak", "memory.events"):
            raw = (root / name).read_text().strip()
            values[name] = dict(line.split() for line in raw.splitlines()) if name.endswith("events") else int(raw)
        return {"path": str(root), **values}
    except (OSError, StopIteration, ValueError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def owned_container(out, token, evidence):
    cidfile = out / "owned.cid"
    if not cidfile.exists():
        return None
    cid = cidfile.read_text().strip()
    if len(cid) != 64 or any(c not in "0123456789abcdef" for c in cid):
        evidence.append({"cleanup_error": "The owned container identifier is invalid."})
        return None
    try:
        inspected = subprocess.run(["docker", "inspect", "--format", "{{json .Config.Labels}}", cid],
                                   capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        evidence.append({"cleanup_error": f"{type(error).__name__}: {error}"})
        return None
    if inspected.returncode:
        evidence.append({"container": cid, "inspect_exit": inspected.returncode,
                         "stderr": inspected.stderr, "state": "The container is absent or the daemon cannot inspect it."})
        return None
    if json.loads(inspected.stdout).get("tessera.t16_owner") != token:
        raise RuntimeError("The cleanup target does not belong to this action.")
    return cid


def terminate(child, out, token, cpu, evidence):
    """Terminate only this process group and its labelled container."""
    cid = None if cpu else owned_container(out, token, evidence)
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if cid is not None:
        try:
            stopped = subprocess.run(["docker", "stop", "--time", "10", cid],
                                     capture_output=True, text=True, timeout=12)
            evidence.append({"container": cid, "stop_exit": stopped.returncode, "stderr": stopped.stderr})
            if stopped.returncode:
                raise RuntimeError("Docker did not confirm the owned container stop.")
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
            evidence.append({"container": cid, "stop_error": f"{type(error).__name__}: {error}"})
            killed = subprocess.run(["docker", "kill", "--signal", "KILL", cid],
                                    capture_output=True, text=True, timeout=5)
            evidence.append({"container": cid, "kill_exit": killed.returncode, "stderr": killed.stderr})
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        evidence.append({"process_group": child.pid, "signal": "SIGKILL"})


def parse_action_args(argv):
    if not argv:
        raise ValueError("Supply the output directory and action arguments.")
    out, args = Path(argv[0]).resolve(), list(argv[1:])
    if "--tests" not in args:
        return out, args, "--cpu-preflight" in args, False
    if "--" not in args:
        raise ValueError("The test mode requires a -- separator before pytest arguments.")
    separator = args.index("--")
    flags, tests = args[:separator], args[separator + 1:]
    if any(flag not in {"--tests", "--cpu-preflight", "--collect-only"} for flag in flags):
        raise ValueError("The test mode accepts only test and CPU collection flags before --.")
    if not any(item.split("::", 1)[0].endswith(".py") for item in tests):
        raise ValueError("Select an explicit test file. The full suite is not this action.")
    cpu = "--cpu-preflight" in flags or "--collect-only" in flags
    return out, tests, cpu, True


def uses_xdist(args):
    return any(arg == "--numprocesses" or arg.startswith("--numprocesses=")
               or arg == "-n" or arg.startswith("-n") and arg[2:].isdigit() for arg in args)


def test_preflight(args, runner_sp):
    """Read the selected inputs and the existing pure Python runner."""
    if not runner_sp:
        raise ValueError("TEST_RUNNER_SP must name the existing pure Python runner.")
    runner = Path(runner_sp).resolve()

    def small_read(path):
        with path.open("rb") as handle:
            data = handle.read(4096)
        if not data:
            raise ValueError(f"The preflight input is empty: {path}")
        return {"path": str(path), "bytes_read": len(data)}

    tests = []
    for item in args:
        name = item.split("::", 1)[0]
        if name.endswith(".py"):
            path = Path(name)
            tests.append(small_read(path.resolve() if path.is_absolute() else (ROOT / path).resolve()))
    packages = ["pytest", "_pytest", "pluggy", "iniconfig", "packaging"]
    if uses_xdist(args):
        packages += ["xdist", "execnet"]
    reads = [small_read(runner / package / "__init__.py") for package in packages]
    reads.append(small_read(runner / "py.py"))
    return {"test_reads": tests, "runner_reads": reads,
            "wrapper_read": small_read(ROOT / "experiments/routed_fused_tests.sh"),
            "cuda_executed": False, "collection_required": True}


def main():
    out, args, cpu, tests = parse_action_args(sys.argv[1:])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "memory_guard.json").exists() or (out / "owned.cid").exists():
        raise RuntimeError("Use a new output directory. Preserve earlier runs.")
    estimate = start_estimate(cpu)
    available = mem_available()
    safety = {"estimate": estimate, "start_available_bytes": available,
              "mode": "test_collection" if tests and cpu else "tests" if tests else "benchmark",
              "abort_below_bytes": 2 * GIB, "minimum_available_bytes": available,
              "term_then_kill_seconds": 10, "guard_triggered": False,
              "timeout_seconds": 1680, "start_unix": time.time(), "cleanup": [],
              "cgroup_start": cgroup_memory()}
    guard = out / "memory_guard.json"
    save(guard, safety)
    if available < estimate["required_bytes"]:
        safety.update(exit_code=2, end_unix=time.time(), refusal="The host lacks the D30 start margin.")
        save(guard, safety)
        raise RuntimeError(safety["refusal"])
    token = secrets.token_hex(16)
    environment = dict(os.environ, T16_OWNER_TOKEN=token)
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("HOST_NAME", os.uname().nodename)
    environment.setdefault("PB_ACTION_KEY", os.environ.get("PRISMABUILD_ACTION_KEY", ""))
    if tests:
        environment["PYTHONPATH"] += os.pathsep + str(ROOT / "tests") + os.pathsep + str(ROOT / "experiments")
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        if uses_xdist(args):
            environment["TEST_XDIST"] = "1"
        if cpu:
            safety["cpu_test_preflight"] = test_preflight(args, environment.get("TEST_RUNNER_SP"))
            environment["PYTHONPATH"] += os.pathsep + environment["TEST_RUNNER_SP"]
            plugins = ["-p", "xdist.plugin"] if uses_xdist(args) else []
            command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *plugins,
                       "--collect-only", "-q", "-rA", f"--junitxml={out / 'collection.xml'}", *args]
            save(guard, safety)
        else:
            command = ["bash", str(HERE / "t16_decode_once.sh"), str(ROOT), str(out), "--tests", *args]
    elif cpu:
        environment.setdefault("TESSERA_HEAD", subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip())
        command = [sys.executable, str(HERE / "bench_t16_decode_once.py"), "--out", str(out), *args]
    else:
        command = ["bash", str(HERE / "t16_decode_once.sh"), str(ROOT), str(out), *args]
    interrupted = []

    def on_signal(number, _frame):
        interrupted.append(number)

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    child = None
    rc = 1
    try:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, start_new_session=True)
        started = time.monotonic()
        while True:
            try:
                rc = child.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                available = mem_available()
                safety["minimum_available_bytes"] = min(safety["minimum_available_bytes"], available)
                low = available < safety["abort_below_bytes"]
                expired = time.monotonic() - started >= safety["timeout_seconds"]
                if not (low or expired or interrupted):
                    continue
                safety.update(guard_triggered=low, timeout_triggered=expired, signals=interrupted)
                terminate(child, out, token, cpu, safety["cleanup"])
                rc = 137 if low else 124 if expired else 128 + interrupted[0]
                break
    finally:
        if child is not None and child.poll() is None:
            terminate(child, out, token, cpu, safety["cleanup"])
        for number, handler in previous.items():
            signal.signal(number, handler)
        safety.update(end_unix=time.time(), exit_code=rc, cgroup_end=cgroup_memory())
        save(guard, safety)
        (out / "action-window.txt").write_text(f"{safety['start_unix']} {safety['end_unix']}\n")
    if not cpu:
        # Reuse the prior both-Spark instrument and retain its actual exit status.
        try:
            netdata = subprocess.run([sys.executable, str(HERE / "routed_gate_netdata.py"), str(out)],
                                     cwd=ROOT, check=False, timeout=45)
            safety["netdata_exit_code"] = netdata.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            safety["netdata_error"] = f"{type(error).__name__}: {error}"
        safety["energy_status"] = "HOLD_pending_coverage_and_instrument_agreement"
        save(guard, safety)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
