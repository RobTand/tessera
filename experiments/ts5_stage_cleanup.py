"""Bounded cleanup for the one-shot, exclusive-GPU LFM campaign stages."""
import subprocess
import time


CLEANUP_TIMEOUT_S = 90.0
MONITOR_JOIN_TIMEOUT_S = 2.0


def cleanup_stage(name, *, launched, completed, stop, monitor,
                  run=subprocess.run, clock=time.monotonic):
    """Observe cleanup, removing only a name this action could have launched.

    The single deadline includes stopping telemetry and all subprocess waits,
    leaving room within the caller's 120-second outer cleanup grace to persist
    the receipt. A timeout or failed inspection is unsafe, never an empty
    process list. Prelaunch refusals do not grant container ownership.
    """
    started = clock()
    deadline = started + CLEANUP_TIMEOUT_S
    result = {"container_name": name, "measurement_completed": completed,
              "launched": launched, "safe_to_release": False,
              "cleanup_timeout_s": CLEANUP_TIMEOUT_S}

    def remaining(cap):
        seconds = deadline - clock()
        if seconds <= 0:
            raise TimeoutError("stage cleanup deadline exhausted")
        return min(cap, seconds)

    def capture(command):
        return run(command, check=True, text=True, capture_output=True,
                   timeout=remaining(30.0)).stdout.strip()

    try:
        stop.set()
        monitor.join(timeout=remaining(MONITOR_JOIN_TIMEOUT_S))
        inspect = ["docker", "ps", "-aq", "--filter", f"name=^/{name}$"]
        existing = capture(inspect)
        result["container_before_cleanup"] = existing
        if existing and launched:
            run(["docker", "rm", "-f", name], check=True,
                timeout=remaining(45.0))
        elif existing:
            result["cleanup_skipped"] = "pre-existing container is not owned by this action"
        result["container_after_cleanup"] = capture(inspect)
        result["gpu_compute_processes"] = capture([
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader"])
        result["safe_to_release"] = (
            not result["container_after_cleanup"] and not result["gpu_compute_processes"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["cleanup_elapsed_s"] = clock() - started
    return result
