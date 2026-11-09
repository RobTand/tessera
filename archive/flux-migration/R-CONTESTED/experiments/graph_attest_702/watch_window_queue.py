"""One code-timed, read-only PB queue decision; never submits or starts a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import re
import json
from pathlib import Path
import subprocess
import sys
import time

PB = Path("/mnt/shared/prismabuild-fleet/repo")
CLIENT = sys.executable
REQUESTS = Path("/mnt/shared/prismabuild-fleet/cas/requests")


def inspect(at: str, keys: list[str], out: Path) -> dict:
    deadline = datetime.fromisoformat(at).timestamp()
    entered_unix = time.time()
    time.sleep(max(0, deadline - time.time()))
    observed = datetime.now(timezone.utc).isoformat()
    done = subprocess.run([CLIENT, str(PB / "tools/pbstatus.py"), "--transport", "pool", "--json",
                           "--recent", "0", "--timeout-s", "10"], capture_output=True, text=True, timeout=20)
    value = dict(decision_at=at, read_started_at=observed, command_returncode=done.returncode,
                 queue_snapshot=json.loads(done.stdout), rows=[], submit_allowed=False,
                 entered_after_requested_time=entered_unix > deadline, observed_lateness_s=time.time() - deadline,
                 scope="read-only observation, not admission, source review or physical cleanup")
    # The published public reader supplies the exact named READY/CLAIMED rows, not a roster or name match.
    sys.path.insert(0, str(PB / "src"))
    from prismabuild.pool import PoolQueue, read_queue_record
    queue = PoolQueue("/mnt/shared/prismabuild-fleet/pb-queue")
    for key in keys:
        for state in ("ready", "claimed"):
            row = read_queue_record(queue.item_path(state, key))
            if row is not None:
                if row.get("action_key") != key:
                    raise RuntimeError("queue row does not bind the requested full action key")
                value["rows"].append(dict(state=state, action_key=key, row=row))
    snapshot = value["queue_snapshot"]
    value["complete"] = (snapshot.get("complete") is True and snapshot.get("pool", {}).get("complete") is True
                         and not snapshot.get("timed_out_sections") and not snapshot.get("unavailable_sections")
                         and not snapshot.get("pool", {}).get("unreadable"))
    if done.returncode == 0 and value["complete"]:
        value["decision"] = "defer until census769 terminal and physical owned cleanup" if value["rows"] else "no named census769 half queued/claimed; source-safe window may proceed"
        value["submit_allowed"] = not value["rows"]
    else:
        value["decision"] = "queue view incomplete; do not submit a second pair"
    out.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k != "queue_snapshot"}, sort_keys=True), flush=True)
    return value


def other_windows(observed):
    """Read managed-pair ownership, not an obsolete measurement supply cap."""
    if observed.get("complete") is not True:
        raise RuntimeError("paired-window isolation needs a complete queue snapshot")
    sys.path.insert(0, str(PB / "src"))
    from prismabuild.core import validate_action
    found = []
    for row in observed["queue_snapshot"]["jobs"]:
        if row.get("state") not in ("READY", "CLAIMED"):
            raise RuntimeError("paired-window census has an unreadable live row")
        key = row["action_key"]
        path = REQUESTS / key[:2] / (key + ".json")
        request = validate_action(json.loads(path.read_bytes()))
        if request["action_key"] != key:
            raise RuntimeError("paired-window census request does not bind its key")
        command = request["task"]["argv"]
        if any(Path(word).name == "rank_window.py" for word in command) and "--role-preflight" not in command:
            variables = request["environment"]["variables"]
            rank_slot = command.index("--rank") + 1 if "--rank" in command else len(command)
            run_slot = command.index("--run") + 1 if "--run" in command else len(command)
            if (request["task"]["task_class"] != "measurement" or rank_slot >= len(command) or run_slot >= len(command)
                    or command[rank_slot] not in ("0", "1")
                    or re.fullmatch("[a-f0-9]{64}", variables.get("GRAPH_WINDOW_INPUT_SHA256", "")) is None):
                raise RuntimeError("paired-window live row has invalid sealed rank/input ownership")
            found.append(dict(action_key=key, state=row["state"], request_path=str(path), run=command[run_slot]))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", required=True)
    ap.add_argument("--key", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    value = inspect(args.at, args.key, args.out)
    return 0 if value["command_returncode"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
