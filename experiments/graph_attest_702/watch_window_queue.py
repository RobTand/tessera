"""One code-timed, read-only PB queue decision; never submits or starts a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

PB = Path("/mnt/shared/prismabuild-fleet/repo")
CLIENT = sys.executable


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
    if done.returncode == 0:
        value["decision"] = "defer until census769 terminal and physical owned cleanup" if value["rows"] else "no named census769 half queued/claimed; source-safe window may proceed"
        value["submit_allowed"] = not value["rows"]
    else:
        value["decision"] = "queue view incomplete; do not submit a second pair"
    out.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k != "queue_snapshot"}, sort_keys=True), flush=True)
    return value


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
