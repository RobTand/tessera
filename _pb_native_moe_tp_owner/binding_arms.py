"""Two real ranks of a real process group, bound the way a TP2 owner binds.

This is the CPU half of the world-size contract: ``bind_owner_rank`` reads
``torch.distributed``, so it is exercised here on gloo with two real processes
rather than asserted against a stub.  No CUDA, no device, no vLLM.

Arms, per rank:

* its OWN declared pair is accepted, and the rank it returns is the rank
  ``torch.distributed`` gave it;
* the other rank's pair is refused -- a world of two cannot be relabelled;
* a world of one is refused, so a TP2 owner cannot run half a module and call
  it the whole;
* a rank outside the world is refused;
* once the group is gone, no group means no answer.

Exits non-zero on any arm that does not refuse or does not bind.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))
sys.path.insert(0, str(TREE / "src"))

from experiments import bench_native_moe_operator as moe  # noqa: E402


def _pair(rank):
    return {"world_size": 2, "rank": rank, "init_method": "tcp://127.0.0.1:29557",
            "timeout_seconds": 120}


def _arm(rank):
    import torch.distributed as dist
    live = {"world_size": int(dist.get_world_size()), "rank": int(dist.get_rank())}
    bound = list(moe.bind_owner_rank(_pair(rank)))
    refusals = {}
    for name, block in (("other_rank", _pair(1 - rank)),
                        ("world_of_one", {**_pair(rank), "world_size": 1}),
                        ("rank_out_of_world", {**_pair(rank), "rank": 2})):
        try:
            moe.bind_owner_rank(block)
        except ValueError:
            refusals[name] = "refused"
        else:
            refusals[name] = "ACCEPTED"
    return {"rank": rank, "live": live, "bound": bound, "refusals": refusals}


def worker(rank, world_size, port, results):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world_size)
    try:
        results[rank] = _arm(rank)
    except Exception as exc:  # noqa: BLE001 -- the arm reports its own failure
        results[rank] = {"rank": rank, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        dist.destroy_process_group()


def main(port=29557):
    import torch.multiprocessing as mp
    results = mp.Manager().dict()
    mp.spawn(worker, args=(2, port, results), nprocs=2, join=True)
    report = {"schema": "tessera.native_moe_tp_owner_binding.v1",
              "ranks": [results[0], results[1]], "no_group_refused": None}
    for entry in report["ranks"]:
        assert "error" not in entry, entry
        assert entry["bound"] == [entry["rank"], 2], entry
        assert entry["live"] == {"world_size": 2, "rank": entry["rank"]}, entry
        assert all(value == "refused" for value in entry["refusals"].values()), entry
    try:
        moe.bind_owner_rank(_pair(0))
    except ValueError:
        report["no_group_refused"] = True
    else:
        report["no_group_refused"] = False
    assert report["no_group_refused"] is True, report
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 29557))
