#!/usr/bin/env python3
"""Run the suite on the merge result, on both device populations, once.

tessera#112: master was red on three CUDA-gated tests while GitHub Actions,
the x86 pool suite and a local CPU run all read green.  Not one of those three
signals could have seen the failure, and none of them said so.  The CUDA-gated
surface -- some 450-480 tests -- was exercised by nothing automatic.

Two facts shape what this tool is.

*A suite result is meaningless without its population.*  "1404 passed / 487
skipped" and "1381 passed / 497 skipped" are not two measurements of the same
thing; they are two different populations, and quoting either alone is how a
red commit reads green.  So this submits **both** arms and writes **one**
receipt holding both, side by side.  There is no way to quote one from it
without the other being on the same screen.

*The GPU arm must not be able to lie.*  It runs with ``--strict-cuda``, which
refuses a session that has no CUDA device instead of skipping the surface it
was submitted to cover.  A placement that lands the GPU arm on a box without a
device now fails loudly; before, it would have returned a green tick.

Everything about scheduling is PrismaBuild's: this composes ``pbrun``
invocations and reads what they return.  It never runs a suite itself, never
sshes anywhere, and a refused placement comes back as the refusal it is.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PBRUN = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py")
SHARED_ROOT = Path("/mnt/shared")
#: Surface reports are written OUTSIDE the checkout on purpose.  pbrun binds a
#: command's identity to the checkout's git delta, so an artefact dropped into
#: the tree moves the action key of every later submission from it -- a cache
#: miss dressed up as a different action.
DEFAULT_RECEIPT_ROOT = SHARED_ROOT / "tessera-suite-receipts"

#: The two arms, and why each is spelled the way it is.  The interpreter is
#: named rather than inherited: a pool action runs in a sealed environment, so
#: ``sys.executable`` here is a fact about the submitting box, not the target.
ARMS = {
    "gpu": {
        "why": "the CUDA-gated surface; nothing else exercises it",
        "python": "/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python",
        "pbrun_flags": ["--gpu"],
        "strict_cuda": True,
    },
    "x86": {
        "why": "the torch-free population: what a box with no torch collects",
        "python": "/home/rob/venvs/pb-cpu/bin/python",
        "pbrun_flags": ["--tag", "x86"],
        "strict_cuda": False,
        "needs_shared_checkout": True,
    },
}


def _git(checkout: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(checkout), *args],
                         capture_output=True, text=True, timeout=30)
    return out.stdout.strip() if out.returncode == 0 else ""


def _population_of(checkout: Path) -> dict:
    """What tree this receipt is about, stated so it cannot be mistaken.

    A receipt from a feature branch is not a merge receipt.  Recording the head
    of ``master`` alongside the commit under test makes that difference
    readable rather than assumed.
    """

    head = _git(checkout, "rev-parse", "HEAD")
    # A clone made for a pool run has no local ``master`` -- only
    # ``origin/master`` -- and a bare ``rev-parse master`` there returns
    # nothing, which would silently make ``is_master_head`` false for the very
    # commit that IS master.  Say which ref answered, so a null comparison
    # reads as "not established" and never as "not master".
    master, master_ref = "", ""
    for ref in ("master", "origin/master", "refs/remotes/origin/master"):
        master = _git(checkout, "rev-parse", ref)
        if master:
            master_ref = ref
            break
    return {
        "checkout": str(checkout),
        "commit": head,
        "describe": _git(checkout, "log", "-1", "--pretty=%s"),
        "master_ref_used": master_ref or "none resolved",
        "master_head_at_submit": master or None,
        "is_master_head": (head == master) if (head and master) else None,
        "working_tree_dirty": bool(_git(checkout, "status", "--porcelain")),
    }


def _command(arm: dict, surface_json: Path, extra: list[str]) -> list[str]:
    command = [arm["python"], "-m", "pytest", "tests", "-q",
               "-p", "no:cacheprovider",
               "--surface-json", str(surface_json)]
    if arm["strict_cuda"]:
        command.append("--strict-cuda")
    return command + extra


def _submit(name: str, arm: dict, args, receipt_dir: Path) -> dict:
    surface_json = receipt_dir / f"surface.{name}.json"
    command = _command(arm, surface_json, args.pytest_arg)
    invocation = [
        sys.executable, str(PBRUN),
        *arm["pbrun_flags"],
        "--cpus", str(args.cpus),
        "--demand", f"{'gpu=1,' if arm['pbrun_flags'][0] == '--gpu' else ''}"
                    f"mem_gb={args.mem_gb}",
        "--cwd", str(args.checkout),
        "--timeout-s", str(args.timeout_s),
        "--wait-s", str(args.wait_s),
        "--", *command,
    ]
    record = {
        "arm": name,
        "why": arm["why"],
        "python": arm["python"],
        "pbrun": " ".join(shlex.quote(part) for part in invocation),
    }
    if args.dry_run:
        record["status"] = "not submitted (--dry-run)"
        return record

    started = time.monotonic()
    proc = subprocess.run(invocation, capture_output=True, text=True)
    record["returncode"] = proc.returncode
    record["elapsed_s"] = round(time.monotonic() - started, 1)
    record["stderr_tail"] = proc.stderr.strip().splitlines()[-6:]
    record["stdout_tail"] = proc.stdout.strip().splitlines()[-6:]
    if surface_json.exists():
        record["surface"] = json.loads(surface_json.read_text())
    else:
        # The distinction that matters: a suite that ran and failed published a
        # surface; one that was never placed, or was refused before collection,
        # did not.  Saying which is the whole value of a receipt.
        record["surface"] = None
        record["no_surface_means"] = (
            "the suite published no population: it was refused, never placed, "
            "or died before the terminal summary. This is not a pass and not a "
            "fail; it is an absent measurement."
        )
    return record


def _verdict(arms: list[dict]) -> str:
    if any(record.get("status") for record in arms):
        return "not run"
    if any(record.get("surface") is None for record in arms):
        return "incomplete: an arm published no population"
    if any(record.get("returncode") != 0 for record in arms):
        return "red"
    return "green on both populations"


LEDGER_HEADER = """# Suite populations

One row per arm per `tools/merge_suite.py` run. The two arms of a run are
adjacent on purpose: a pass count means nothing without the device population
it was measured on, and this file exists so neither can be read without the
other (tessera#112).

`master head?` is whether the commit under test was master's tip at submit
time. `yes` is a merge receipt; `no` is a branch's own run; `unknown` means no
master ref resolved in that checkout and the question was not answered.

| when (UTC) | commit | master head? | arm | device | passed | failed | skipped | not collected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""


def _record_markdown(path: Path, receipt: dict) -> None:
    """Append this run's arms to a ledger a reader of the repo can check.

    A receipt on shared storage is not a scrollback, but it is also not
    somewhere anybody looks. This is.
    """

    population = receipt["population"]
    head = population["is_master_head"]
    head_text = "unknown" if head is None else ("yes" if head else "no")
    rows = []
    for record in receipt["arms"]:
        surface = record.get("surface") or {}
        counts = surface.get("counts") or {}
        cell = lambda key: str(counts.get(key, "--"))          # noqa: E731
        rows.append(
            f"| {receipt['generated_utc']} | `{population['commit'][:12]}` | "
            f"{head_text} | {record['arm']} | "
            f"{surface.get('device', 'no population published')} | "
            f"{cell('passed')} | {cell('failed')} | {cell('skipped')} | "
            f"{len(surface.get('not_collected', []))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(LEDGER_HEADER)
    with path.open("a") as handle:
        handle.write("\n".join(rows) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkout", default=str(Path(__file__).resolve().parents[1]),
                    help="tree to test; the x86 arm needs it under /mnt/shared")
    ap.add_argument("--arm", action="append", choices=sorted(ARMS),
                    default=[], help="repeatable; default is both")
    ap.add_argument("--cpus", type=int, default=8,
                    help="cores the run will actually use, declared to the pool")
    ap.add_argument("--mem-gb", type=int, default=16)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--wait-s", type=float, default=5400.0)
    ap.add_argument("--out", default="",
                    help="receipt path; default is a timestamped file under "
                         f"{DEFAULT_RECEIPT_ROOT}")
    ap.add_argument("--pytest-arg", action="append", default=[],
                    help="repeatable, passed through to pytest")
    ap.add_argument("--record", default="",
                    help="also append one row per arm to this markdown ledger; "
                         "docs/status/suite-populations.md is the one a reader "
                         "of the repo checks")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pbrun invocations and submit nothing")
    args = ap.parse_args()

    args.checkout = Path(args.checkout).resolve()
    wanted = args.arm or sorted(ARMS)

    shared = str(args.checkout).startswith(str(SHARED_ROOT))
    for name in wanted:
        if ARMS[name].get("needs_shared_checkout") and not shared:
            print(
                f"merge_suite: the {name} arm needs a checkout under "
                f"{SHARED_ROOT} -- pbrun pins an action to the submitting box "
                f"when its tree exists on one box only, and {args.checkout} "
                "does. Copy the tree there, or drop this arm and say in the "
                "receipt that only one population was covered.",
                file=sys.stderr)
            return 2

    stamp = time.strftime("%Y%m%dT%H%M%S")
    receipt_dir = DEFAULT_RECEIPT_ROOT / stamp
    # A dry run composes paths and creates nothing: littering shared storage
    # with an empty directory per invocation is how a receipt root stops being
    # readable.
    if not args.dry_run:
        receipt_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else receipt_dir / "receipt.json"

    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        futures = {name: pool.submit(_submit, name, ARMS[name], args, receipt_dir)
                   for name in wanted}
        arms = [futures[name].result() for name in wanted]

    receipt = {
        "schema": "tessera.merge_suite.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "submitted_from": os.uname().nodename,
        "population": _population_of(args.checkout),
        "verdict": _verdict(arms),
        "arms": arms,
        "reading_note": (
            "Each arm's counts belong to that arm's device population and to "
            "no other. A pass count quoted without the device beside it is the "
            "misreading tessera#112 is about."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.record:
        _record_markdown(Path(args.record), receipt)

    print(f"merge_suite: {receipt['verdict']}")
    for record in arms:
        surface = record.get("surface") or {}
        counts = surface.get("counts") or {}
        print(f"  {record['arm']:4s} rc={record.get('returncode', '-')} "
              f"cuda={surface.get('cuda', '?')} "
              f"passed={counts.get('passed', '?')} "
              f"failed={counts.get('failed', '?')} "
              f"skipped={counts.get('skipped', '?')} "
              f"not_collected={len(surface.get('not_collected', []))}")
    print(f"merge_suite: receipt {out}")
    return 0 if receipt["verdict"] == "green on both populations" else 1


if __name__ == "__main__":
    raise SystemExit(main())
