#!/usr/bin/env python3
"""Run the tests a change selects as shards, and judge them from merge_suite's receipt.

``tools/impacted_tests.py`` says which test files a change reaches.  This tool
splits that set into shards, submits each shard through
``tools/merge_suite.py`` as one arm, and writes one receipt for the whole
population.  It owns no verdict of its own: every shard is read the way a
merge_suite arm is read (its published population, its source identity, and the
exit status PrismaBuild's worker recorded), and the receipt comes from
``merge_suite._assemble_receipt``.  What this tool adds is what a set of shards
needs and a single arm does not:

* the sealed commands of the shards must run the selected files exactly, each
  file once;
* in certified mode the shards' effective source must be the source of the
  checkout that chose the files, not only the same source as each other; in
  dev mode (D32, the default) that comparison is a seal, so it is stamped and
  not computed;
* no pbrun return code may be non-zero, and no shard may leave a module
  uncollected.

Without those three, a population could be green over a file nobody ran, over
a tree the selector never read, or after a shard the pool had already failed
(tessera#1069).

Usage::

    python3 tools/selected_population.py --base origin/master --shards 12
    python3 tools/selected_population.py --resume <receipt dir>

``--resume`` submits nothing.  It rebuilds the receipt from the selection and
the populations the pool published, for a run whose submitting process died.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_suite  # noqa: E402
from tessera._dev.suite_source import measured_source  # noqa: E402
from tessera.dev_mode import NOT_COMPUTED, dev_mode_enabled, seal_check  # noqa: E402

#: The selection, saved beside the shard populations so ``--resume`` knows
#: which files each shard was given.  It is data this tool wrote, and every
#: claim it supports is re-checked against the pool's sealed commands.
SELECTION = "selection.json"
RECEIPT = "receipt.json"
#: What each pbrun client reported when it exited, written then, so a resume
#: after the submitter dies reads the client's own result and not only the
#: pool's.  The pool's record of an action can be clean while pbrun returned 1.
CLIENT = "client.{name}.json"
_CLIENT_FIELDS = ("returncode", "elapsed_s", "stderr_tail", "pbrun")
SELECTOR = Path(__file__).resolve().parent / "impacted_tests.py"
DEFAULT_RECEIPT_ROOT = merge_suite.SHARED_ROOT / "tessera-selected-populations"
#: D21: one agent keeps at most eight PrismaBuild clients at once, because each
#: client snapshots and hashes its checkout on the control seat.  The shard
#: count is a separate number: shards past this many queue behind the clients.
MAX_CLIENTS = 8
#: pytest's slowest-test report, so a slow shard is visible in its own stdout.
PYTEST_ARGS = ["--durations=5"]


def shard_name(index: int) -> str:
    return f"shard-{index:02d}"


def balanced_shards(sizes: dict[str, int], count: int) -> list[list[str]]:
    """Split the files into at most ``count`` shards of near-equal total size.

    Largest file first, each into the lightest shard.  Ties go to the lower
    shard and equal sizes keep path order, so one selection always splits the
    same way.  A shard is never empty: a selection smaller than ``count`` gets
    fewer shards.
    """

    if count < 1:
        raise ValueError("a population needs at least one shard")
    bins: list[tuple[int, list[str]]] = [(0, []) for _ in range(min(count, len(sizes)))]
    for path in sorted(sizes, key=lambda name: (-sizes[name], name)):
        index = min(range(len(bins)), key=lambda i: (bins[i][0], i))
        total, files = bins[index]
        bins[index] = (total + sizes[path], [*files, path])
    return [sorted(files) for _, files in bins]


def choose(checkout: Path, base: str, count: int) -> dict:
    """The selector's answer for ``base...HEAD``, split into shards.

    Refuses, by name, an answer it cannot run: no JSON (the selector prints
    nothing for an empty change), or a selected file the tree does not hold.
    """

    done = subprocess.run(
        [sys.executable, str(SELECTOR), "--ref", f"{base}...HEAD", "--json", "--root", "."],
        cwd=checkout, capture_output=True, text=True)
    if done.returncode != 0 or not done.stdout.strip():
        raise ValueError(
            f"the selector gave no selection for {base}...HEAD (exit {done.returncode}): "
            + (done.stderr.strip()[-300:] or "no output; the change may be empty"))
    result = json.loads(done.stdout)
    tests = sorted(result["tests"])
    missing = [path for path in tests if not (checkout / path).is_file()]
    if missing:
        raise ValueError(f"the selector named test files the tree does not hold: {missing}")
    shards = balanced_shards({path: (checkout / path).stat().st_size for path in tests}, count)
    return {
        "base": base,
        "verdict": result["verdict"],
        "forces_full": result.get("forces_full"),
        "excluded_tests": result.get("excluded_tests"),
        "tests": tests,
        "shards": {shard_name(i): files for i, files in enumerate(shards)},
    }


def shard_arm(name: str, files: list[str], python: str | None = None) -> dict:
    """merge_suite's x86 arm, running these files under the declared ``-n`` mode.

    The arm's own interpreter is used unless the run names one: the arm's
    default has no ``jsonschema``, so a test that imports it fails there for
    want of the package.
    """

    arm = {**merge_suite.ARMS["x86"], "targets": list(files), "dist": "worksteal",
           "why": f"{name}: {len(files)} selected test file(s) on the device-less population"}
    if python:
        arm["python"] = python
    return arm


def write_client_result(receipt_dir: Path, name: str, sent: dict) -> None:
    """Record one client's result atomically, the moment it is known."""

    path = receipt_dir / CLIENT.format(name=name)
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps({key: sent.get(key) for key in _CLIENT_FIELDS}))
    scratch.replace(path)


def read_client_result(receipt_dir: Path, name: str) -> dict | None:
    """The client's recorded result, or ``None`` where none was recorded.

    ``None`` is an unknown result, stated as such on the record.  It is not a
    zero exit.
    """

    try:
        recorded = json.loads((receipt_dir / CLIENT.format(name=name)).read_text())
    except (OSError, ValueError):
        return None
    return recorded if isinstance(recorded, dict) else None


def submit_all(selection: dict, args, receipt_dir: Path) -> dict[str, dict]:
    """Submit every shard, at most ``MAX_CLIENTS`` at once; each answer is merge_suite's record."""

    submission = SimpleNamespace(
        cpus=args.cpus, mem_gb=args.mem_gb, pytest_arg=[*PYTEST_ARGS, *args.pytest_arg],
        timeout_s=args.timeout_s, wait_s=args.wait_s, checkout=args.checkout,
        dry_run=args.dry_run, artifact_root=args.artifact_root)
    shards = selection["shards"]

    def client(name: str, files: list[str]) -> dict:
        sent = merge_suite._submit(name, shard_arm(name, files, selection.get("python")),
                                   submission, receipt_dir)
        if not args.dry_run:
            write_client_result(receipt_dir, name, sent)
        return sent

    with ThreadPoolExecutor(max_workers=min(MAX_CLIENTS, len(shards))) as pool:
        futures = {name: pool.submit(client, name, files) for name, files in shards.items()}
        return {name: future.result() for name, future in futures.items()}


class SourceMismatch(RuntimeError):
    """The shards' effective source is not the checkout's (certified mode only)."""


def checkout_source(records: list[dict], checkout: Path) -> tuple[dict, list[str]]:
    """The tie between the shards' source and the checkout's, as a seal (D32).

    It compares a stored run identity with the live checkout, so it goes
    through ``seal_check``.  Dev mode prints one ``[DEV-MODE]`` line and
    continues, and it computes no digest of the checkout for this alone.
    Certified mode (``PRISMAQUANT_DEV_MODE=0``) measures the checkout and
    refuses a tree that is unverified or different.
    """

    seen = sorted({record["surface"]["source_identity"].get("sha256") for record in records
                   if (record.get("surface") or {}).get("source_identity", {}).get("verification")
                   == "verified"} - {None})
    if dev_mode_enabled():
        seal_check("effective source of the shards", seen, NOT_COMPUTED, where=__name__)
        return {"verification": "not computed",
                "reason": "dev mode (D32): the checkout's digest is not computed for this comparison"}, []
    head = measured_source(checkout, verifier=None)
    if head.get("verification") != "verified":
        return head, ["the checkout's own source identity is not verified "
                      f"({head.get('reason')}), so the shards cannot be tied to it"]
    try:
        seal_check("effective source of the shards", [head["sha256"]], seen, where=__name__,
                   refusal=SourceMismatch)
    except SourceMismatch:
        return head, ["the shards' effective source is not the checkout's: "
                      f"checkout {head['sha256'][:12]}, shards "
                      f"{[digest[:12] for digest in seen] or 'none verified'}"]
    return head, []


def population_problems(selection: dict, records: list[dict]) -> list[str]:
    """What the shards together fail to establish, beyond what each one did.

    merge_suite judges each shard.  Four things belong to the set: the files
    the sealed commands ran, the exit codes pbrun gave, the modules a shard did
    not collect (a pass count never shows them), and, as a seal,
    the tree the selector read (``checkout_source``).  Every problem is named;
    none is a warning.
    """

    problems: list[str] = []
    expected = selection["shards"]
    ran: dict[str, int] = {}
    for record in records:
        name = record["arm"]
        action = record.get("pool_action")
        if not action:
            problems.append(f"{name}: no sealed command is bound to this shard, so what it ran is unknown")
            continue
        argv, why = merge_suite._pytest_argv(action.get("command") or [])
        if argv is None:
            problems.append(f"{name}: the sealed command is not read as pytest ({why})")
            continue
        targets = merge_suite._targets_of(argv)
        if sorted(targets) != sorted(expected.get(name, [])):
            problems.append(f"{name}: the sealed command ran {len(targets)} file(s), "
                            f"not the {len(expected.get(name, []))} it was given")
        for path in targets:
            ran[path] = ran.get(path, 0) + 1
        uncollected = (record.get("surface") or {}).get("not_collected")
        if uncollected:
            problems.append(f"{name}: {len(uncollected)} module(s) were not collected: "
                            f"{', '.join(map(str, uncollected[:3]))}")
        code = record.get("submit_returncode")
        if code not in (0, None):
            problems.append(f"{name}: pbrun returned {code}")
    selected = set(selection["tests"])
    if set(expected) != {record["arm"] for record in records}:
        problems.append("the shards read back are not the shards the selection names")
    never_ran = sorted(selected - set(ran))
    extra = sorted(set(ran) - selected)
    twice = sorted(path for path, times in ran.items() if times > 1)
    for label, paths in (("selected but never run", never_ran), ("run but not selected", extra),
                         ("run more than once", twice)):
        if paths:
            problems.append(f"{len(paths)} file(s) {label}: {', '.join(paths[:5])}"
                            + (" ..." if len(paths) > 5 else ""))
    return problems


def assemble(selection: dict, receipt_dir: Path, checkout: Path, *,
             assembled_by: str = "resume") -> dict:
    """The one receipt for this population.

    Each shard's record is merge_suite's resumed reading of its published
    population: the pool's own outcome record supplies the exit status, bound
    to the population by its producer stamp.  pbrun's return code is not that
    record, so each client's own result is read back from the file it wrote
    when it exited.  A run this process watched and a resume read the same
    files; a shard with none says ``not recorded``.
    """

    records = []
    for name, files in selection["shards"].items():
        record = merge_suite._resume(
            name, shard_arm(name, files, selection.get("python")), receipt_dir)
        sent = read_client_result(receipt_dir, name)
        record["client_result"] = "recorded" if sent is not None else "not recorded"
        if sent is not None:
            record["submit_returncode"] = sent.get("returncode")
            record["submit_elapsed_s"] = sent.get("elapsed_s")
            record["submit_stderr_tail"] = sent.get("stderr_tail")
            record["pbrun"] = sent.get("pbrun")
        records.append(record)
    receipt = merge_suite._assemble_receipt(records, checkout, assembled_by=assembled_by)
    head_source, source_problems = checkout_source(records, checkout)
    problems = population_problems(selection, records) + source_problems
    receipt["schema"] = "tessera.selected_population.v1"
    receipt["selection"] = selection
    receipt["checkout_source"] = head_source
    receipt["population_problems"] = problems
    receipt["clients_not_recorded"] = [record["arm"] for record in records
                                       if record["client_result"] == "not recorded"]
    if problems and receipt["verdict"].startswith("green on"):
        receipt["verdict"] = "incomplete: " + problems[0]
    return receipt


def report_lines(receipt: dict) -> list[str]:
    """One block per shard: its counts, then each skip reason once, verbatim."""

    selection = receipt["selection"]
    lines = [f"selected_population: {receipt['verdict']}",
             f"  {len(selection['tests'])} test file(s) in {len(selection['shards'])} shard(s); "
             f"selector verdict {selection['verdict']} against {selection['base']}"]
    for problem in receipt["population_problems"]:
        lines.append(f"  PROBLEM: {problem}")
    if receipt["clients_not_recorded"]:
        lines.append("  client result not recorded for: "
                     + ", ".join(receipt["clients_not_recorded"])
                     + "; the pool's records decide those shards alone")
    for record in receipt["arms"]:
        surface = record.get("surface") or {}
        counts = surface.get("counts") or {}
        lines.append(
            f"  {record['arm']} rc={record.get('returncode')} "
            f"observed={'yes' if record.get('exit_status_observed') else 'no'} "
            f"passed={counts.get('passed', '?')} failed={counts.get('failed', '?')} "
            f"errors={counts.get('error', '?')} skipped={counts.get('skipped', '?')} "
            f"not_collected={len(surface.get('not_collected') or [])} "
            f"device={surface.get('device', '?')}")
        for reason, count in (surface.get("skip_reasons") or {}).items():
            lines.append(f"      {count:>5}  {reason}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkout", default=str(Path(__file__).resolve().parents[1]),
                    help="the tree to test; it must be clean and under /mnt/shared")
    ap.add_argument("--base", default="origin/master", help="the left endpoint of base...HEAD")
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--python", default="",
                    help="interpreter every shard runs under; the x86 arm's own when omitted")
    ap.add_argument("--cpus", type=int, default=2, help="cores per shard, also pytest's -n")
    ap.add_argument("--mem-gb", type=int, default=4)
    ap.add_argument("--timeout-s", type=float, default=2400.0)
    ap.add_argument("--wait-s", type=float, default=3000.0)
    ap.add_argument("--artifact-root", action="append", default=[], metavar="ENV=PATH")
    ap.add_argument("--pytest-arg", action="append", default=[])
    ap.add_argument("--out", default="", help="receipt directory; default is a timestamped one")
    ap.add_argument("--resume", default="", help="a receipt directory; submit nothing")
    ap.add_argument("--pool-root", default=str(merge_suite.POOL_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.shards < 1 or args.cpus < 1 or args.mem_gb < 1:
        ap.error("shards, cpus and memory must be positive")
    merge_suite.use_pool_root(args.pool_root)
    args.checkout = Path(args.checkout).resolve()

    if args.resume:
        receipt_dir = Path(args.resume).resolve()
        selection = json.loads((receipt_dir / SELECTION).read_text())
    else:
        if not str(args.checkout).startswith(str(merge_suite.SHARED_ROOT)) and not args.dry_run:
            print(f"selected_population: the checkout must be under {merge_suite.SHARED_ROOT}",
                  file=sys.stderr)
            return 2
        try:
            selection = choose(args.checkout, args.base, args.shards)
        except ValueError as error:
            print(f"selected_population: {error}", file=sys.stderr)
            return 2
        if args.python:
            selection["python"] = args.python
        if not selection["tests"]:
            print("selected_population: the selector selected no test files; nothing to run")
            return 3
        receipt_dir = (Path(args.out) if args.out
                       else DEFAULT_RECEIPT_ROOT / time.strftime("%Y%m%dT%H%M%S")).resolve()
        # A dry run composes paths and creates nothing, as merge_suite's does.
        if not args.dry_run:
            receipt_dir.mkdir(parents=True, exist_ok=True)
            (receipt_dir / SELECTION).write_text(json.dumps(selection, indent=1) + "\n")
        submitted = submit_all(selection, args, receipt_dir)
        if args.dry_run:
            for record in submitted.values():
                print(record["pbrun"])
            return 0

    receipt = assemble(selection, receipt_dir, args.checkout,
                       assembled_by="resume" if args.resume else "submit")
    (receipt_dir / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
    print("\n".join(report_lines(receipt)))
    print(f"selected_population: receipt {receipt_dir / RECEIPT}")
    return 0 if receipt["verdict"].startswith("green on") else 1


if __name__ == "__main__":
    raise SystemExit(main())
