"""Paired CPU calculator profiles at a GLM expert shape; no encoding or serving."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import socket
import subprocess
import sys
import tarfile
import tempfile
import time


def arm(source: Path, out: Path):
    sys.path.insert(0, str(source / "src"))
    from tessera.calculator import terminal_rate

    out.mkdir()
    profile = cProfile.Profile()
    start_unix, start = time.time(), time.perf_counter()
    with profile:
        values = [terminal_rate(
            q256, 2048, 4096, cap=7, window_bits=12,
            with_scale_base=False, with_row_scale=True,
        ) for q256 in range(256, 7 * 256 + 1)]
    seconds, end_unix = time.perf_counter() - start, time.time()
    profile.dump_stats(out / "calculator.pstats")
    with (out / "profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(50)
    raw = json.dumps([[v.numerator, v.denominator] for v in values], separators=(",", ":")).encode()
    (out / "rates.json").write_bytes(raw)
    result = dict(start_unix=start_unix, end_unix=end_unix, seconds=seconds,
                  count=len(values), rates_sha256=hashlib.sha256(raw).hexdigest(),
                  host=socket.gethostname(), python=sys.version,
                  affinity=sorted(os.sched_getaffinity(0)),
                  sources={name: hashlib.sha256((source / "src/tessera" / name).read_bytes()).hexdigest()
                           for name in ("calculator.py", "layout.py", "planes.py", "manifest.py")})
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base")
    parser.add_argument("--arm-source", type=Path)
    args = parser.parse_args()
    if args.arm_source:
        arm(args.arm_source.resolve(), args.out)
        return
    if not args.base:
        parser.error("--base is required for a paired run")
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    baseline = subprocess.check_output(["git", "rev-parse", args.base], cwd=source, text=True).strip()
    results = []
    with tempfile.TemporaryDirectory(prefix="tessera-extent-baseline-") as temp:
        base = Path(temp)
        archive = base / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), baseline, "src"], cwd=source, check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(base, filter="data")
        for index, label in enumerate(("before", "after", "after", "before")):
            output = args.out / f"{index}-{label}"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out", str(output),
                            "--arm-source", str(base if label == "before" else source)], check=True)
            results.append(dict(label=label, **json.loads((output / "result.json").read_text())))
    from box_power_window import SERIES, _fetch

    after = int(min(row["start_unix"] for row in results))
    before = int(max(row["end_unix"] for row in results)) + 1
    queries = [(host, context, dims) for host in ("sparky", "sparklina") for context, dims in SERIES]
    def fetch(query):
        host, context, dims = query
        return host, context, _fetch(host, context, dims, after, before, before - after)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, queries))
    telemetry = {host: {context: data for h, context, data in records if h == host}
                 for host in ("sparky", "sparklina")}
    (args.out / "netdata.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    assert all(data["doc"]["result"]["data"] for _, _, data in records), "missing Netdata series"
    assert len({row["rates_sha256"] for row in results}) == 1, "calculator rates changed"
    result = dict(scope="CPU calculator only; 1537 window-body rates at 2048x4096, cap7, window12, row scales",
                  baseline_commit=baseline, arms=results, rates_identical=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
