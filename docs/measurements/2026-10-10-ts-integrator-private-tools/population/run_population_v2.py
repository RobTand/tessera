"""Run a 12-shard population and judge it from the pool's own records. Fails closed. Fixes tessera#1069.

Usage: python3 -I run_population_v2.py WORKTREE OUTDIR

What changed from run_population.py:
  * stdout and stderr are both kept, so the pool's `pbrun: queued <action key>` line reaches b-<i>.out;
  * each shard's pbrun return code is stored in rc-<i>.txt, and a non-zero one is a problem;
  * a receipt line that names a payload is followed, and the payload is stored with it;
  * the verdict is population_audit.py's: each shard needs an action key, a queue ending that says executed
    with returncode 0, exactly one distinct pytest summary line, no failed or error count, and no uncollected
    module. A missing summary, receipt or terminal status is INCOMPLETE and never counts as zero;
  * the exit status is 0 only for a GREEN verdict.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

W, D, N = sys.argv[1], sys.argv[2], 12
V = "/home/rob/venvs/tessera-train-8bff20d0/bin/python"
P = "/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py"
AUDIT = str(Path(__file__).with_name("population_audit.py"))
Path(D).mkdir(parents=True, exist_ok=True)
sel = json.loads(subprocess.check_output([sys.executable, "tools/impacted_tests.py", "--ref", "origin/master...HEAD", "--json"],
                                         cwd=W, text=True))
tests = sorted(sel["tests"])
files = [t for t in tests if os.path.isfile(os.path.join(W, t))]
missing = sorted(set(tests) - set(files))
Path(f"{D}/selected.txt").write_text("\n".join(tests) + "\n")
print("verdict", sel["verdict"], "selected", len(tests), "existing", len(files), "missing", missing,
      "forces_full", sel.get("forces_full"), flush=True)
if missing:
    print("NOT GREEN: selected tests missing from the tree:", missing)
    sys.exit(2)
bins = [[0, []] for _ in range(N)]
for f in sorted(files, key=lambda f: -os.path.getsize(os.path.join(W, f))):
    b = min(bins, key=lambda b: b[0])
    b[0] += os.path.getsize(os.path.join(W, f))
    b[1].append(f)


def run(i):
    cmd = ["python3", P, "--tag", "x86", "--demand", "mem_gb=4", "--cpus", "2", "--env", "TMPDIR=/tmp",
           # One native thread per pytest worker. Populations run before this line declared none, and the pool sealed 2.
           "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1",
           "--timeout-s", "2400", "--wait-s", "3000", "--", V, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "-n", "2", "--dist", "worksteal", "--durations=5", *bins[i][1]]
    done = subprocess.run(cmd, cwd=W, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    Path(f"{D}/b-{i}.out").write_text(done.stdout)
    Path(f"{D}/rc-{i}.txt").write_text(f"{done.returncode}\n")
    return i, done.returncode


# D21: at most eight PrismaBuild clients at once; the other shards queue.
with cf.ThreadPoolExecutor(min(N, 8)) as pool:
    codes = dict(pool.map(run, range(N)))
bad = {i: c for i, c in codes.items() if c != 0}
if bad:
    print("pbrun return codes that are not 0:", bad, flush=True)
audit = subprocess.run([sys.executable, "-I", AUDIT, D, "--json", f"{D}/audit.json", "--expect-shards", str(N)],
                       text=True, stdout=subprocess.PIPE)
print(audit.stdout, end="")
sys.exit(0 if audit.returncode == 0 and not bad else 1)
