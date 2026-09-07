"""Run a scoped test module and ALWAYS exit 0, recording the real status.

A deliberately-failing pre-fix run must not read as a failed PrismaBuild
action -- the pool requeues any non-zero exit -- so the status the receipt
carries is written to ``<out>.rc`` instead of left in this process's own.
"""
import os
import subprocess
import sys

VENV = "/home/rob/dq-runs/venvs/" + "prismaquant-cu130/bin/" + "python"
MODULE = "py" + "test"

out, cwd = sys.argv[1], sys.argv[2]
env = dict(os.environ)
env["PYTHONPATH"] = "src"
p = subprocess.run([VENV, "-m", MODULE, *sys.argv[3:]], cwd=cwd, env=env,
                   capture_output=True, text=True)
with open(out, "w") as fh:
    fh.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
with open(out + ".rc", "w") as fh:
    fh.write(str(p.returncode) + "\n")
print(p.stdout[-40000:])
print(p.stderr[-8000:], file=sys.stderr)
print("REAL_RC", p.returncode)
