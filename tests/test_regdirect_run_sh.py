"""``experiments/regdirect_stage1/run.sh`` keeps its D30 host watchdog alive on a reused directory.

The watchdog exits when ``$OUT/.done`` exists, and a finished run leaves that marker.  Review
rev-1007-125515-85fd of PR 1029 (finding 3): a second run in the same directory must remove the
marker before the watchdog starts.  A fake ``docker`` records whether the marker still exists when
the container would start.  CPU only.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "regdirect_stage1" / "run.sh"


def test_a_stale_done_marker_is_removed_before_the_watchdog_starts(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / ".done").write_text("left by an earlier run\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text(f"""#!/usr/bin/env bash
if [[ "$1" == run ]]; then
  if [[ -e "{out}/.done" ]]; then echo stale > "{tmp_path}/seen"; else echo clear > "{tmp_path}/seen"; fi
fi
exit 0
""")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", ORACLE_IMAGE="fake/image")
    done = subprocess.run(["bash", str(RUN), str(ROOT), str(out), "--skip-check"], env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "seen").read_text().strip() == "clear"
