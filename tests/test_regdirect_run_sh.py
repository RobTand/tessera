"""``experiments/regdirect_stage1/run.sh`` keeps its D30 host watchdog alive on a reused directory.

The watchdog exits when ``$OUT/.done`` exists, and a finished run leaves that marker.  Review
rev-1007-125515-85fd of PR 1029 (finding 3): a second run in the same directory must remove the
marker before the watchdog starts.  A fake ``podman`` records whether the marker still exists when
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
    fake = bin_dir / "podman"
    fake.write_text(f"""#!/usr/bin/env python3
import json, pathlib, sys
args=sys.argv[1:]
if args[:2]==['image','inspect']: print('sha256:'+'c'*64+'\\t'+json.dumps([args[2]]))
elif args[0]=='info': print('true')
elif args[0]=='create':
 pathlib.Path('{tmp_path}/seen').write_text('stale' if pathlib.Path('{out}/.done').exists() else 'clear')
 pathlib.Path(args[args.index('--cidfile')+1]).write_text('b'*64)
 scratch=args[args.index('--volume')+1].split(':')[0]
 pathlib.Path(scratch,'container-status').write_text(pathlib.Path('/proc/self/status').read_text())
 pathlib.Path('{tmp_path}/owned').touch()
elif args[0]=='rm': pathlib.Path('{tmp_path}/owned').unlink()
elif args[:2]==['container','exists']: sys.exit(int(not pathlib.Path('{tmp_path}/owned').exists()))
""")
    flux = bin_dir / 'flux'
    flux.write_text('#!/bin/sh\necho f123\n')
    flux.chmod(0o755)
    findmnt = bin_dir / 'findmnt'
    findmnt.write_text('#!/bin/sh\necho ext4\n')
    findmnt.chmod(0o755)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", ORACLE_IMAGE="example.invalid/cuda@sha256:" + "a" * 64,
               FLUX_JOB_ID="f123", CUDA_VISIBLE_DEVICES="0", FLUX_SCRATCH=str(tmp_path / "scratch"))
    done = subprocess.run(["bash", str(RUN), str(ROOT), str(out), "--skip-check"], env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "seen").read_text().strip() == "clear"
