"""The MoE campaign must propagate every required serve's result."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_failed_route_census_cannot_report_successful_campaign(tmp_path):
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    driver = experiments / "ts5_moe_served.sh"
    shutil.copyfile(ROOT / "experiments" / driver.name, driver)
    for name, status in (("serve_and_dump_kl.sh", 0),
                         ("tessera_plugin_served.sh", 0),
                         ("tessera_plugin_run.sh", 23)):
        stub = experiments / name
        stub.write_text(f"#!/bin/sh\nexit {status}\n")
        stub.chmod(0o755)
    env = dict(os.environ, EXT=str(tmp_path / "ext"),
               TESSERA_CENSUS_LOCAL=str(tmp_path / "census"))
    result = subprocess.run(
        ["bash", str(driver), "bf16", "wire", str(tmp_path / "output")],
        env=env, text=True, capture_output=True, timeout=15)
    assert "teacher=0 student=0 census=23" in result.stdout
    assert result.returncode != 0, result.stdout
