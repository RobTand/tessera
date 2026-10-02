"""CPU ordering controls: durable published names must precede PB progress."""
import json
import stat
import sys

import pytest

from test_kda_probe_gate import probe


def observe_publication(probe, monkeypatch):
    events = []
    fsync, replace = probe.os.fsync, probe.os.replace

    def synced(fd):
        kind = "directory" if stat.S_ISDIR(probe.os.fstat(fd).st_mode) else "file"
        fsync(fd)
        events.append(kind)

    def renamed(source, target):
        replace(source, target)
        events.append("replace")

    monkeypatch.setattr(probe.os, "fsync", synced)
    monkeypatch.setattr(probe.os, "replace", renamed)
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_HELPER", "CPU-ordering-fixture")
    return events


def test_partial_progress_waits_for_durable_published_name(probe, monkeypatch, tmp_path):
    events = observe_publication(probe, monkeypatch)
    path = tmp_path / "kdafwd_partial.json"
    payload = {"cells": [{"completed_calls": 7}]}

    def commit(units, phase):
        assert units == 1 and phase == "measure"
        assert json.loads(path.read_text()) == payload
        assert events == ["file", "replace", "directory"]
        events.append("progress")

    monkeypatch.setattr(probe.runpy, "run_path", lambda _: {"commit": commit})
    probe.kda_commit(payload, path, 1)
    assert events[-1] == "progress"


def test_publish_progress_waits_for_complete_durable_result(probe, monkeypatch, tmp_path):
    events = observe_publication(probe, monkeypatch)
    path = tmp_path / "mhc_probe.json"
    monkeypatch.setattr(probe, "part_kdafwd", lambda *_: {"cells": [{"completed_calls": 7}]})
    monkeypatch.setattr(sys, "argv", ["mhc_probe.py", "--out", str(tmp_path), "--parts", "kdafwd"])

    def commit(units, phase):
        assert units == 1 and phase == "publish"
        saved = json.loads(path.read_text())
        assert saved["meta"]["utc_end"] >= saved["meta"]["utc_start"]
        assert saved["meta"]["power_sampler"] == "test"
        assert saved["kdafwd"]["cells"] == [{"completed_calls": 7}]
        assert "netdata" in saved
        assert events == ["file", "replace", "directory"]
        events.append("progress")

    monkeypatch.setattr(probe.runpy, "run_path", lambda _: {"commit": commit})
    assert probe.main() == 0
    assert events[-1] == "progress"


def test_directory_sync_failure_refuses_progress(probe, monkeypatch, tmp_path):
    fsync = probe.os.fsync
    published = []

    def sync_or_fail(fd):
        if stat.S_ISDIR(probe.os.fstat(fd).st_mode):
            raise OSError("directory persistence failed")
        return fsync(fd)

    monkeypatch.setattr(probe.os, "fsync", sync_or_fail)
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_HELPER", "CPU-ordering-fixture")
    monkeypatch.setattr(probe.runpy, "run_path", lambda _: {"commit": lambda *a: published.append(a)})
    with pytest.raises(OSError, match="directory persistence failed"):
        probe.kda_commit({"cells": [{}]}, tmp_path / "kdafwd_partial.json", 1)
    assert not published
