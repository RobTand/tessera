"""The export refuses an unstampable commit before writing any byte (#714).

R3 attempt 3 streamed 120 shards and only then refused: the refusal is
correct, its timing is the defect. These tests drive ``main()`` on a tiny
CPU fixture with every commit source blinded (no git, no TESSERA_GIT, no
install commit -- the linked-worktree case), counting shard writes through
a fake writer. The driver is mutated (argv, env, monkeypatched writer);
the fixture is an ordinary tiny checkpoint.
"""
import json
import subprocess

import pytest
import torch
from safetensors.torch import save_file

from tessera import export_serving

FAKE_COMMIT = "failfast-fake-commit"


def _tiny_source(path):
    path.mkdir()
    (path / "config.json").write_text("{}")
    save_file({"model.layers.0.mlp.down_proj.weight": torch.zeros(64, 64)},
              str(path / "model.safetensors"))


def _blind_commit(monkeypatch):
    monkeypatch.delenv("TESSERA_GIT", raising=False)
    monkeypatch.setattr(subprocess, "check_output", _no_git)
    monkeypatch.setattr(export_serving, "_installed_commit_id", lambda: None)


def _no_git(*args, **kwargs):
    raise FileNotFoundError("no git here")


def _counting_writer(monkeypatch):
    writes = []
    real = export_serving.save_serving_shard

    def _count(payload, destination):
        writes.append(destination.name)
        return real(payload, destination)

    monkeypatch.setattr(export_serving, "save_serving_shard", _count)
    return writes


def _run_main(monkeypatch, src, out):
    # --allow-unrouted: the synthetic module is uncensused, so the pinned
    # runtime routes nothing through the plugin; the research escape encodes
    # it as dead wire (stamping the refusal into serving_gate) instead of
    # refusing or passing everything through. That keeps the driver on the
    # normal encode path to the manifest stamp. Driver mutation, not fixture.
    # --grid BF16: the default NVFP4 route needs static input scales; BF16
    # needs none, and the encode of one tiny weight on CPU is seconds.
    monkeypatch.setattr("sys.argv",
                        ["export_serving", str(src), str(out), "--device", "cpu",
                         "--allow-unrouted", "--grid", "BF16"])
    return export_serving.main()


def test_unstampable_commit_refuses_before_any_write(tmp_path, monkeypatch):
    src = tmp_path / "src"
    _tiny_source(src)
    out = tmp_path / "out"
    _blind_commit(monkeypatch)
    writes = _counting_writer(monkeypatch)
    with pytest.raises(SystemExit, match="cannot stamp the Tessera commit"):
        _run_main(monkeypatch, src, out)
    assert writes == [], f"the write phase ran before the refusal: {writes}"


def test_tessera_git_export_stamps_the_same_value(tmp_path, monkeypatch):
    """Value continuity: a TESSERA_GIT export stamps the env value, before
    and after the hoist."""
    src = tmp_path / "src"
    _tiny_source(src)
    out = tmp_path / "out"
    # Blind git as well: where git resolves, the git path shadows the env
    # path by design, so the env path is only deterministic with git blind.
    _blind_commit(monkeypatch)
    monkeypatch.setenv("TESSERA_GIT", FAKE_COMMIT)
    _run_main(monkeypatch, src, out)
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["git"] == FAKE_COMMIT
