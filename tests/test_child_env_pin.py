"""The child-environment helper pins this checkout's ``src``, not the wheel."""
from __future__ import annotations

import os
import subprocess
import sys

import child_env


def test_the_checkout_src_is_first_and_the_ambient_value_is_kept(monkeypatch):
    ambient = os.pathsep.join(["/ambient/kept-first", "/ambient/kept-second"])
    monkeypatch.setenv("PYTHONPATH", ambient)
    parts = child_env.child_env()["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(child_env.SRC), parts
    assert parts[0].endswith("src"), parts
    assert parts[1:] == ["/ambient/kept-first", "/ambient/kept-second"], parts


def test_without_ambient_pythonpath_the_child_gets_the_checkout_alone(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert child_env.child_env()["PYTHONPATH"] == str(child_env.SRC)


def test_a_child_resolves_tessera_from_the_checkout_the_helper_names():
    """End to end: a fresh interpreter given the helper's env imports the
    checkout, not whichever wheel its site-packages carries."""
    proc = subprocess.run(
        [sys.executable, "-c", "import tessera; print(tessera.__file__)"],
        env=child_env.child_env(), capture_output=True, text=True, check=True)
    assert proc.stdout.strip().startswith(str(child_env.SRC)), proc.stdout
