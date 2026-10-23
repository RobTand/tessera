"""Prepare the source-owned scorer from its immutable Git bundle."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile


def prepare_scorer_source(bundle, expected_sha256):
    bundle = Path(bundle).resolve()
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("scorer bundle bytes differ from the declared input")
    listed = subprocess.run(["git", "bundle", "list-heads", str(bundle)],
                            check=True, text=True, capture_output=True).stdout.splitlines()
    if len(listed) != 1:
        raise ValueError("the scorer input must contain one immutable source head")
    commit, reference = listed[0].split()
    root = Path(tempfile.mkdtemp(prefix="pq2459-scorer-"))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "fetch", "--quiet", str(bundle), reference], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--quiet", "--detach", commit], check=True)
    return {"root": str(root), "commit": commit, "bundle": str(bundle), "bundle_sha256": digest}
