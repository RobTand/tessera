"""Transport-independent population records and Git histories for suite tests."""

import subprocess


def _master_history(checkout, master_ref):
    """Two real master tips, with measured HEAD fixed at the first one.

    May extend an existing fixture checkout. No remote, live pool, or runner's
    default git branch/configuration is needed.
    """

    checkout.mkdir(exist_ok=True)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(checkout), "-c", "user.name=Test",
             "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
             *args], text=True).strip()

    if not (checkout / ".git").exists():
        git("init", "-q", "--initial-branch=measured")
        git("commit", "-q", "--allow-empty", "-m", "measured source")
    else:
        git("checkout", "-q", "-B", "measured")
    git("update-ref", "-d", "refs/heads/master")
    git("update-ref", "-d", "refs/remotes/origin/master")
    submitted = git("rev-parse", "HEAD")
    advanced = git("commit-tree", "HEAD^{tree}", "-p", submitted, "-m", "master advances")
    ref = "refs/heads/master" if master_ref == "master" else "refs/remotes/origin/master"
    git("update-ref", ref, submitted)
    return git, ref, submitted, advanced


def _population(arm="gpu", *, source="c" * 64, commit="a" * 40, **overrides):
    """An arm record holding a population with every field green requires.

    A hand-made surface with three fields in it is not a smaller version of a
    published population; it is a different object, and a verdict tested only
    against it is a verdict tested against nothing the suite writes.  This is
    what ``tests/conftest.py`` actually publishes, so a test that wants a gap
    states the gap as an override.
    """

    cuda = arm == "gpu"
    surface = {
        "schema": "tessera.test_surface.v3",
        "role": "population",
        "worker_id": None,
        "commit": commit,
        "source_identity": {"schema": "tessera.suite_source.v1",
                            "verification": "verified",
                            "snapshot_commit": commit, "sha256": source},
        "cuda": cuda,
        "strict_cuda": cuda,
        "device": ("torch 2.11.0+cu130, 1 CUDA device(s), device 0 = NVIDIA GB10"
                   if cuda else "torch 2.11.0+cpu reports no CUDA device"),
        "counts": {"passed": 1827, "failed": 0, "error": 0, "skipped": 10,
                   "xfailed": 0, "xpassed": 0},
        "skip_reasons": {},
        "cuda_surface": {"executed": 311 if cuda else 0,
                         "is_a_floor": True,
                         "box_artifact_skips": {}},
        "not_collected": [],
    }
    surface.update(overrides)
    return {"arm": arm, "requires_cuda": cuda, "returncode": 0,
            "surface": surface}
