"""Shared fixtures: small, fully-determined artifacts built from bytes."""

import hashlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.container import serialize  # noqa: E402
from tessera.grammar import bresenham_rate_schedule, root_from_q256  # noqa: E402
from tessera.layout import (  # noqa: E402
    TerminalSpec,
    build_plane_region,
    build_planes,
    build_terminal,
)
from tessera.planes import PlaneKind  # noqa: E402
from tessera.manifest import (  # noqa: E402
    ArrangementMode,
    BranchIdentity,
    ContainerClass,
    Geometry,
    Manifest,
    RotationState,
)

ALPHABET_BLOB = bytes(range(16))
DESCENDANT_BLOB = bytes(range(32))


def make_geometry(rows=8, columns=32, superblock_columns=8):
    return Geometry(
        rows=rows,
        columns=columns,
        superblock_columns=superblock_columns,
        group_weights=32,
        half_weights=16,
        quantizable_params=rows * columns,
    )


def make_artifact(q256=512, rows=8, columns=32, superblock_columns=8):
    """Build a complete, self-consistent artifact plus its terminal ladder."""
    geometry = make_geometry(rows, columns, superblock_columns)
    root = root_from_q256(q256)
    rates = bresenham_rate_schedule(root, columns)
    payloads = {
        PlaneKind.ALPHABET: ALPHABET_BLOB,
        PlaneKind.DESCENDANT: DESCENDANT_BLOB,
    }
    planes = build_planes(
        geometry,
        rates,
        ALPHABET_BLOB,
        DESCENDANT_BLOB,
        max_released=4,
        payloads=payloads,
    )
    plane_region = build_plane_region(planes, payloads)

    specs = [
        TerminalSpec("t-po2", (0,) * columns, with_scale_base=True),
        TerminalSpec(
            "t-c3",
            tuple(3 - rate for rate in rates),
            with_scale_base=True,
        ),
        TerminalSpec(
            "t-nvfp4",
            tuple(3 - rate for rate in rates),
            released_positions=4,
            with_scale_base=True,
            with_scale_refine=True,
            with_diagonals=True,
        ),
    ]
    terminals = tuple(
        build_terminal(
            geometry,
            rates,
            spec,
            planes,
            len(ALPHABET_BLOB),
            len(DESCENDANT_BLOB),
            plane_region=plane_region,
        )
        for spec in specs
    )

    manifest = Manifest(
        encoder_profile_id=hashlib.sha256(b"profile").digest(),
        branch=BranchIdentity(
            unit_id="layers.0.mlp.down_proj",
            root_q256=q256,
            rotation=RotationState.NONE,
            container=ContainerClass.GRIDBOOK,
        ),
        geometry=geometry,
        arrangement=ArrangementMode.BRESENHAM,
        rates=rates,
        planes=planes,
        terminals=terminals,
        payload_digest=hashlib.sha256(plane_region).digest(),
    )
    return manifest, plane_region, serialize(manifest, plane_region)


@pytest.fixture
def artifact():
    return make_artifact()


# --- what a torch-free interpreter can collect -----------------------------
#
# The `pure` CI job installs pytest and nothing else, to prove structurally
# that the byte layer needs no torch.  Most test modules DO need torch, and
# they cannot be collected there.
#
# Naming them is the obvious move and the wrong one: a roster of forty-odd
# filenames is a second place to remember, and the failure mode is silent --
# a new torch-needing test file turns the job red, someone appends it, and
# nothing checks that the entries still describe reality.  The roster is not
# the decision here; "this module cannot import without its dependency" is,
# and that is a question the module answers itself.
#
# So ask it.  In an interpreter that HAS the dependency this costs nothing:
# the loop does not run at all.  In one that does not, each failing import
# fails on its own import line, before any work.
def _third_party_import_failure(exc: BaseException) -> bool:
    """Did this fail for a missing dependency, or for a real bug?

    Only the former is a reason to skip a file.  A module that raises
    ``ImportError`` because one of ITS OWN names moved must still fail the
    run -- that is the breakage this whole job exists to catch.
    """
    missing = getattr(exc, "name", None) or ""
    root = missing.split(".", 1)[0]
    return bool(root) and root != "tessera" and root not in {"", "tests"}


def _uncollectable() -> list:
    import importlib
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        return []
    here = Path(__file__).resolve().parent
    skip = []
    for path in sorted(here.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(
            f"_probe_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except (ImportError, ModuleNotFoundError) as exc:
            if _third_party_import_failure(exc):
                skip.append(path.name)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            # Not ``Exception``: a module that already says
            # ``pytest.importorskip("torch")`` raises pytest's own ``Skipped``,
            # which derives from ``BaseException``.  That file is handling
            # itself correctly and must NOT be ignored -- pytest will report
            # it as a skip, which is the truth.  Anything else here is the
            # file's own problem and belongs in the run as the failure it is.
            pass
    return skip


collect_ignore = _uncollectable()


# --- what this run did NOT exercise ---------------------------------------
#
# tessera#112: master was red on three CUDA-gated tests while GitHub Actions,
# the x86 pool suite and a local CPU run all read green.  None of the three
# could have seen it.  The failure was not any one signal; it was that a run
# does not say which population it covered, so "41 passed, 4 skipped" reads as
# coverage when three of those four skips are the tests being asked about.
#
# Two things fix that, and they are different in kind.
#
# The *diagnostic* is unconditional: every run prints the device it had, how
# many tests it skipped, how many modules it never collected, and -- verbatim,
# never pattern-matched -- what the skip reasons said.  A reason histogram is
# the exact answer where a regex over "cuda|gpu|triton" would be a guess, and
# a guess that undercounts is the same blindness one level down.  There is no
# threshold: "more than N" would be a constant nobody derived, and the honest
# statement is the count itself (principle 2).
#
# The *gate* is ``--strict-cuda`` (or ``TESSERA_STRICT_CUDA=1``, for a wrapper
# that seals a command line).  It refuses the session outright when this
# interpreter has no CUDA device, so a merge check cannot read green by
# skipping the surface it was submitted to cover.  It asserts device presence,
# which is exact, rather than classifying individual skips, which is not.
#
# Nothing here imports torch at module scope: the ``pure`` CI job imports this
# file and asserts torch stayed out of ``sys.modules``.

_STRICT_CUDA_HELP = (
    "refuse the session unless a CUDA device is present, so a run submitted to "
    "cover the CUDA-gated surface cannot pass by skipping it"
)


def pytest_addoption(parser):
    parser.addoption(
        "--strict-cuda",
        action="store_true",
        default=False,
        help=_STRICT_CUDA_HELP,
    )
    parser.addoption(
        "--surface-json",
        action="store",
        default=os.environ.get("TESSERA_SURFACE_JSON", ""),
        metavar="PATH",
        help=(
            "write this run's population -- device, counts, verbatim skip "
            "reasons, uncollected modules -- as JSON, so a receipt is derived "
            "from a table this run publishes rather than scraped from its prose"
        ),
    )


def _strict_cuda(config) -> bool:
    if config.getoption("--strict-cuda"):
        return True
    return os.environ.get("TESSERA_STRICT_CUDA", "") not in ("", "0", "false")


def _cuda_device() -> tuple[bool, str]:
    """(has a device, one line naming it or naming its absence).

    Imported here rather than at module scope, and tolerant of every way this
    can go wrong: an interpreter with no torch, a torch built without CUDA, and
    a driver that raises on probe are three different absences and all three
    mean the same thing for coverage.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 -- any import failure is an absence
        return False, f"no torch in this interpreter ({type(exc).__name__})"
    try:
        if not torch.cuda.is_available():
            return False, f"torch {torch.__version__} reports no CUDA device"
        count = torch.cuda.device_count()
        return True, (
            f"torch {torch.__version__}, {count} CUDA device(s), "
            f"device 0 = {torch.cuda.get_device_name(0)}"
        )
    except Exception as exc:  # noqa: BLE001 -- a probe that raises is an absence
        return False, f"CUDA probe raised {type(exc).__name__}: {exc}"


def pytest_report_header(config):
    present, detail = _cuda_device()
    lines = [f"tessera surface: {'CUDA' if present else 'NO CUDA'} -- {detail}"]
    if collect_ignore:
        lines.append(
            f"tessera surface: {len(collect_ignore)} test module(s) not "
            "collected at all (their third-party imports are missing)"
        )
    return lines


def pytest_sessionstart(session):
    # xdist workers inherit the controller's verdict; one refusal, not one per
    # worker.
    if hasattr(session.config, "workerinput"):
        return
    if not _strict_cuda(session.config):
        return
    present, detail = _cuda_device()
    if present:
        return
    raise pytest.UsageError(
        "--strict-cuda: " + detail + ". This interpreter cannot exercise the "
        "CUDA-gated test surface, so a pass here would certify a population it "
        "never ran. Refusing rather than skipping (tessera#112)."
    )


def _skip_reason(report) -> str:
    """The reason text pytest recorded, verbatim."""

    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
    else:
        reason = str(longrepr)
    # pytest prefixes a bare ``Skipped: ``; the rest is the author's sentence.
    return reason[len("Skipped: "):] if reason.startswith("Skipped: ") else reason


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    from collections import Counter

    present, detail = _cuda_device()
    write = terminalreporter.write_line

    skipped = terminalreporter.stats.get("skipped", [])
    write("")
    write(f"tessera surface: {'CUDA' if present else 'NO CUDA'} -- {detail}")
    write(
        f"tessera surface: {len(skipped)} test(s) skipped, "
        f"{len(collect_ignore)} module(s) not collected"
    )
    if not present:
        write(
            "tessera surface: this run did not exercise the CUDA-gated "
            "surface. Its pass count is not coverage of it."
        )
    counts = Counter(_skip_reason(report) for report in skipped)
    if skipped:
        write("tessera surface: skip reasons, verbatim --")
        for reason, count in counts.most_common():
            write(f"    {count:5d}  {reason}")
    if collect_ignore:
        import textwrap

        write("tessera surface: modules not collected --")
        for line in textwrap.wrap(" ".join(collect_ignore), width=72):
            write(f"    {line}")

    destination = config.getoption("--surface-json")
    if destination:
        _write_surface_json(Path(destination), terminalreporter, present,
                            detail, counts, _strict_cuda(config))


def _write_surface_json(path, terminalreporter, present, detail, reasons, strict):
    """The same population, as a table rather than as prose.

    A receipt that scrapes "1404 passed / 487 skipped" out of a terminal is
    reading a rendering; this is the run stating its own population, which is
    what a downstream gate or a stored receipt should read.
    """

    import json

    stats = terminalreporter.stats
    payload = {
        "schema": "tessera.test_surface.v1",
        "cuda": present,
        "device": detail,
        "strict_cuda": strict,
        "counts": {
            outcome: len(stats.get(outcome, []))
            for outcome in ("passed", "failed", "error", "skipped",
                            "xfailed", "xpassed")
        },
        "skip_reasons": dict(sorted(reasons.items(),
                                    key=lambda kv: (-kv[1], kv[0]))),
        "not_collected": list(collect_ignore),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    terminalreporter.write_line(f"tessera surface: population written to {path}")
