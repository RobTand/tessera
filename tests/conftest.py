"""Shared fixtures: small, fully-determined artifacts built from bytes."""

import hashlib
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
