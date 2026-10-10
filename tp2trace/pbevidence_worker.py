#!/usr/bin/env python3
"""Print the live platform and accelerator facts of this box as one packet (#1598).

A class-scoped GPU measurement seals the platform and accelerator facts of its
class.  ``pbrun`` read them from the box that submits, so a box without an
accelerator could not submit one.  Run this tool as a normal PrismaBuild action
on a worker of the class.  It prints the packet that ``pbrun --target-evidence``
reads.

The packet is the evidence the worker preflight already collects, in the shape
the core already validates.  It names the host and the device UUIDs as
provenance.  The sealed identity does not contain them.  Each worker still
checks the declared facts against its own live facts before it runs, so a wrong
packet fails closed.

    pbevidence.py [--out PATH]

With ``--out`` the packet goes to that file in one atomic step.  Otherwise it
goes to stdout.  The tool exits 1 and writes nothing when this box cannot
attest an accelerator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve(strict=True).parent
sys.path.insert(0, str(HERE))
from runtime_paths import generation_root  # noqa: E402

RUNTIME_ROOT = generation_root(__file__)
sys.path.insert(0, str(RUNTIME_ROOT / "src"))
from prismabuild import core as pb, materialize  # noqa: E402
from prismabuild.movement_actions import SEALED_ARGV0  # noqa: E402


#: The packet also names the identity of the executable a class-scoped action
#: seals as argv[0], as the WORKER sees it.  A class spans boxes whose
#: ``/bin/bash`` differs (the sizes differ between an x86_64 submitter and an
#: aarch64 worker), and the worker refuses a declared size that is not its own,
#: so the submitter must not read this from its own file system (#1598).
ARGV0_KEY = "argv0"

#: A scratch recorder runs the worker's own interpreter as argv[0], and its
#: sealed toolchain names that interpreter's digest, size and version.  The
#: submitter cannot read those from its own file system for the same reason, so
#: a packet collected with ``--recorder-python`` carries them (#1598).
RECORDER_KEY = "recorder"


class PacketError(ValueError):
    """A packet that cannot seal the facts of an accelerator class."""


def vet(evidence: dict) -> dict:
    """The normalized packet, or a ``PacketError`` naming what is wrong.

    This is the class-independent half of the vetting.  ``pbrun`` adds the
    check that the packet agrees with the class it is used for.  The
    worker's argv[0] identity rides in the same file but is not evidence: the
    core normalizer refuses an extra field, so it is taken out here and read
    by :func:`argv0_contract`.
    """

    if isinstance(evidence, dict) and (ARGV0_KEY in evidence or RECORDER_KEY in evidence):
        evidence = {key: value for key, value in evidence.items()
                    if key not in (ARGV0_KEY, RECORDER_KEY)}
    # A SLURM packet is refused for what it is, whatever its job record holds.
    if isinstance(evidence, dict) and (
            evidence.get("source") == "slurm" or evidence.get("slurm") is not None):
        raise PacketError(
            "the packet must be local: a SLURM packet names a job, not a box")
    try:
        packet = pb._normalize_worker_evidence(evidence)
    except pb.ActionContractError as exc:
        raise PacketError(f"it is not a worker evidence packet: {exc}") from None
    if packet["source"] != "local" or packet["slurm"] is not None:
        raise PacketError(
            "the packet must be local: a SLURM packet names a job, not a box")
    accelerators = packet["accelerators"]
    assert isinstance(accelerators, list)
    if not accelerators:
        raise PacketError("the packet reports no accelerator")
    try:
        pb.accelerator_models_contract(packet)
    except pb.ActionContractError:
        raise PacketError(
            "the packet carries no device identity (name and uuid) for an "
            "accelerator") from None
    # Several devices of one model are one class.  Two models are not, even
    # when they share a compute capability and a driver: the sealed model hash
    # would cover a mixed set that no class has.
    if len({str(row["name"]) for row in accelerators}) != 1:
        raise PacketError(
            "the packet must report one accelerator model, one compute "
            "capability and one driver")
    toolchain = pb.live_platform_toolchain_contract(evidence=packet)
    if "cuda_compute_capability" not in toolchain:
        raise PacketError(
            "the packet must report one accelerator model, one compute "
            "capability and one driver")
    return packet


def argv0_contract(packet: dict) -> dict[str, str]:
    """The ``argv0.*`` toolchain fields a packet declares for the class.

    Refuses a packet with no identity, a path other than the sealed argv[0], a
    digest that is not 64 hex characters, or a size that is not a canonical
    positive integer.
    """

    value = packet.get(ARGV0_KEY)
    if not isinstance(value, dict):
        raise PacketError(
            "the packet carries no argv0 identity of the worker's "
            f"{SEALED_ARGV0}; collect it again with the current pbevidence.py")
    if value.get("path") != SEALED_ARGV0:
        raise PacketError(f"the packet's argv0 is not {SEALED_ARGV0}")
    digest, size = value.get("sha256"), value.get("bytes")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise PacketError("the packet's argv0 sha256 is not 64 hex characters")
    if (not isinstance(size, str) or not size.isascii() or not size.isdigit()
            or size != str(int(size)) or int(size) <= 0):
        raise PacketError("the packet's argv0 bytes is not a canonical positive integer")
    return {"argv0.sha256": digest, "argv0.bytes": size}


def recorder_contract(packet: dict) -> dict[str, str] | None:
    """The worker's recorder interpreter facts, or ``None`` when none is declared.

    The result is the exact toolchain fields a scratch recorder seals, plus the
    interpreter path they describe under ``path``.  A malformed declaration is
    refused, never ignored.
    """

    value = packet.get(RECORDER_KEY)
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes", "python"}:
        raise PacketError("the packet's recorder must name path, sha256, bytes and python")
    path, digest, size, version = (value[k] for k in ("path", "sha256", "bytes", "python"))
    if not isinstance(path, str) or not path.startswith("/"):
        raise PacketError("the packet's recorder path is not absolute")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise PacketError("the packet's recorder sha256 is not 64 hex characters")
    if (not isinstance(size, str) or not size.isascii() or not size.isdigit()
            or size != str(int(size)) or int(size) <= 0):
        raise PacketError("the packet's recorder bytes is not a canonical positive integer")
    if not isinstance(version, str) or not version:
        raise PacketError("the packet's recorder python version is empty")
    return {"path": path, "argv0.sha256": digest, "argv0.bytes": size, "python": version}


def collect_packet(recorder_python: str | None = None) -> dict:
    """This box's evidence packet, with the device identity attested.

    ``recorder_python`` adds the identity of the interpreter a scratch recorder
    runs as argv[0] on this box.
    """

    try:
        evidence = pb._collect_worker_evidence(attest_accelerator_identity=True)
        identity = pb.executable_toolchain_contract(SEALED_ARGV0)
    except pb.ActionContractError as exc:
        raise PacketError(f"this box cannot attest its facts: {exc}") from None
    packet = dict(vet(evidence))
    packet[ARGV0_KEY] = {"path": SEALED_ARGV0, "sha256": identity["argv0.sha256"],
                         "bytes": identity["argv0.bytes"]}
    argv0_contract(packet)
    if recorder_python is not None:
        try:
            facts = {**pb.executable_toolchain_contract(recorder_python),
                     **pb._probe_python_toolchain(Path(recorder_python))}
        except pb.ActionContractError as exc:
            raise PacketError(f"this box cannot describe {recorder_python}: {exc}") from None
        packet[RECORDER_KEY] = {"path": recorder_python, "sha256": facts["argv0.sha256"],
                                "bytes": facts["argv0.bytes"], "python": facts.get("python", "")}
        recorder_contract(packet)
    return packet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="write the packet to this file in one atomic step "
                         "instead of stdout")
    ap.add_argument("--recorder-python", default=None,
                    help="absolute path of the interpreter a scratch recorder "
                         "runs as argv[0] on this box; its identity joins the packet")
    args = ap.parse_args(argv)
    try:
        packet = collect_packet(args.recorder_python)
    except PacketError as exc:
        print(f"pbevidence: {exc}", file=sys.stderr)
        return 1
    if args.out is None:
        sys.stdout.write(pb._sorted_lf_bytes(packet).decode("utf-8"))
    else:
        # The repo's one owner of rename-atomic JSON records (#1330).
        materialize._write_json_atomic(args.out, packet, trailing_newline=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
