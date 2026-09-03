"""Which container image a serve harness is allowed to run, and the refusal.

WHY THIS EXISTS (issue #100).  Every serve harness in ``experiments/`` named
``vllm/vllm-openai:latest`` and the docs called it the pinned runtime.  A tag
is not a pin: it is a name upstream can repoint, and two boxes can hold two
builds under it while every receipt records the same four words.  A two-box A/B
under a floating tag can put two runtimes on two arms and write down nothing
that would say so.

The defect is not that the boxes disagreed -- measured on 2026-09-03 they did
not, both carrying ``vllm/vllm-openai@sha256:61fc8a89...`` -- it is that nothing
in the system could have told us either way.  This module is what tells us.

THE TRAP, AND IT IS THE WHOLE MODULE.  ``docker image inspect``'s ``.Id`` is
**not** a stable name for an image across boxes.  On sparky (docker 29 with the
containerd snapshotter) ``.Id`` is the *manifest* digest; on sparklina
(overlay2) it is the *config* digest.  The same image reads ``61fc8a896b0a``
there and ``89154ef00dd1`` here.  Issue #100 was filed off exactly that
difference, and a gate that compared ``.Id`` would have refused sparklina
forever for holding identical bytes -- a refusal that permanently disables one
box is not a fix.  What is stable is ``RepoDigests``: the manifest digests the
image was pulled under, which both boxes report identically.  So the pin is a
digest *reference*, the check is membership in ``RepoDigests``, and the local
``.Id`` is recorded as provenance only.  ``docs/measurements/tessera-serving-
plugin-2026-09-02.md`` (section 9) had already measured this; the gate had to
be built to know it.

PRINCIPLE 9, AND WHY THIS REFUSES RATHER THAN WARNS.  A warning nothing reads
is a confession log, not a gate, and this repo has been bitten by precisely
that.  :func:`require_pinned` raises; the CLI exits 2 and prints one JSON line
on stdout so a *program* -- not only a human reading a log -- can read the
refusal, the resolved digests, and the exact ``docker pull`` that fixes it.

SCOPE.  The pin governs one repository: the vanilla vLLM image named by the
packaged contract's ``versions.attested_on.image``.  A harness that runs some
other image -- ``prismaquant/glm53-mia-sm121``, say, which is a different
runtime and has no pin -- is *resolved and stamped* but not refused: gating it
against a pin that does not exist would break GLM serves to enforce nothing.
Widening the pinned set is a decision, not an inference; add repositories here
only when someone has priced them.

ONE LITERAL.  The digest lives in ``runtime_contract.json`` and nowhere else.
That is deliberate and it is principle 14's shape: the contract is where this
package says what runtime it was attested on, so moving the pin means
re-attesting, and a second copy of a 64-hex string is a second thing to forget
to update.  Nothing else -- not a harness, not a test, not a doc -- may hold
the digest; they read it from here.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any, Callable, Mapping

__all__ = [
    "PIN_CONTRACT_FIELD",
    "RuntimeImageError",
    "docker_inspector",
    "parse_reference",
    "pinned_reference",
    "resolve",
    "require_pinned",
]

#: Where the one literal lives.  Read, never copied.
PIN_CONTRACT_FIELD = ("versions", "attested_on", "image")

#: A pull reference that names *bytes*: ``repo[:port]/name@sha256:<64 hex>``.
#: A tag reference does not match, which is the entire point of the pin.
_DIGEST_REFERENCE = re.compile(
    r"^(?P<repository>[a-z0-9][a-z0-9._/-]*[a-z0-9])@(?P<digest>sha256:[0-9a-f]{64})$")

#: A tag reference, for naming what was found when the pin is malformed.
_TAG_REFERENCE = re.compile(
    r"^(?P<repository>[a-z0-9][a-z0-9._/-]*[a-z0-9])(?::(?P<tag>[\w][\w.-]*))?$")


class RuntimeImageError(RuntimeError):
    """A serve was asked to run an image the pin does not allow.

    ``payload`` is the machine-readable refusal -- the same object the CLI
    prints -- so a caller in Python does not have to parse a message.
    """

    def __init__(self, message: str, payload: Mapping[str, Any]):
        super().__init__(message)
        self.payload = dict(payload)


# -------------------------------------------------------------- the pin ---

def pinned_reference(contract: Mapping[str, Any] | None = None) -> str:
    """The pinned pull reference, read from the packaged contract.

    Refuses a tag: a contract that names ``vllm/vllm-openai:latest`` here is
    not pinning anything, and letting it through would make every downstream
    check vacuous while still reading like a gate.
    """
    if contract is None:
        from tessera.serving.contract import contract_path

        contract = json.loads(contract_path().read_text(encoding="utf-8"))
    node: Any = contract
    for key in PIN_CONTRACT_FIELD:
        if not isinstance(node, Mapping) or key not in node:
            raise RuntimeImageError(
                f"runtime_contract.json has no {'.'.join(PIN_CONTRACT_FIELD)}: "
                "the serve-image pin lives there and nowhere else",
                {"refused": True, "reason": "pin_missing"})
        node = node[key]
    if not isinstance(node, str) or not _DIGEST_REFERENCE.match(node):
        raise RuntimeImageError(
            f"runtime_contract.json {'.'.join(PIN_CONTRACT_FIELD)} is {node!r}, "
            "which is not a digest reference (repository@sha256:<64 hex>). A tag "
            "is a name upstream can repoint, not a pin.",
            {"refused": True, "reason": "pin_not_a_digest", "pinned": node})
    return node


def parse_reference(reference: str) -> tuple[str, str | None, str | None]:
    """``(repository, tag, digest)`` for a pull reference; tag/digest may be None.

    An unparsable reference yields ``(reference, None, None)`` rather than
    raising: the caller's job is to decide whether the *pinned* repository is
    involved, and an image name this does not understand is definitionally not
    the pinned one.
    """
    if (m := _DIGEST_REFERENCE.match(reference)):
        return m.group("repository"), None, m.group("digest")
    if (m := _TAG_REFERENCE.match(reference)):
        return m.group("repository"), m.group("tag"), None
    return reference, None, None


# ------------------------------------------------------------ inspection ---

def docker_inspector(reference: str) -> dict[str, Any]:
    """What the local daemon holds for ``reference``, or ``present: False``.

    Two fields, and the asymmetry between them is the point: ``repo_digests``
    names bytes and is comparable across boxes; ``local_id`` is what this
    daemon's image store happens to call them and is comparable with nothing.
    """
    proc = subprocess.run(
        ["docker", "image", "inspect", reference,
         "--format", "{{.Id}}\t{{json .RepoDigests}}"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return {"present": False, "local_id": None, "repo_digests": [],
                "error": (proc.stderr or proc.stdout).strip().splitlines()[-1:] or None}
    local_id, _, digests = proc.stdout.strip().partition("\t")
    try:
        parsed = json.loads(digests)
    except ValueError:
        parsed = []
    return {"present": True, "local_id": local_id or None,
            "repo_digests": sorted(parsed or []), "error": None}


# --------------------------------------------------------------- resolve ---

def resolve(requested: str, *,
            inspector: Callable[[str], Mapping[str, Any]] = docker_inspector,
            contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ``requested`` to the bytes it names, and say whether they pass.

    Always returns; never raises on a mismatch.  :func:`require_pinned` is the
    enforcing wrapper, so a caller that only wants the receipt fields (a
    non-pinned image, say) does not have to catch an exception to get them.
    """
    pinned = pinned_reference(contract)
    pinned_repo, _, pinned_digest = parse_reference(pinned)
    repository, tag, digest = parse_reference(requested)

    found = dict(inspector(requested))
    repo_digests = list(found.get("repo_digests") or [])
    # One image can carry several manifest digests (the same bytes pulled under
    # two repositories, or re-tagged).  Report the one for the repository that
    # was actually asked for; an image can be a legitimate match on one of its
    # names and irrelevant under another.
    own = [ref for ref in repo_digests if parse_reference(ref)[0] == repository]
    resolved_digest = parse_reference(own[0])[2] if own else None

    gated = repository == pinned_repo
    record: dict[str, Any] = {
        "schema": "tessera.runtime_image/1",
        "requested": requested,
        "requested_tag": tag,
        "pinned": pinned,
        "gated": gated,
        "present": bool(found.get("present")),
        "repo_digests": repo_digests,
        "resolved_digest": resolved_digest,
        "local_id": found.get("local_id"),
        "refused": False,
        "reason": None,
        "fix": None,
    }
    if not gated:
        # Stamped, not gated -- see SCOPE in the module docstring.
        record["reason"] = "not_pinned_repository"
        return record
    if not found.get("present"):
        record.update(refused=True, reason="image_absent",
                      fix=f"docker pull {pinned}")
        return record
    if pinned not in repo_digests:
        record.update(refused=True, reason="image_pin_mismatch",
                      fix=f"docker pull {pinned}")
        return record
    record["reason"] = "pinned"
    return record


def require_pinned(requested: str, *,
                   inspector: Callable[[str], Mapping[str, Any]] = docker_inspector,
                   contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """:func:`resolve`, but a mismatch is a refusal rather than a field."""
    record = resolve(requested, inspector=inspector, contract=contract)
    if record["refused"]:
        raise RuntimeImageError(_message(record), record)
    return record


def _message(record: Mapping[str, Any]) -> str:
    if record["reason"] == "image_absent":
        held = "this box does not hold that image at all"
    else:
        held = ("it holds " + ", ".join(record["repo_digests"])
                if record["repo_digests"] else "it holds no manifest digest for it")
    return (
        f"REFUSED: {record['requested']} is not the pinned serving image.\n"
        f"  pinned:   {record['pinned']}\n"
        f"  this box: {held}\n"
        f"  local id: {record['local_id']} (box-local; never compare it across boxes)\n"
        f"  fix:      {record['fix']}")


# ------------------------------------------------------------------- cli ---

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tessera.serving.runtime_image",
        description="Resolve a serve image to the bytes it names, and refuse a "
                    "floating tag on the pinned repository.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pin", help="print the pinned pull reference")
    res = sub.add_parser("resolve", help="resolve an image; exit 2 if refused")
    res.add_argument("--image", required=True)
    res.add_argument("--allow-unpinned", action="store_true",
                     help="do not exit 2 for an image outside the pinned "
                          "repository (the default already permits it)")
    args = parser.parse_args(argv)

    try:
        if args.command == "pin":
            print(pinned_reference())
            return 0
        record = resolve(args.image)
    except RuntimeImageError as exc:
        print(json.dumps(exc.payload, sort_keys=True), flush=True)
        print(exc, file=sys.stderr)
        return 2
    # The refusal is on stdout as JSON so a program reads it, and in prose on
    # stderr so a person does.  Both, always: a gate only a human can read gets
    # skipped by the script that should have honoured it.
    print(json.dumps(record, sort_keys=True), flush=True)
    if record["refused"]:
        print(_message(record), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
