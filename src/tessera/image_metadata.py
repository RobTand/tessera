"""Replay a launcher image declaration without loading the serving runtime.

The launcher records Docker RepoDigests. Replay checks the supplied record and
names its mechanism as a declaration, without consulting Docker or a live pin.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

__all__ = [
    "CENSUS_IMAGE_ENV", "CENSUS_DECLARATION_ENV", "DECLARATION_SCHEMA",
    "DECLARATION_SOURCE", "RuntimeImageError", "declared_reference",
]

#: The two variables a container launcher writes so a process INSIDE the
#: container can check which image it is running in.  The names live here and
#: are read by both sides -- the launcher through the ``container-env``
#: subcommand, the census through :func:`declared_reference` -- so neither
#: spells them a second time.
CENSUS_IMAGE_ENV = "TESSERA_CENSUS_RUNTIME_IMAGE"
CENSUS_DECLARATION_ENV = "TESSERA_CENSUS_RUNTIME_IMAGE_DECLARATION"

#: What a receipt records about HOW its runtime scope was established.  A
#: value a gate switches on, never prose.  It is a DECLARATION and says so:
#: the launcher resolved the reference from docker's ``RepoDigests`` and wrote
#: it into the container's environment, which a host process can also do by
#: hand.  Calling that an attestation would make the receipt claim more than
#: the mechanism delivers, and this tree's rule is that a claim about another
#: runtime is read from a machine-readable table or refused -- a misnamed one
#: is worse than an absent one.
DECLARATION_SCHEMA = "tessera.runtime-image-declaration/1"
DECLARATION_SOURCE = "launcher_repo_digests"


class RuntimeImageError(RuntimeError):
    """An image selection or launcher declaration was refused.

    ``payload`` is the machine-readable refusal -- the same object the CLI
    prints -- so a caller in Python does not have to parse a message.
    """

    def __init__(self, message: str, payload: Mapping[str, Any]):
        super().__init__(message)
        self.payload = dict(payload)


def declared_reference(requested: str, *,
                       env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Check ``requested`` against what the launcher declared, or refuse.

    Returns the declaration a receipt publishes: the mechanism, the image the
    launcher named, the variables it was read from, and the launcher's record
    verbatim.  Raises :class:`RuntimeImageError` when the launcher declared
    nothing, when the pair disagrees with itself, or when the caller names
    other bytes than the ones the container was started on.
    """
    env = os.environ if env is None else env
    declared = (env.get(CENSUS_IMAGE_ENV) or "").strip()
    raw = (env.get(CENSUS_DECLARATION_ENV) or "").strip()
    if not declared or not raw:
        raise RuntimeImageError(
            f"REFUSED: the launcher declared no runtime image for this container.\n"
            f"  requested: {requested}\n"
            f"  {CENSUS_IMAGE_ENV} / {CENSUS_DECLARATION_ENV} are not both set, so the "
            "image is an operator's assertion and a receipt scoped to it would name a "
            "runtime nothing measured.\n"
            "  fix:      run under experiments/tessera_plugin_run.sh, which resolves the "
            "image through docker's RepoDigests and exports both",
            {"refused": True, "reason": "image_declaration_missing",
             "requested": requested, "env": [CENSUS_IMAGE_ENV, CENSUS_DECLARATION_ENV]})
    try:
        record = json.loads(raw)
    except ValueError:
        record = None
    if not isinstance(record, Mapping) or record.get("schema") != "tessera.runtime_image/1":
        raise RuntimeImageError(
            f"REFUSED: {CENSUS_DECLARATION_ENV} does not hold a tessera.runtime_image/1 "
            "record, so there is no table to check the image against",
            {"refused": True, "reason": "image_declaration_unreadable",
             "requested": requested, "env": [CENSUS_DECLARATION_ENV]})
    if record.get("refused"):
        raise RuntimeImageError(
            f"REFUSED: the launcher's own record says it refused this image "
            f"({record.get('reason')}); a container it refused to start declares nothing",
            {"refused": True, "reason": "image_declaration_refused",
             "requested": requested, "declared": declared})
    if record.get("resolved_reference") != declared or declared not in (
            record.get("repo_digests") or []):
        raise RuntimeImageError(
            f"REFUSED: {CENSUS_IMAGE_ENV} is {declared!r}, which the record beside it does "
            f"not resolve to (it resolved {record.get('resolved_reference')!r} out of "
            f"{record.get('repo_digests')!r}); half a forged pair is a forged pair",
            {"refused": True, "reason": "image_declaration_inconsistent",
             "requested": requested, "declared": declared})
    if requested != declared:
        raise RuntimeImageError(
            f"REFUSED: this run was started on {declared}, not on {requested}.\n"
            f"  the digest is a join key of the cell scope, so a receipt naming the other "
            "image would be a covered verdict for a runtime nobody measured.\n"
            f"  fix:      pass --runtime-image \"${CENSUS_IMAGE_ENV}\", the value the "
            "launcher resolved from docker's RepoDigests",
            {"refused": True, "reason": "image_declaration_mismatch",
             "requested": requested, "declared": declared})
    return {"schema": DECLARATION_SCHEMA, "source": DECLARATION_SOURCE, "image": declared,
            "env": [CENSUS_IMAGE_ENV, CENSUS_DECLARATION_ENV], "record": dict(record)}
