"""Pad-bit census over every ``.tessera`` artifact on this box.

Padding canonicality (schema D4, S3c rule 3) is enforced in two places and was
*declared* in a third that nothing called (#23).  Before rewiring which
function owns the rule, the question that has to be answered with a number
rather than an argument is: does any artifact already written depend on the
rule being unenforced?

The answer, at the time #23 was closed, is no -- and by a wider margin than the
issue's baseline claimed:

    artifacts=22 refusals=0 planes=198 non_zero_sub_byte_pad_bits=0
      non_zero_alignment_pad_bits=0
    planes=198 non_empty=74 with_sub_byte_slack=0 with_alignment_pad=0

Not one of the 74 non-empty planes on disk even *has* a partial final byte or an
alignment tail, so no tightening of the pad rule can reject a written artifact.

CPU only, by construction -- it runs while a GPU measurement is in flight.
"""

from __future__ import annotations

import glob
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from tessera.container import parse, plane_ranges

ARTIFACT_GLOBS = (
    "/home/rob/tessera-runs/*/*/cache/wire/*.tessera",
    "/home/rob/tessera-runs/*/cache/wire/*.tessera",
)


def main() -> int:
    paths = sorted({p for pattern in ARTIFACT_GLOBS for p in glob.glob(pattern)})
    planes = non_empty = slack_planes = align_planes = 0
    dirty_slack = dirty_align = refusals = 0
    for path in paths:
        with open(path, "rb") as handle:
            data = handle.read()
        try:
            artifact = parse(data)
        except Exception as exc:  # a refusal is the measurement, not a crash
            refusals += 1
            print(f"REFUSED {path}: {type(exc).__name__}: {exc}")
            continue
        order = {kind: i for i, kind in enumerate(artifact.manifest.plane_order)}
        for descriptor, offset, content, total in plane_ranges(
            artifact.manifest, artifact.terminal
        ):
            planes += 1
            chunk = artifact.plane_region[offset : offset + total]
            if content:
                non_empty += 1
            if total > content:
                align_planes += 1
                dirty_align += sum(bin(byte).count("1") for byte in chunk[content:])
            bits = (
                artifact.terminal.plane_elements[order[descriptor.kind]]
                * descriptor.element_bits
            )
            slack = (-bits) % 8
            if slack and content:
                slack_planes += 1
                dirty_slack += bin(chunk[content - 1] & ((1 << slack) - 1)).count("1")
    print(
        f"artifacts={len(paths)} refusals={refusals} planes={planes} "
        f"non_zero_sub_byte_pad_bits={dirty_slack} "
        f"non_zero_alignment_pad_bits={dirty_align}"
    )
    print(
        f"planes={planes} non_empty={non_empty} with_sub_byte_slack={slack_planes} "
        f"with_alignment_pad={align_planes}"
    )
    return 1 if refusals else 0


if __name__ == "__main__":
    raise SystemExit(main())
