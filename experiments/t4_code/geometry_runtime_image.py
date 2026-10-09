"""The D41 wrapper's image identity seam, using the existing D32 stamp owner."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys

from tessera.dev_mode import seal_check
from tessera.serving.runtime_image import RuntimeImageError, _message, docker_inspector, resolve


def measurement_image(image, *, inspector=docker_inspector, contract=None):
    """Resolve actual Docker facts; identity drift stamps, absence still refuses."""
    record = resolve(image, inspector=inspector, contract=contract)
    if record["refused"]:
        refusal = RuntimeImageError(_message(record), record)
        if record["reason"] not in ("image_digest_mismatch", "image_pin_mismatch"):
            raise refusal
        seal_check("measurement image identity", record["required"],
                   record["resolved_reference"], where="D41 T4 wrapper",
                   refusal=refusal, same=False)
        record["identity_refusal"] = record["reason"]
        record["refused"] = False
        record["reason"] = "development_identity_stamp"
    elif record.get("dev_uncertified"):
        # resolve already stamped a default-pin mismatch (D32) and continued: it is the one owner
        # of that stamp, so report it in this record's shape instead of stamping it twice.
        record["identity_refusal"] = record["reason"]
        record["reason"] = "development_identity_stamp"
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args(argv)
    try:
        # Keep stdout one machine-readable JSON record; stamps remain in logs.
        with redirect_stdout(sys.stderr):
            record = measurement_image(args.image)
    except RuntimeImageError as exc:
        print(json.dumps(exc.payload))
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
