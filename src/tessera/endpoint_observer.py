"""Read the live serving producer and publish immutable JSON receipts.

This module imports no serving module. An input directory, alias, lease,
argument, or worker file cannot supply a runtime observation through this API.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from tessera import endpoint_witness as ew


ROUTE = "/tessera/endpoint-witness"


def fetch_witness(base_url, *, publish=False, timeout_s=30):
    """Ask the listener to observe its own engine through a fresh collective RPC."""
    parsed = urlsplit(base_url)
    ew.require(parsed.scheme in ("http", "https") and parsed.hostname and parsed.port
               and parsed.path in ("", "/") and not parsed.username and not parsed.query
               and not parsed.fragment, "runtime listener address is not a direct HTTP address")
    request_id = secrets.token_hex(16)
    request = urllib.request.Request(base_url.rstrip("/") + ROUTE + "?request_id=" + request_id,
                                     method="POST" if publish else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            envelope = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read())
            reason = detail.get("reason", str(exc)) if isinstance(detail, dict) else str(exc)
        except (OSError, ValueError):
            reason = str(exc)
        raise ValueError(f"runtime evidence is unavailable (HTTP {exc.code}): {reason}") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"runtime evidence is unavailable: {exc}") from exc
    ew.fields(envelope, ("receipt", "public_receipt_path"), "runtime response")
    ew.check_join(envelope["receipt"])
    ew.require(envelope["receipt"]["lifetime"]["request_id"] == request_id,
               "response does not observe this fresh runtime request")
    observed = urlsplit(envelope["receipt"]["listener"]["endpoint"])
    ew.require(observed.port == parsed.port, "observed listener port differs from the contacted endpoint")
    # The socket address comes from the server's ASGI scope, not the Host header.
    # A DNS name may resolve to that address. The response must still name a
    # concrete socket address, and the verifier must query that endpoint again.
    if publish:
        ew.text(envelope["public_receipt_path"], "public runtime receipt path")
    else:
        ew.require(envelope["public_receipt_path"] is None, "observation unexpectedly publishes a receipt")
    return envelope


def publish_witness(publish_root, *, witness):
    """Publish one complete observation without overwrites or partial JSON visibility."""
    ew.check_join(witness)
    roots = {model["model_path"] for rank in witness["artifacts"] for model in rank["models"]}
    ew.require(len(roots) == 1, "publication requires one loaded artifact root")
    ew.prove_files(witness, roots.pop(), witness["tokenizer"]["path"])
    root = Path(publish_root)
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / f"endpoint_runtime_witness.v1.{witness['fingerprint']}.json"
    raw = (ew.canonical(witness) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".endpoint-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, receipt)
        except FileExistsError:
            ew.require(receipt.read_bytes() == raw, "immutable runtime receipt already has different bytes")
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt
