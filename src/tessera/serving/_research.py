"""tessera#508 research instrumentation for the GLM53 NoPE backend.

Both knobs are read once at import and are unset in every production path:

* ``TESSERA_RESEARCH_GLM53_NOPE_SYNC=1`` turns Tessera-owned sites into device
  sync points. An asynchronous CUDA fault surfaces at the next sync, so a fault
  raised at a site happened between the previous sync and that site. Skipped
  while a stream is capturing (a sync is illegal in capture).
* ``TESSERA_RESEARCH_GLM53_NOPE_DUMP=<path>`` appends one JSON line per call
  with a digest of every tensor handed to :func:`dump`, so two repeats of one
  request can be diffed call by call (which input or output first differs).
  Each digest is a device sync; research only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import torch

SYNC = bool(os.environ.get("TESSERA_RESEARCH_GLM53_NOPE_SYNC"))
DUMP = os.environ.get("TESSERA_RESEARCH_GLM53_NOPE_DUMP") or None


def sync(site: str) -> None:
    if not SYNC:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 -- name the site, then re-raise
        raise RuntimeError(f"#508 sync site {site}: {exc}") from exc


def _digest(t: torch.Tensor) -> str:
    t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    return hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def dump(site: str, **fields) -> None:
    if DUMP is None:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    rec: dict = {"t": time.time(), "site": site}
    for key, value in fields.items():
        if isinstance(value, torch.Tensor):
            rec[key] = {"shape": list(value.shape),
                        "dtype": str(value.dtype).replace("torch.", ""),
                        "sha": _digest(value)}
        else:
            rec[key] = value
    with open(DUMP, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
