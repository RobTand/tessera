"""A held-out Hessian must not be able to shape bytes.

``capture_h_full.py --eval-h-out`` writes the held-out second moment in the
same shape as the fit capture and carrying the *same* three
``HESSIAN_IDENTITY`` fields.  That is deliberate: a scorer has to be able to
prove the two halves came from one split, and the split is named by those
fields.  It is also exactly what makes the file dangerous -- it is
indistinguishable from the fit capture to every guard downstream, so an encode
against it would be stamped with the fit capture's identity and then scored on
the rows it was fitted to.

The separating fact is a machine-readable ``hessian_role``.  The payload's
``role`` field is prose and no guard reads prose, which is the same reason the
serving-lane route status is a structured field and not a sentence.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.export import ActivationSource, GrammarError, HESSIAN_IDENTITY


def _payload(tmp_path, name, **extra):
    provenance = {"text_sha256": "abc", "fit_tokens": 16384,
                  "fit_ids_sha256": "def", "eval_ids_sha256": "ghi", **extra}
    path = tmp_path / name
    torch.save({"H": {"unit": torch.eye(4)}, "provenance": provenance}, path)
    return path


def test_a_held_out_capture_is_refused_as_an_encode_source(tmp_path):
    path = _payload(tmp_path, "eval.pt", hessian_role="held-out")
    with pytest.raises(GrammarError) as caught:
        ActivationSource.from_capture(path)
    message = str(caught.value)
    assert "held-out" in message and "must not shape bytes" in message


def test_the_identity_fields_alone_cannot_tell_the_two_apart(tmp_path):
    """Why the marker has to exist: the guard that was there passes both."""
    fit = _payload(tmp_path, "fit.pt")
    ev = _payload(tmp_path, "eval.pt", hessian_role="held-out")
    import torch as _t
    a, b = (_t.load(p, map_location="cpu", weights_only=False)["provenance"]
            for p in (fit, ev))
    assert all(a[f] == b[f] for f in HESSIAN_IDENTITY)


def test_a_capture_written_before_the_marker_still_loads(tmp_path):
    """Absent means a fit capture from before the field existed, not unknown."""
    assert ActivationSource.from_capture(_payload(tmp_path, "old.pt"))


def test_an_explicit_fit_marker_loads(tmp_path):
    assert ActivationSource.from_capture(
        _payload(tmp_path, "fit.pt", hessian_role="fit"))
