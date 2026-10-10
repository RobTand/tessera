"""Full-vocabulary KL instrument: writer, exact compare and refusals (#1172).

The serve surface cannot supply a 128k-vocabulary JSON dump, so the exact
number comes from an in-runtime logits dump (``experiments/
full_vocab_kl_dump.py``) instead of ``kl_tool dump``. These tests pin the
file contract that dump writes, the exact KL ``compare`` computes, and the
refusals that keep a prefill number from standing in for a decode one.
"""
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import box_artifacts
import numpy as np
import pytest
CHECKOUT = Path(__file__).resolve().parents[1]
INSTRUMENT = CHECKOUT / "experiments" / "full_vocab_kl_dump.py"


def _load_instrument():
    # Same loader the attest tests use: the instrument is a script, not a
    # package, so it loads by path. Before the fix the path is absent and
    # this raises, which is the pre-fix failure.
    spec = importlib.util.spec_from_file_location("_fv_kl_dump", INSTRUMENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fv = _load_instrument()

VOCAB = 64
POSITIONS = 16


def _arrays(seed=0):
    rng = np.random.default_rng(seed)
    teacher = rng.normal(size=(POSITIONS, VOCAB)).astype(np.float32)
    student = teacher + rng.normal(scale=0.05,
                                   size=(POSITIONS, VOCAB)).astype(np.float32)
    return teacher, student

def _meta(regime="prefill", contract="c" * 64, vocab=VOCAB,
          positions=POSITIONS):
    return {
        "schema": "prismaquant.kl_position_dump/2",
        "role": "teacher",
        "teacher_label": "BF16",
        "metric": {"full_vocab": True, "requested_top_k": None},
        "corpus": {"source_sha256": "a" * 64, "tokens": 4096,
                   "contract_sha256": contract},
        "tokenizer": {"identity_sha256": "t" * 64},
        "regime": {"name": regime},
        "model": {"checkpoint": "/mnt/shared/models/X", "arch": "Qwen3",
                  "dtype": "bfloat16"},
        "payload": {"positions": positions, "vocab_size": vocab},
    }


def _write(tmp_path, name, arrays, meta):
    stem = tmp_path / name
    fv.write_full_vocab_payload(str(stem), meta, arrays)
    return str(stem) + ".meta.json"


def test_writer_round_trip_preserves_logprobs(tmp_path):
    teacher, _ = _arrays()
    meta_path = _write(tmp_path, "t", teacher, _meta())
    meta, arr = fv.read_full_vocab_payload(meta_path)
    assert tuple(arr.shape) == (POSITIONS, VOCAB)
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(arr), teacher)
    assert meta["metric"] == {"full_vocab": True, "requested_top_k": None}
    assert meta["payload"]["vocab_size"] == VOCAB


def test_compare_exact_matches_brute_force(tmp_path, capsys):
    teacher, student = _arrays()
    t = _write(tmp_path, "t", teacher, _meta())
    s_meta = _meta()
    s_meta["role"] = "student"
    del s_meta["teacher_label"]
    s = _write(tmp_path, "s", student, s_meta)
    out = tmp_path / "kl.json"
    assert fv.main(["compare", "--teacher", t, "--student", s,
                    "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["metric_identity"]["bound"] == "exact (full vocabulary)"
    assert result["regime"] == "prefill"
    assert result["positions"] == POSITIONS
    lp = teacher.astype(np.float64)
    lq = student.astype(np.float64)
    p = np.exp(lp - lp.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    q = np.exp(lq - lq.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    expect = float((p * (np.log(p) - np.log(q))).sum(axis=1).mean())
    assert result["all"]["kl_mean"] == pytest.approx(expect, rel=1e-6)
    assert "regime=prefill" in capsys.readouterr().out


def test_compare_refuses_a_cross_regime_pair(tmp_path):
    teacher, student = _arrays()
    t = _write(tmp_path, "t", teacher, _meta(regime="prefill"))
    s_meta = _meta(regime="decode")
    s_meta["role"] = "student"
    s = _write(tmp_path, "s", student, s_meta)
    with pytest.raises(SystemExit, match="cross-regime"):
        fv.main(["compare", "--teacher", t, "--student", s])


def test_compare_refuses_a_contract_mismatch(tmp_path):
    teacher, student = _arrays()
    t = _write(tmp_path, "t", teacher, _meta())
    s_meta = _meta(contract="d" * 64)
    s_meta["role"] = "student"
    s = _write(tmp_path, "s", student, s_meta)
    with pytest.raises(SystemExit, match="contract"):
        fv.main(["compare", "--teacher", t, "--student", s])


def test_compare_refuses_a_vocab_mismatch(tmp_path):
    teacher, _ = _arrays()
    student = np.zeros((POSITIONS, VOCAB + 1), dtype=np.float32)
    t = _write(tmp_path, "t", teacher, _meta())
    s_meta = _meta(vocab=VOCAB + 1)
    s_meta["role"] = "student"
    s = _write(tmp_path, "s", student, s_meta)
    with pytest.raises(SystemExit, match="vocab"):
        fv.main(["compare", "--teacher", t, "--student", s])


def test_histories_and_masks_must_match(tmp_path):
    assert fv.histories_and_masks(_meta(), _meta()) is True
    other = _meta(contract="d" * 64)
    assert fv.histories_and_masks(_meta(), other) is False


@box_artifacts.require("kl_instrument", "kl_tool.py")
def test_payload_opens_in_kl_tool_reader(tmp_path):
    import importlib.util
    import sys
    tool_dir = box_artifacts.root("kl_instrument")
    spec = importlib.util.spec_from_file_location(
        "_kl1172_tool", tool_dir / "kl_tool.py")
    kl_tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kl_tool
    spec.loader.exec_module(kl_tool)
    teacher, _ = _arrays()
    meta_path = _write(tmp_path, "t", teacher, _meta())
    meta, arr = kl_tool.read_full_vocab_payload(meta_path)
    assert tuple(arr.shape) == (POSITIONS, VOCAB)
    assert (meta["payload"]["positions"],
            meta["payload"]["vocab_size"]) == (POSITIONS, VOCAB)
