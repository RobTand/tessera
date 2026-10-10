"""Full-vocabulary KL instrument: production dump, exact compare, refusals (#1172).

The serve surface cannot supply a 128k-vocabulary JSON dump, so the exact
number comes from an in-runtime logits dump (``experiments/
full_vocab_kl_dump.py``) instead of ``kl_tool dump``. These tests drive the
production ``dump_payload`` itself on a tiny bigram model, then pin the exact
KL ``compare`` computes and the refusals that keep a prefill number from
standing in for a decode one.
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


TINY_VOCAB = 32
TINY_CHUNKS = 2
TINY_SEQLEN = 8
TINY_POSITIONS = TINY_CHUNKS * (TINY_SEQLEN - 1)


def _tiny_corpus(tmp_path, chunks=TINY_CHUNKS, seqlen=TINY_SEQLEN,
                 vocab=TINY_VOCAB, seed=7):
    rng = np.random.default_rng(seed)
    body = {"chunks": [rng.integers(0, vocab, size=seqlen).tolist()
                       for _ in range(chunks)],
            "seqlen": seqlen,
            "source_sha256": "a" * 64,
            "tokens": chunks * seqlen,
            "contract_sha256": "c" * 64,
            "tokenizer": {"identity_sha256": "t" * 64,
                         "vocab_size": vocab}}
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(body))
    return path, body


def _tiny_bigram(vocab=TINY_VOCAB, seed=1):
    """A stateless bigram model: one input token decides one logit row."""
    import torch
    from types import SimpleNamespace

    class TinyBigram(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.emb = torch.nn.Embedding(vocab, 16)
            self.head = torch.nn.Linear(16, vocab)
            with torch.no_grad():
                torch.nn.init.normal_(self.emb.weight, generator=g)
                torch.nn.init.normal_(self.head.weight, generator=g)
                torch.nn.init.normal_(self.head.bias, generator=g)
            self.config = SimpleNamespace(vocab_size=vocab)
            self.calls = []

        def forward(self, input_ids, past_key_values=None, use_cache=False):
            self.calls.append(int(input_ids.numel()))
            logits = self.head(self.emb(input_ids))
            return SimpleNamespace(logits=logits, past_key_values=None)

    return TinyBigram()


def _dump_args(corpus_path, out_stem, role, regime, device="cpu",
               seed_tag=""):
    import argparse
    return argparse.Namespace(
        checkpoint="tiny-bigram" + seed_tag, corpus_contract=str(corpus_path),
        role=role, teacher_label=("BF16-TINY" if role == "teacher" else None),
        out=str(out_stem), regime=regime, dtype="float32", device=device)


def _dump_production(tmp_path, name, role="teacher", regime="prefill",
                     device="cpu", seed=1):
    corpus_path, corpus = _tiny_corpus(tmp_path)
    model = _tiny_bigram(seed=seed).to(device).eval()
    args = _dump_args(corpus_path, tmp_path / name, role, regime,
                      device=device)
    assert fv.dump_payload(args, model) == 0
    meta, arr = fv.read_full_vocab_payload(str(tmp_path / name) + ".meta.json")
    return corpus, model, meta, arr


def test_dump_payload_prefill_writes_mask_rows_and_meta(tmp_path):
    pytest.importorskip("torch")
    corpus, model, meta, arr = _dump_production(tmp_path, "t_pre")
    assert tuple(arr.shape) == (TINY_POSITIONS, TINY_VOCAB)
    assert meta["payload"]["positions"] == TINY_POSITIONS
    assert meta["payload"]["vocab_size"] == TINY_VOCAB
    assert meta["regime"]["name"] == "prefill"
    assert meta["regime"]["forwards"] == TINY_CHUNKS
    assert meta["corpus"]["contract_sha256"] == corpus["contract_sha256"]
    assert (meta["tokenizer"]["identity_sha256"] ==
            corpus["tokenizer"]["identity_sha256"])
    # One full-chunk forward per chunk: the mask scores non-first positions.
    assert model.calls == [TINY_SEQLEN] * TINY_CHUNKS


def test_dump_payload_decode_is_teacher_forced_m1(tmp_path):
    pytest.importorskip("torch")
    corpus, model, meta, arr = _dump_production(tmp_path, "t_dec",
                                                regime="decode")
    assert tuple(arr.shape) == (TINY_POSITIONS, TINY_VOCAB)
    assert meta["regime"]["name"] == "decode"
    assert meta["regime"]["forwards"] == TINY_POSITIONS
    assert meta["regime"]["rows_per_scored_forward"] == 1
    # Every decode forward carries exactly one row: teacher-forced M=1.
    assert model.calls == [1] * TINY_POSITIONS
    # A stateless bigram scores the same row in both regimes.
    _, _, _, pre = _dump_production(tmp_path, "t_pre_cmp", seed=1)
    np.testing.assert_allclose(np.asarray(arr), np.asarray(pre),
                               rtol=1e-5, atol=1e-6)


def test_dump_payload_compare_runs_on_production_files(tmp_path):
    pytest.importorskip("torch")
    _dump_production(tmp_path, "t", seed=1)
    _dump_production(tmp_path, "s", role="student", seed=2)
    out = tmp_path / "kl.json"
    assert fv.main(["compare", "--teacher", str(tmp_path / "t.meta.json"),
                    "--student", str(tmp_path / "s.meta.json"),
                    "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["positions"] == TINY_POSITIONS
    assert result["all"]["kl_mean"] > 0.0


def test_dump_payload_opens_in_kl_tool_reader(tmp_path):
    pytest.importorskip("torch")
    tool_dir = box_artifacts.root("kl_instrument")
    if tool_dir is None or not (tool_dir / "kl_tool.py").exists():
        target = None if tool_dir is None else tool_dir / "kl_tool.py"
        pytest.skip(box_artifacts.reason("kl_instrument", target))
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "_kl1172_tool_prod", tool_dir / "kl_tool.py")
    kl_tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kl_tool
    spec.loader.exec_module(kl_tool)
    _dump_production(tmp_path, "t_prod")
    meta, arr = kl_tool.read_full_vocab_payload(
        str(tmp_path / "t_prod.meta.json"))
    assert tuple(arr.shape) == (TINY_POSITIONS, TINY_VOCAB)


def test_dump_payload_allocates_on_cuda(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.cuda.reset_peak_memory_stats()
    _, _, meta, _ = _dump_production(tmp_path, "t_cu", device="cuda")
    assert meta["execution"]["device"] == "cuda"
    assert torch.cuda.max_memory_allocated() > 0


def test_histories_and_masks_refuses_missing_identity(tmp_path):
    full = _meta()
    for key in ("corpus", "tokenizer", "payload"):
        partial = {k: v for k, v in full.items() if k != key}
        assert fv.histories_and_masks(full, partial) is False
        assert fv.histories_and_masks(partial, full) is False
    # Two payloads that both lack an identity field never match: before the
    # fix None equalled None and this pair passed.
    thin_a = {"payload": {"positions": POSITIONS, "vocab_size": VOCAB}}
    thin_b = {"payload": {"positions": POSITIONS, "vocab_size": VOCAB}}
    assert fv.histories_and_masks(thin_a, thin_b) is False
