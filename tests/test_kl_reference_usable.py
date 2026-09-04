"""A KL is only a number when the reference has an opinion.

Twice a checkpoint served, produced a KL, and the KL was noise because the BF16
teacher was nearly uniform -- and both times the teacher's own dump already
held the evidence.  ``experiments/kl_reference_usable.py`` reads that evidence.
These tests build synthetic dumps whose answer is known, so the refusals are
tested for what they claim rather than against a checkpoint that might move.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "experiments" / "kl_reference_usable.py"

N_CHUNKS, SEQLEN = 2, 8
N_POS = N_CHUNKS * (SEQLEN - 1)
K = 6


def write_corpus(path: Path, contract_sha: str, tokenizer_path: str, *,
                 prepends_bos: bool = False) -> np.ndarray:
    rng = np.random.default_rng(0)
    chunks = [rng.integers(0, 500, size=SEQLEN).tolist() for _ in range(N_CHUNKS)]
    targets = np.concatenate([
        np.asarray(c if prepends_bos else c[1:], dtype=np.int64) for c in chunks
    ])
    path.write_text(
        json.dumps(
            {
                "schema": "prismaquant.kl_corpus_contract/1",
                "n_chunks": N_CHUNKS,
                "seqlen": SEQLEN,
                "scored_positions": len(targets),
                "chunks": chunks,
                "prepends_bos": prepends_bos,
                "contract_sha256": contract_sha,
                "tokenizer": {"path": tokenizer_path, "identity_sha256": contract_sha},
            }
        )
    )
    return targets


def write_dump(path: Path, targets: np.ndarray, *, peaked: bool, contract_sha: str,
               tokenizer_path: str = "/models/toy") -> None:
    """A dump whose top-1 is the true token (peaked) or is not (flat)."""
    n = len(targets)          # sized by the targets, so a short dump IS short
    ids = np.zeros((n, K), dtype=np.int32)
    lps = np.zeros((n, K), dtype=np.float32)
    for i, t in enumerate(targets):
        others = [int(x) for x in range(900, 900 + K) if x != t][: K - 1]
        if peaked:
            row_ids = [int(t)] + others
            probs = np.array([0.90] + [0.02] * (K - 1))
        else:
            # the true token is last, and nothing is confident
            row_ids = others + [int(t)]
            probs = np.full(K, 0.01, dtype=np.float64)
        ids[i] = row_ids
        lps[i] = np.log(probs)
    meta = {
        "role": "teacher",
        "tokenizer": {"path": tokenizer_path},
        "corpus": {"contract_sha256": contract_sha},
    }
    np.savez(path, ids=ids, lps=lps, meta=np.array(json.dumps(meta)))


def run(dump: Path, corpus: Path | None = None, *extra: str):
    argv = [sys.executable, str(TOOL), str(dump)]
    if corpus is not None:
        argv.append(str(corpus))
    argv += list(extra)
    return subprocess.run(argv, capture_output=True, text=True)


@pytest.fixture
def toy(tmp_path):
    corpus = tmp_path / "corpus.json"
    targets = write_corpus(corpus, "aaaa1111", "/models/toy")
    return tmp_path, corpus, targets


def test_a_peaked_reference_is_usable(toy):
    tmp_path, corpus, targets = toy
    dump = tmp_path / "good.npz"
    write_dump(dump, targets, peaked=True, contract_sha="aaaa1111")
    # K=6 cannot hold 60% of a real vocab's mass; ask only what the toy can show
    proc = run(dump, corpus, "--min-support-mass", "0.5")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "usable as a KL reference" in proc.stdout
    assert "next-token top-1 accuracy                      100.00%" in proc.stdout


def test_a_prepended_bos_reference_scores_every_corpus_token(tmp_path):
    """A corpus that prepends BOS has a predecessor for its first token.

    ``kl_tool dump`` therefore emits one scored position per corpus token,
    rather than dropping the first token of every chunk as an unconditioned
    position.  The reference gate must construct the same target population.
    """
    corpus = tmp_path / "bos_corpus.json"
    targets = write_corpus(
        corpus, "bos11111", "/models/toy-bos", prepends_bos=True
    )
    dump = tmp_path / "good_bos.npz"
    write_dump(
        dump, targets, peaked=True, contract_sha="bos11111",
        tokenizer_path="/models/toy-bos",
    )

    proc = run(dump, corpus, "--min-support-mass", "0.5")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"positions {N_CHUNKS * SEQLEN}" in proc.stdout
    assert "next-token top-1 accuracy                      100.00%" in proc.stdout


def test_a_flat_reference_is_refused_for_three_reasons(toy):
    tmp_path, corpus, targets = toy
    dump = tmp_path / "flat.npz"
    write_dump(dump, targets, peaked=False, contract_sha="aaaa1111")
    proc = run(dump, corpus)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "REFUSED as a KL reference" in proc.stdout
    assert "mass is inside the compared support" in proc.stdout
    assert "no position is confident" in proc.stdout
    assert "does not predict its own corpus" in proc.stdout
    assert "plumbing receipt, not a quality number" in proc.stdout


def test_the_true_token_rank_is_reported_not_just_the_argmax(toy):
    """The flat dump puts the true token LAST of K.  Rank, not accuracy alone,
    is what distinguishes 'nearly right' from 'ranks the truth below
    everything' -- the GLM cuts were the second."""
    tmp_path, corpus, targets = toy
    dump = tmp_path / "flat.npz"
    write_dump(dump, targets, peaked=False, contract_sha="aaaa1111")
    proc = run(dump, corpus)
    assert f"median rank of the true token                  {K - 1}" in proc.stdout


def test_a_corpus_from_another_tokenizer_is_refused(toy):
    """A mismatched corpus and a broken model look identical from the numbers,
    so the identity is compared instead: the GLM contract against the Qwen dump
    scored 1.54% and read exactly like a refusal."""
    tmp_path, corpus, targets = toy
    other = tmp_path / "other_corpus.json"
    write_corpus(other, "bbbb2222", "/models/other")
    dump = tmp_path / "good.npz"
    write_dump(dump, targets, peaked=True, contract_sha="aaaa1111")
    proc = run(dump, other)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "the dump was taken on corpus aaaa1111" in proc.stderr
    assert "/models/toy" in proc.stderr and "/models/other" in proc.stderr


def test_it_runs_without_a_corpus_and_says_less(toy):
    """Without the corpus there is no next-token accuracy -- the load-bearing
    number -- so the tool reports the distribution shape and nothing it cannot
    know."""
    tmp_path, corpus, targets = toy
    dump = tmp_path / "flat.npz"
    write_dump(dump, targets, peaked=False, contract_sha="aaaa1111")
    proc = run(dump)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "next-token top-1" not in proc.stdout
    assert "no position is confident" in proc.stdout


def test_a_position_count_mismatch_is_refused(toy):
    tmp_path, corpus, targets = toy
    dump = tmp_path / "short.npz"
    write_dump(dump, targets[:-3], peaked=True, contract_sha="aaaa1111")
    proc = run(dump, corpus)
    assert proc.returncode != 0
    assert "not the same contract" in proc.stderr
