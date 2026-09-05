"""The decode-to-prefill row mapping is the corpus contract's, not a convention.

``experiments/decode_regime_subset.py`` publishes the prefill regime restricted
to the decode regime's positions, and prints that "the only remaining
difference ... is which forward ran".  It computed the row of chunk ``c``'s
prefix length ``L`` as ``c * (seqlen - 1) + (L - 1)`` -- correct only for a
corpus that does NOT prepend BOS, because a corpus that does contributes
``seqlen`` scored prefill rows per chunk and not ``seqlen - 1`` (tessera#249).
On a BOS corpus every later chunk's positions folded into the preceding chunk,
in bounds, and the wrong number was published under the matched-position
claim.  ``experiments/kl_reference_usable.py`` already derived the same
distinction from the same field.

These payloads are toys with a known answer: the teacher and student agree
exactly on the even prefill rows and disagree on the odd ones, so the correct
BOS mapping (``[0, 2, 4, 6]``) reads KL 0 and the assumed one (``[0, 2, 3, 5]``)
cannot.  The identity refusals are built the same way -- one field changed at a
time on an otherwise valid triple.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import box_artifacts

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "experiments" / "decode_regime_subset.py"
KL_TOOL_DIR = box_artifacts.root("kl_instrument")
box_artifacts.require_module("kl_instrument", "kl_tool.py")
box_artifacts.require_module("kl_instrument", "kl_estimator.py")

N_CHUNKS, SEQLEN, K = 2, 4, 4
CONTRACT = "d" * 64
TOKENIZER = {"schema": "prismaquant.tokenizer_identity/1",
             "path": "/models/toy",
             "files": {"tokenizer.json": "a" * 64},
             "vocab_size": 512, "bos_token_id": 1}
ARTIFACT = "/runs/armA"

#: The decode sweep both chunks were scored at: prefix lengths 1 and 3.
PREFIXES = [[1, 3], [1, 3]]


def payload_meta(role, regime, *, scored, artifact=ARTIFACT, contract=CONTRACT,
                 tokenizer=None, n_chunks=N_CHUNKS, seqlen=SEQLEN,
                 prefixes=None):
    meta = {
        "schema": "prismaquant.kl_dump/2",
        "role": role,
        "corpus": {"source_text": "/corpus.txt", "n_chunks": n_chunks,
                   "seqlen": seqlen, "tokens": n_chunks * seqlen,
                   "scored_positions": scored, "contract_sha256": contract},
        "tokenizer": tokenizer or TOKENIZER,
        "model": {"served_model_name": "kl-target", "artifact_path": artifact},
        "regime": {"name": regime},
    }
    if regime == "decode":
        meta["regime"].update(
            prefix_lengths=prefixes if prefixes is not None else PREFIXES,
            scored_positions=sum(len(p) for p in (prefixes or PREFIXES)))
    return meta


def write_payload(path: Path, meta: dict, probs: np.ndarray) -> Path:
    ids = np.tile(np.arange(K, dtype=np.int32), (probs.shape[0], 1))
    with np.errstate(divide="ignore"):
        lps = np.log(probs).astype(np.float32)      # an all-zero row is -inf
    np.savez(path, ids=ids, lps=lps, meta=np.array(json.dumps(meta)))
    return path


def rows_of(n: int, *, disagree: set[int] | None = None,
            blank: set[int] | None = None) -> np.ndarray:
    """One distribution per row; ``disagree`` rows are reversed, ``blank`` rows
    hold no finite logprob at all."""
    base = np.array([0.4, 0.3, 0.2, 0.1])
    out = np.tile(base, (n, 1))
    for row in (disagree or ()):
        out[row] = base[::-1]
    for row in (blank or ()):
        out[row] = 0.0
    return out


def build(tmp_path, *, prepends_bos, teacher_meta=None, student_meta=None,
          decode_meta=None, teacher_rows=None, student_rows=None,
          teacher_blank=None):
    """A valid triple over a BOS or a no-BOS corpus, with optional overrides."""
    rows_per_chunk = SEQLEN if prepends_bos else SEQLEN - 1
    scored = N_CHUNKS * rows_per_chunk
    n_t = teacher_rows if teacher_rows is not None else scored
    n_s = student_rows if student_rows is not None else scored
    teacher = write_payload(
        tmp_path / "teacher.npz",
        teacher_meta or payload_meta("teacher", "prefill", scored=scored,
                                     artifact="/models/bf16"),
        rows_of(n_t, blank=teacher_blank))
    student = write_payload(
        tmp_path / "student.npz",
        student_meta or payload_meta("student", "prefill", scored=scored),
        rows_of(n_s, disagree={r for r in range(n_s) if r % 2}))
    decode = write_payload(
        tmp_path / "decode.npz",
        decode_meta or payload_meta("student", "decode", scored=scored),
        rows_of(sum(len(p) for p in PREFIXES)))
    return teacher, student, decode


def run(tmp_path, teacher, student, decode, *extra):
    out = tmp_path / "subset.json"
    env = os.environ | {"KL_TOOL_DIR": str(KL_TOOL_DIR)}
    proc = subprocess.run(
        [sys.executable, str(TOOL),
         "--teacher-prefill", str(teacher),
         "--student-prefill", str(student),
         "--decode-student", str(decode),
         "--json", str(out), *extra],
        capture_output=True, text=True, env=env)
    record = json.loads(out.read_text()) if out.exists() else None
    return proc, record


def test_a_bos_corpus_maps_to_the_bos_row_stride(tmp_path):
    """Two chunks of four with a prepended BOS score eight prefill rows, so
    chunk 1's prefix lengths 1 and 3 are rows 4 and 6 -- not 3 and 5."""
    proc, record = run(tmp_path, *build(tmp_path, prepends_bos=True),
                       "--seqlen", str(SEQLEN))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert record["prefill_on_decode_positions"]["kl_lower_mean"] == 0.0, (
        "the decode positions are the rows the teacher and student agree on; "
        "a non-zero KL here is the wrong subset")
    assert record["prefill_rows"] == [0, 2, 4, 6]
    assert record["alignment"]["prepends_bos"] is True
    assert record["alignment"]["prefill_rows_per_chunk"] == SEQLEN


def test_a_no_bos_corpus_keeps_the_mapping_it_had(tmp_path):
    """The historical Qwen contract prepends nothing, and its mapping is
    unchanged: six rows over two chunks, prefix 1 and 3 of chunk 1 at 3 and 5."""
    proc, record = run(tmp_path, *build(tmp_path, prepends_bos=False),
                       "--seqlen", str(SEQLEN))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert record["prefill_rows"] == [0, 2, 3, 5]
    assert record["alignment"]["prepends_bos"] is False
    assert record["alignment"]["prefill_rows_per_chunk"] == SEQLEN - 1


def test_a_seqlen_that_contradicts_the_contract_refuses(tmp_path):
    """``--seqlen`` is a cross-check on the contract, never the mapping's
    source: a wrong but in-bounds value used to move every row silently."""
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True), "--seqlen", "5")
    assert proc.returncode != 0, proc.stdout
    assert "contradicts the corpus contract" in proc.stderr


def test_a_second_corpus_refuses(tmp_path):
    """Same shapes, different corpus: the indices line up and mean nothing."""
    other = payload_meta("student", "prefill", scored=N_CHUNKS * SEQLEN,
                         contract="e" * 64)
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True,
                                   student_meta=other))
    assert proc.returncode != 0, proc.stdout
    assert "contract_sha256" in proc.stderr


def test_a_second_tokenizer_refuses(tmp_path):
    other = payload_meta(
        "student", "prefill", scored=N_CHUNKS * SEQLEN,
        tokenizer={**TOKENIZER, "path": "/models/other",
                   "files": {"tokenizer.json": "b" * 64}})
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True,
                                   student_meta=other))
    assert proc.returncode != 0, proc.stdout
    assert "tokenizer" in proc.stderr


def test_a_prefill_student_of_another_artifact_refuses(tmp_path):
    other = payload_meta("student", "prefill", scored=N_CHUNKS * SEQLEN,
                         artifact="/runs/armB")
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True,
                                   student_meta=other))
    assert proc.returncode != 0, proc.stdout
    assert "/runs/armB" in proc.stderr and "which forward ran" in proc.stderr


def test_a_short_student_prefill_refuses(tmp_path):
    """The teacher was bounds-checked and the student was not indexed-checked
    at all, so a short student payload reached ``s_ids[i]``."""
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True,
                                   student_rows=5))
    assert proc.returncode != 0, proc.stdout
    assert "--student-prefill holds 5 positions" in proc.stderr


def test_a_short_teacher_prefill_refuses(tmp_path):
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True,
                                   teacher_rows=5))
    assert proc.returncode != 0, proc.stdout
    assert "--teacher-prefill holds 5 positions" in proc.stderr


def test_a_prefix_length_with_no_prefill_counterpart_refuses(tmp_path):
    """A no-BOS corpus scores three rows per chunk of four, so a decode sweep
    that reached prefix length 4 has a position the prefill dump never held."""
    decode = payload_meta("student", "decode", scored=N_CHUNKS * (SEQLEN - 1),
                          prefixes=[[1, 4], [1, 3]])
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=False,
                                   decode_meta=decode))
    assert proc.returncode != 0, proc.stdout
    assert "outside the 1..3" in proc.stderr


def test_a_prefix_record_for_the_wrong_chunk_count_refuses(tmp_path):
    decode = payload_meta("student", "decode", scored=N_CHUNKS * SEQLEN,
                          prefixes=[[1, 3]])
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True,
                                   decode_meta=decode))
    assert proc.returncode != 0, proc.stdout
    assert "prefix lengths for 1 chunks" in proc.stderr


def test_a_corpus_record_without_a_contract_refuses(tmp_path):
    """A legacy free-length corpus records no contract, so it cannot be
    aligned -- only guessed at."""
    meta = payload_meta("student", "prefill", scored=N_CHUNKS * SEQLEN)
    meta["corpus"]["contract_sha256"] = None
    meta["corpus"]["scored_positions"] = None
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True,
                                   student_meta=meta))
    assert proc.returncode != 0, proc.stdout
    assert "does not state" in proc.stderr


def test_a_contract_whose_scored_count_is_neither_refuses(tmp_path):
    """Neither ``seqlen`` nor ``seqlen - 1`` rows per chunk: this script says
    it cannot derive the mapping rather than picking an interpretation."""
    scored = N_CHUNKS * (SEQLEN - 2)
    metas = {
        "teacher_meta": payload_meta("teacher", "prefill", scored=scored,
                                     artifact="/models/bf16"),
        "student_meta": payload_meta("student", "prefill", scored=scored),
        "decode_meta": payload_meta("student", "decode", scored=scored),
    }
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True, **metas))
    assert proc.returncode != 0, proc.stdout
    assert "is not derivable from this contract" in proc.stderr


def test_a_row_with_no_comparable_distribution_refuses(tmp_path):
    """A row whose teacher map is empty was silently skipped, and the mean was
    published over the rest -- a number for positions it did not measure.
    ``kl_tool dump`` drops unscored positions before it writes, so such a row
    is a malformed payload rather than a normal one."""
    proc, _ = run(tmp_path, *build(tmp_path, prepends_bos=True,
                                   teacher_blank={0}),
                  "--seqlen", str(SEQLEN))
    assert proc.returncode != 0, proc.stdout
    assert "carried a comparable teacher and student distribution" in proc.stderr
    assert "prefill_on_decode_positions" in proc.stderr
