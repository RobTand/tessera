#!/usr/bin/env python3
"""The prefill regime restricted to the decode regime's positions (issue #102).

A decode-regime KL differs from a prefill-regime KL for two reasons at once:
the scored forward is M=1 instead of 512 rows, and the position set is a
stride subsample rather than all 511 positions of each chunk. Quoting the two
regime numbers side by side confounds them.

This script removes the second reason. The decode dump records the prefix
length L of every position it scored; the distribution at prefix L is
conditioned on exactly the same tokens as prefill position L, so it has a
counterpart row in a prefill dump of the same corpus. Restricting the prefill
pair to those rows leaves ONE difference between the two numbers: which
forward produced them.

The KL itself comes from ``kl_estimator.lumped_kl`` -- the same estimator
``kl_tool compare`` uses, imported rather than re-implemented, because a
second copy of an estimator is a second answer.

**The row mapping is derived from the corpus contract, never assumed**
(tessera#249).  How many prefill rows a chunk contributes is a property of the
corpus, not of this script: without an injected BOS the first token of a chunk
has no predecessor and is not scored, so a chunk contributes ``seqlen - 1``
rows; with one, BOS conditions the first corpus token and it contributes
``seqlen``.  Every dump carries its corpus contract's own ``n_chunks``,
``seqlen`` and ``scored_positions``, so the stride is read
(``scored_positions / n_chunks``) rather than spelled ``seqlen - 1``.  It used
to be spelled: on a BOS corpus of two chunks with decode prefixes
``[[1, 3], [1, 3]]`` the matching prefill rows are ``[0, 2, 4, 6]`` and the
assumed formula returned ``[0, 2, 3, 5]`` -- the second chunk's positions
folded into the first, every index in bounds, and the "only remaining
difference is which forward ran" claim printed over the wrong positions.
``experiments/kl_reference_usable.py`` already derived the same distinction
from the same field.

For the same reason the three payloads must be shown to be one population
before any of them is indexed: same corpus contract digest, same tokenizer
bytes, same chunk geometry, the two prefill arrays sized as the contract says,
and the prefill student the same artifact as the decode student.  Two dumps of
the same shape over different corpora produce a confident number about
nothing.

usage::

    decode_regime_subset.py --teacher-prefill T.npz --student-prefill S.npz \
        --decode-student D.npz [--seqlen 512] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

#: The served-KL instrument is untracked and lives outside every checkout, one
#: copy per box.  ``KL_TOOL_DIR`` is the variable ``tests/box_artifacts.py``
#: declares for it; the literal is the documented default.
KL_TOOL_DIR = Path(os.environ.get("KL_TOOL_DIR") or "/home/rob/dq-runs")
sys.path.insert(0, str(KL_TOOL_DIR))
from kl_estimator import (  # noqa: E402
    DEFAULT_STUDENT_PROB_FLOOR, lumped_kl, tokenizer_identities_agree)
import kl_tool  # noqa: E402

#: What a payload's corpus record must state before its rows can be aligned.
#: Each is written by ``kl_tool dump`` out of the corpus contract itself.
REQUIRED_CORPUS_FIELDS = ("contract_sha256", "n_chunks", "seqlen",
                          "scored_positions")


def corpus_of(meta, label):
    """One payload's corpus contract record, refusing an incomplete one."""
    corpus = (meta or {}).get("corpus") or {}
    missing = [field for field in REQUIRED_CORPUS_FIELDS
               if corpus.get(field) is None]
    if missing:
        raise SystemExit(
            f"{label}: its corpus record does not state {missing}. The row "
            "mapping between the two regimes is derived from the corpus "
            "contract; a payload that does not carry one (a legacy "
            "free-length corpus, say) cannot be aligned, only guessed at")
    return corpus


def require_one_corpus(payloads):
    """Refuse three payloads that are not one corpus, one tokenizer, one shape.

    Same-sized arrays from two corpora index perfectly and mean nothing, which
    is the failure ``kl_reference_usable`` was written for one level up.  The
    identities are recorded in every dump; compare them rather than the counts.
    """
    (ref_label, ref_meta, ref_corpus) = payloads[0]
    for label, meta, corpus in payloads[1:]:
        for field in ("contract_sha256", "n_chunks", "seqlen",
                      "scored_positions"):
            if corpus[field] != ref_corpus[field]:
                raise SystemExit(
                    f"{label} and {ref_label} disagree on corpus {field}: "
                    f"{corpus[field]!r} against {ref_corpus[field]!r}. These "
                    "score different positions and mean different things")
        if not tokenizer_identities_agree(meta.get("tokenizer") or {},
                                          ref_meta.get("tokenizer") or {}):
            raise SystemExit(
                f"{label} was dumped with tokenizer "
                f"{(meta.get('tokenizer') or {}).get('path')!r} and "
                f"{ref_label} with "
                f"{(ref_meta.get('tokenizer') or {}).get('path')!r}; the file "
                "digests do not agree, so the two token-id populations are "
                "not the same corpus")
    return ref_corpus


def artifact_of(meta, label):
    path = ((meta or {}).get("model") or {}).get("artifact_path")
    if not path:
        raise SystemExit(
            f"{label}: the dump records no model.artifact_path, so the "
            "prefill student cannot be shown to be the decode student")
    return path


def prefill_layout(corpus, *, seqlen_arg=None):
    """The prefill row layout this corpus contract defines.

    ``scored_positions`` is what the contract says a prefill dump of it holds,
    so ``scored_positions / n_chunks`` is the number of rows one chunk
    contributes -- ``seqlen`` when the corpus prepends BOS and ``seqlen - 1``
    when it does not.  Anything else is a contract this script cannot align,
    and it says so rather than picking an interpretation.
    """
    n_chunks = int(corpus["n_chunks"])
    seqlen = int(corpus["seqlen"])
    scored = int(corpus["scored_positions"])
    if n_chunks < 1 or seqlen < 2 or scored < 1:
        raise SystemExit(
            f"corpus contract n_chunks={n_chunks} seqlen={seqlen} "
            f"scored_positions={scored} is not a corpus this can align")
    if seqlen_arg is not None and seqlen_arg != seqlen:
        raise SystemExit(
            f"--seqlen {seqlen_arg} contradicts the corpus contract's "
            f"seqlen {seqlen}; the mapping is the contract's, and a --seqlen "
            "that disagrees with it would silently move every row")
    if scored % n_chunks:
        raise SystemExit(
            f"corpus contract scores {scored} positions over {n_chunks} "
            "chunks, which is not a whole number of rows per chunk")
    rows_per_chunk = scored // n_chunks
    if rows_per_chunk == seqlen:
        prepends_bos = True
    elif rows_per_chunk == seqlen - 1:
        prepends_bos = False
    else:
        raise SystemExit(
            f"corpus contract scores {rows_per_chunk} positions per chunk of "
            f"{seqlen} tokens, which is neither {seqlen - 1} (no BOS: the "
            f"chunk's first token has no predecessor) nor {seqlen} (BOS "
            "conditions it). The row mapping is not derivable from this "
            "contract")
    return {"n_chunks": n_chunks, "seqlen": seqlen,
            "scored_positions": scored,
            "prefill_rows_per_chunk": rows_per_chunk,
            "prepends_bos": prepends_bos}


def prefill_rows(decode_meta, *, layout):
    """Prefill row indices for the positions a decode dump scored.

    A prefill dump scores chunk ``c``'s position ``p`` -- 1-based within the
    conditioned prompt, predicting the token at index ``p`` -- as row
    ``c * rows_per_chunk + (p - 1)``, where ``rows_per_chunk`` comes from the
    corpus contract (:func:`prefill_layout`) and not from ``seqlen - 1``. A
    decode position of prefix length ``L`` predicts the token at index ``L``,
    so ``p = L``.
    """
    rows_per_chunk = layout["prefill_rows_per_chunk"]
    regime = (decode_meta or {}).get("regime") or {}
    prefixes = regime.get("prefix_lengths")
    if not isinstance(prefixes, list) or len(prefixes) != layout["n_chunks"]:
        raise SystemExit(
            f"the decode payload records prefix lengths for "
            f"{len(prefixes) if isinstance(prefixes, list) else prefixes!r} "
            f"chunks, and the corpus contract has {layout['n_chunks']}")
    rows = []
    for c, prefixes_c in enumerate(prefixes):
        if not isinstance(prefixes_c, list) or not prefixes_c:
            raise SystemExit(
                f"the decode payload's chunk {c} records no scored prefix "
                f"lengths ({prefixes_c!r})")
        for L in prefixes_c:
            L = int(L)
            if not 1 <= L <= rows_per_chunk:
                raise SystemExit(
                    f"the decode payload scored chunk {c} at prefix length "
                    f"{L}, outside the 1..{rows_per_chunk} this corpus's "
                    "prefill dump holds -- the position has no prefill "
                    "counterpart to compare against")
            rows.append(c * rows_per_chunk + (L - 1))
    return np.array(rows, dtype=np.int64)


def kl_over(t_ids, t_lps, s_ids, s_lps, rows, floor, *, label="the subset"):
    """The lumped KL over ``rows``, refusing to average over fewer than all.

    A row whose teacher or student map is empty is skipped, and the count of
    what was actually compared was published beside -- but not against -- the
    count that was asked for.  ``kl_tool dump`` drops unscored positions before
    it writes, so an empty row is a malformed payload and not a normal one; an
    average over an unstated subset of the subset is the confound this whole
    script exists to remove.  Zero comparable rows used to raise
    ``ValueError: zero-size array`` out of ``per.max()`` rather than say so.
    """
    per = []
    agree = 0
    for i in rows:
        tmap = {int(t): float(v) for t, v in zip(t_ids[i], t_lps[i])
                if int(t) != kl_tool.PAD_ID and np.isfinite(v)}
        smap = {int(t): float(v) for t, v in zip(s_ids[i], s_lps[i])
                if int(t) != kl_tool.PAD_ID and np.isfinite(v)}
        if not tmap or not smap:
            continue
        shared = [t for t in tmap if t in smap]
        stats = lumped_kl(np.exp(np.array([tmap[t] for t in shared])),
                          np.exp(np.array([smap[t] for t in shared])),
                          student_prob_floor=floor, n_tail_tokens=None)
        per.append(stats["kl_lower"])
        if max(tmap, key=tmap.get) == max(smap, key=smap.get):
            agree += 1
    per = np.array(per, dtype=np.float64)
    if per.size != len(rows):
        raise SystemExit(
            f"{label}: {per.size} of {len(rows)} rows carried a comparable "
            "teacher and student distribution.  Averaging over the rest would "
            "publish a number for positions it did not measure")
    return {"positions": int(per.size), "kl_lower_mean": float(per.mean()),
            "kl_lower_max": float(per.max()),
            "top1_agree_pct": 100.0 * agree / per.size}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-prefill", required=True)
    ap.add_argument("--student-prefill", required=True)
    ap.add_argument("--decode-student", required=True)
    ap.add_argument("--seqlen", type=int, default=None,
                    help="a cross-check on the corpus contract's own seqlen, "
                         "not the source of the row mapping; a value that "
                         "disagrees with the contract is refused")
    ap.add_argument("--floor", type=float, default=DEFAULT_STUDENT_PROB_FLOOR)
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    d_meta, d_ids, _d_lps = kl_tool.read_payload(args.decode_student)
    if kl_tool.payload_regime(d_meta) != "decode":
        raise SystemExit("--decode-student is not a decode-regime payload")
    t_meta, t_ids, t_lps = kl_tool.read_payload(args.teacher_prefill)
    s_meta, s_ids, s_lps = kl_tool.read_payload(args.student_prefill)
    for name, meta in (("teacher", t_meta), ("student", s_meta)):
        if kl_tool.payload_regime(meta) != "prefill":
            raise SystemExit(f"--{name}-prefill is not a prefill-regime payload")

    # One corpus, one tokenizer, one geometry -- established from what the
    # three payloads record, before a single row is indexed (tessera#249).
    corpus = require_one_corpus([
        ("--decode-student", d_meta, corpus_of(d_meta, "--decode-student")),
        ("--teacher-prefill", t_meta, corpus_of(t_meta, "--teacher-prefill")),
        ("--student-prefill", s_meta, corpus_of(s_meta, "--student-prefill")),
    ])
    decode_artifact = artifact_of(d_meta, "--decode-student")
    student_artifact = artifact_of(s_meta, "--student-prefill")
    if decode_artifact != student_artifact:
        raise SystemExit(
            f"--student-prefill is a dump of {student_artifact!r} and "
            f"--decode-student of {decode_artifact!r}. The contrast this "
            "writes claims the only difference is which forward ran; two "
            "artifacts is a second difference")

    layout = prefill_layout(corpus, seqlen_arg=args.seqlen)
    expected = layout["n_chunks"] * layout["prefill_rows_per_chunk"]
    for name, ids in (("teacher", t_ids), ("student", s_ids)):
        if ids.shape[0] != expected:
            raise SystemExit(
                f"--{name}-prefill holds {ids.shape[0]} positions and its "
                f"corpus contract defines {expected} "
                f"({layout['n_chunks']} chunks x "
                f"{layout['prefill_rows_per_chunk']} rows, prepends_bos="
                f"{layout['prepends_bos']}); this is not a full prefill dump "
                "of that corpus")

    rows = prefill_rows(d_meta, layout=layout)
    if rows.size != d_ids.shape[0]:
        raise SystemExit(
            f"the decode payload records {rows.size} scored prefix lengths "
            f"and holds {d_ids.shape[0]} scored positions; its own regime "
            "record does not describe its own array")
    result = {
        "schema": "tessera.decode_regime_subset/2",
        "decode_payload": str(Path(args.decode_student).resolve()),
        "teacher_prefill": str(Path(args.teacher_prefill).resolve()),
        "student_prefill": str(Path(args.student_prefill).resolve()),
        "note": ("the prefill regime restricted to the decode regime's "
                 "positions: the only remaining difference between this and "
                 "the decode-regime compare is which forward ran"),
        # The alignment this run used, and where every number in it came
        # from: the corpus contract the three payloads agree on.
        "corpus_contract_sha256": corpus["contract_sha256"],
        "artifact_path": student_artifact,
        "alignment": layout,
        "prefill_rows": [int(r) for r in rows],
        "positions_requested": int(rows.size),
        "all_prefill_positions": int(t_ids.shape[0]),
        "prefill_on_decode_positions": kl_over(
            t_ids, t_lps, s_ids, s_lps, rows, args.floor,
            label="prefill_on_decode_positions"),
        "prefill_on_all_positions": kl_over(
            t_ids, t_lps, s_ids, s_lps, np.arange(t_ids.shape[0]), args.floor,
            label="prefill_on_all_positions"),
    }
    print(json.dumps(result, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
