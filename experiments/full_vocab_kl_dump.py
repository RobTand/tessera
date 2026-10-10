#!/usr/bin/env python3
"""Full-vocabulary served-KL logits dump and exact compare (tessera#1172).

``kl_tool dump`` tops out at the serve's ``--max-logprobs``: a 128k-vocabulary
dump over the completions surface is 523 million JSON entries that vLLM
materialises as Python dictionaries, which is not a serve anyone runs
(``docs/measurements/moe-evidence-debt-2026-09-04.md`` section 4). This
instrument takes the logits inside the runtime instead: it forwards the frozen
corpus through the checkpoint on the GPU and writes the exact file
``kl_tool.read_full_vocab_payload`` reads -- ``<stem>.logprobs.f32.npy``,
shape ``(positions, vocab_size)`` -- beside a ``<stem>.meta.json`` in
``prismaquant.kl_position_dump/2``.

Both regimes score the same histories through the same mask: every non-first
position of every corpus chunk (4088 on the canonical n=8 x 512 contract).
Prefill scores them one 512-row forward per chunk; decode scores each of them
off its own M=1 forward with a KV cache, teacher-forced on the corpus's own
next token so the histories never drift. ``compare`` refuses a cross-regime
pair the way ``kl_tool compare`` does, and refuses payloads whose corpus
contract, tokenizer identity, position count or vocabulary differ: an exact
number over mismatched inputs is a plausible wrong number with no error.

usage: full_vocab_kl_dump.py dump --checkpoint DIR --corpus-contract FILE
                                  --role teacher|student --out STEM
                                  --regime prefill|decode
                                  [--teacher-label LABEL] [--device cuda]
       full_vocab_kl_dump.py compare --teacher T.meta.json
                                     --student S.meta.json --out kl.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

DUMP_SCHEMA = "prismaquant.kl_position_dump/2"
COMPARE_SCHEMA = "tessera.full_vocab_kl_compare/1"


def _commit(phase: str, units: int) -> None:
    """A durable progress commit when a PrismaBuild helper is present."""
    helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
    if not helper:
        return
    try:
        subprocess.run([sys.executable, helper, "--phase", phase,
                        "--units", str(units)],
                       check=False, capture_output=True, timeout=60)
    except Exception:
        pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_full_vocab_payload(stem: str | Path, meta: dict,
                             logprobs: np.ndarray) -> None:
    """The file ``kl_tool.read_full_vocab_payload`` opens, by construction."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(logprobs, dtype=np.float32)
    positions, vocab = arr.shape
    arr_path = stem.parent / (stem.name + ".logprobs.f32.npy")
    out = np.lib.format.open_memmap(str(arr_path), mode="w+",
                                    dtype=np.float32, shape=(positions, vocab))
    out[:] = arr[:]
    out.flush()
    meta = dict(meta)
    meta["schema"] = DUMP_SCHEMA
    meta["payload"] = {"array": arr_path.name, "positions": int(positions),
                       "vocab_size": int(vocab), "dtype": "float32"}
    meta_path = stem.parent / (stem.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")


def read_full_vocab_payload(meta_path: str | Path) -> tuple[dict, np.ndarray]:
    """``<stem>.meta.json`` plus its memmap, shape-checked like kl_tool's."""
    path = Path(meta_path)
    meta = json.loads(path.read_text())
    arr_path = Path(meta["payload"]["array"])
    if not arr_path.is_absolute():
        arr_path = path.parent / arr_path
    arr = np.load(arr_path, mmap_mode="r")
    exp = (meta["payload"]["positions"], meta["payload"]["vocab_size"])
    if tuple(arr.shape) != exp:
        raise SystemExit(f"{arr_path} has shape {arr.shape}, meta says {exp}")
    return meta, arr


def histories_and_masks(meta_a: dict, meta_b: dict) -> bool:
    """Whether two payloads score the same inputs under the same mask."""
    ca, cb = meta_a.get("corpus") or {}, meta_b.get("corpus") or {}
    ta, tb = meta_a.get("tokenizer") or {}, meta_b.get("tokenizer") or {}
    return (ca.get("contract_sha256") == cb.get("contract_sha256")
            and ca.get("source_sha256") == cb.get("source_sha256")
            and (ta.get("identity_sha256") == tb.get("identity_sha256"))
            and meta_a.get("payload", {}).get("positions")
            == meta_b.get("payload", {}).get("positions")
            and meta_a.get("payload", {}).get("vocab_size")
            == meta_b.get("payload", {}).get("vocab_size"))


def _load_model(checkpoint: str, dtype: str):
    import torch
    from transformers import (AutoModelForCausalLM,
                              AutoModelForImageTextToText)
    kw = {"trust_remote_code": True}
    if dtype == "bfloat16":
        kw["dtype"] = torch.bfloat16
    try:
        return AutoModelForCausalLM.from_pretrained(checkpoint, **kw)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        return AutoModelForImageTextToText.from_pretrained(checkpoint, **kw)


def cmd_dump(args: argparse.Namespace) -> int:
    return dump_payload(args)


def dump_payload(args: argparse.Namespace, model=None) -> int:
    import torch
    corpus = json.loads(Path(args.corpus_contract).read_text())
    chunks = corpus["chunks"]
    seqlen = corpus["seqlen"]
    scored_per_chunk = seqlen - 1
    positions = len(chunks) * scored_per_chunk
    if model is None:
        _commit("load", 0)
        model = _load_model(args.checkpoint, args.dtype)
        model = model.to(args.device).eval()
        _commit("load", 1)
    arch = type(model).__name__
    vocab = int(model.config.get_text_config().vocab_size
                if hasattr(model.config, "get_text_config")
                else getattr(getattr(model.config, "text_config",
                                     model.config), "vocab_size"))
    stem = Path(args.out)
    arr_path = stem.parent / (stem.name + ".logprobs.f32.npy")
    stem.parent.mkdir(parents=True, exist_ok=True)
    writer = np.lib.format.open_memmap(
        str(arr_path), mode="w+", dtype=np.float32,
        shape=(positions, vocab))
    forwards = 0
    peak = 0
    with torch.inference_mode():
        row = 0
        for ci, chunk in enumerate(chunks):
            ids = torch.tensor([chunk], dtype=torch.long, device=args.device)
            if args.regime == "prefill":
                out = model(input_ids=ids)
                forwards += 1
                lps = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
                writer[row:row + scored_per_chunk] = lps.cpu().numpy()
                row += scored_per_chunk
            else:
                past = None
                nxt = ids[:, 0:1]
                for i in range(1, seqlen):
                    assert nxt.numel() == 1, (
                        f"chunk {ci} position {i}: decode forwarded "
                        f"{nxt.numel()} rows, not 1")
                    out = model(input_ids=nxt, past_key_values=past,
                                use_cache=True)
                    forwards += 1
                    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                    writer[row] = lp.cpu().numpy()
                    row += 1
                    past = out.past_key_values
                    # teacher-forced: the history stays the corpus's own.
                    nxt = ids[:, i:i + 1] if i + 1 < seqlen else nxt
            if torch.cuda.is_available():
                peak = max(peak, torch.cuda.max_memory_allocated())
            _commit("dump", ci + 1)
    writer.flush()
    meta = {
        "schema": DUMP_SCHEMA,
        "role": args.role,
        "metric": {"full_vocab": True, "requested_top_k": None},
        "corpus": {"source_sha256": corpus["source_sha256"],
                   "tokens": corpus["tokens"],
                   "contract_sha256": corpus["contract_sha256"]},
        "tokenizer": corpus["tokenizer"],
        "regime": {"name": args.regime,
                   "histories": len(chunks),
                   "mask": f"non-first positions 1..{seqlen - 1} of every "
                           f"chunk ({scored_per_chunk} per chunk)",
                   "rows_per_scored_forward":
                       (seqlen if args.regime == "prefill" else 1),
                   "forwards": forwards},
        "model": {"checkpoint": str(Path(args.checkpoint).resolve()),
                  "arch": arch, "dtype": args.dtype},
        "execution": {"device": args.device,
                      "cuda_device_name":
                          torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else None,
                      "peak_bytes": int(peak)},
        "payload": {"array": arr_path.name, "positions": positions,
                    "vocab_size": vocab, "dtype": "float32"},
    }
    if args.role == "teacher":
        if not args.teacher_label:
            raise SystemExit("a teacher dump needs --teacher-label: the "
                             "metric name is 'KL-vs-<label>'")
        meta["teacher_label"] = args.teacher_label
    meta_path = stem.parent / (stem.name + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"regime={args.regime} positions={positions} vocab={vocab} "
          f"forwards={forwards} -> {arr_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    teacher_meta, t_arr = read_full_vocab_payload(args.teacher)
    student_meta, s_arr = read_full_vocab_payload(args.student)
    t_reg = (teacher_meta.get("regime") or {}).get("name", "prefill")
    s_reg = (student_meta.get("regime") or {}).get("name", "prefill")
    if t_reg != s_reg:
        raise SystemExit(
            "REFUSED: cross-regime compare.\n"
            f"    teacher {Path(args.teacher).name} regime={t_reg}\n"
            f"    student {Path(args.student).name} regime={s_reg}\n"
            "  Re-dump the reference in the student's regime.")
    if not histories_and_masks(teacher_meta, student_meta):
        raise SystemExit(
            "REFUSED: histories, masks and vocab differ.\n"
            f"    teacher contract "
            f"{(teacher_meta.get('corpus') or {}).get('contract_sha256')} "
            f"vocab {(teacher_meta.get('payload') or {}).get('vocab_size')}\n"
            f"    student contract "
            f"{(student_meta.get('corpus') or {}).get('contract_sha256')} "
            f"vocab {(student_meta.get('payload') or {}).get('vocab_size')}\n"
            "  An exact number over mismatched inputs is a plausible wrong "
            "number with no error.")
    n, vocab = t_arr.shape
    # Row blocks: a full float64 copy of both sides is 10 GB at this
    # vocabulary. The dump writes true log-softmax rows, so max-subtracted
    # normalisation changes nothing but the rounding.
    cov_k = min(1024, vocab)
    kl_all = np.empty(n, dtype=np.float64)
    agree_all = np.empty(n, dtype=bool)
    tail_all = np.empty(n, dtype=np.float64)
    top1p_all = np.empty(n, dtype=np.float64)
    step = 256
    for lo in range(0, n, step):
        tb = np.asarray(t_arr[lo:lo + step], dtype=np.float64)
        sb = np.asarray(s_arr[lo:lo + step], dtype=np.float64)
        tb -= tb.max(axis=1, keepdims=True)
        sb -= sb.max(axis=1, keepdims=True)
        pb = np.exp(tb)
        pb /= pb.sum(axis=1, keepdims=True)
        qb = np.exp(sb)
        qb /= qb.sum(axis=1, keepdims=True)
        m = tb.shape[0]
        kl_all[lo:lo + m] = (pb * (np.log(pb) - np.log(qb))).sum(axis=1)
        tt = tb.argmax(axis=1)
        agree_all[lo:lo + m] = tt == sb.argmax(axis=1)
        top1p_all[lo:lo + m] = pb[np.arange(m), tt]
        # bare areas for a top-K instrument: teacher mass its own top-K
        # cannot see. K clamps to the vocabulary.
        order = np.argpartition(-tb, cov_k - 1, axis=1)[:, :cov_k]
        tail_all[lo:lo + m] = (
            1.0 - np.take_along_axis(pb, order, axis=1).sum(axis=1))
    kl, agree = kl_all, agree_all
    conf = top1p_all > 0.5
    tail = tail_all
    label = teacher_meta.get("teacher_label") or "unknown"
    corpus = teacher_meta.get("corpus") or {}
    tok = teacher_meta.get("tokenizer") or {}
    ident = {"metric": f"KL-vs-{label}", "bound": "exact (full vocabulary)",
             "partition": "full-vocabulary", "top_k": None,
             "positions": int(n), "vocab_size": int(vocab),
             "corpus_sha256": corpus.get("source_sha256", "unknown"),
             "contract_sha256": corpus.get("contract_sha256", "unknown"),
             "tokenizer_sha256": tok.get("identity_sha256", "unknown"),
             "regime": s_reg}
    result = {
        "schema": COMPARE_SCHEMA,
        "metric_identity": ident,
        "regime": s_reg,
        "teacher_payload": str(Path(args.teacher).resolve()),
        "student_payload": str(Path(args.student).resolve()),
        "teacher_sha256": sha256_file(
            Path(args.teacher).parent /
            teacher_meta["payload"]["array"]),
        "student_sha256": sha256_file(
            Path(args.student).parent /
            student_meta["payload"]["array"]),
        "positions": int(n),
        "all": {"kl_mean": float(kl.mean()),
                "kl_p99": float(np.quantile(kl, 0.99)),
                "kl_max": float(kl.max()),
                "top1_agree_pct": float(100.0 * agree.mean())},
        "confident": {"n": int(conf.sum()),
                      "fraction": float(conf.mean()),
                      "kl_mean": (float(kl[conf].mean())
                                  if conf.any() else None)},
        "topk_coverage": {"k": int(cov_k),
                          "teacher_tail_mass_mean": float(tail.mean()),
                          "teacher_tail_mass_max": float(tail.max()),
                          "nonconfident_positions": int(n - conf.sum())},
        "produced_at_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "argv": sys.argv,
    }
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(result, indent=1) + "\n")
    a, c = result["all"], result["confident"]
    print(f"KL-vs-{label} regime={s_reg} positions={n} vocab={vocab}")
    print(f"  ALL  exact KL {a['kl_mean']:.6f} "
          f"(p99 {a['kl_p99']:.6f} max {a['kl_max']:.6f}) "
          f"top1_agree={a['top1_agree_pct']:.2f}%")
    if c["n"]:
        print(f"  CONFIDENT n={c['n']} KL {c['kl_mean']:.6f}")
    print(f"  top-1024 tail mass mean "
          f"{result['topk_coverage']['teacher_tail_mass_mean']:.6f} max "
          f"{result['topk_coverage']['teacher_tail_mass_max']:.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--corpus-contract", required=True)
    d.add_argument("--role", required=True, choices=["teacher", "student"])
    d.add_argument("--teacher-label", default=None)
    d.add_argument("--out", required=True)
    d.add_argument("--regime", required=True,
                   choices=["prefill", "decode"])
    d.add_argument("--dtype", default="bfloat16")
    d.add_argument("--device", default="cuda")
    c = sub.add_parser("compare")
    c.add_argument("--teacher", required=True)
    c.add_argument("--student", required=True)
    c.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "dump":
        return cmd_dump(args)
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
