#!/usr/bin/env python3
"""One-receipt full-vocabulary KL dumps for tessera#1172.

Loads the staged BF16 teacher, dumps prefill and decode full-vocabulary
logits over the frozen corpus, builds the FP8-RTN student in memory
(per-tensor E4M3 round-trip, deterministic, labelled a baseline screen and
never a promoted rate), saves it beside the teacher, and dumps it in both
regimes. All four payloads share one input set, one mask and one vocabulary;
the driver asserts that before it exits.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]


def _load_dump():
    spec = importlib.util.spec_from_file_location(
        "_fv_dump", CHECKOUT / "experiments" / "full_vocab_kl_dump.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _commit(phase, units):
    helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
    if not helper:
        return
    try:
        subprocess.run([sys.executable, helper, "--phase", phase,
                        "--units", str(units)],
                       check=False, capture_output=True, timeout=60)
    except Exception:
        pass


def build_fp8rtn_student(model, torch):
    """Per-tensor E4M3 round-trip of every >1D weight, on a deep copy."""
    student = copy.deepcopy(model).eval()
    changed, total = 0, 0
    with torch.no_grad():
        for p in student.parameters():
            if p.dim() < 2:
                continue
            w = p.data.float()
            amax = float(w.abs().max())
            if amax == 0.0:
                continue
            scale = amax / 448.0
            q = (w / scale).to(torch.float8_e4m3fn).float() * scale
            changed += int((q != w).sum())
            total += w.numel()
            p.data.copy_(q.to(p.dtype))
    return student, changed, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--corpus-contract", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--teacher-label", default="BF16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--slice-chunks", type=int, default=0)
    ap.add_argument("--slice-tokens", type=int, default=0)
    args = ap.parse_args()
    fv = _load_dump()
    import torch
    corpus = json.loads(Path(args.corpus_contract).read_text())
    chunks = corpus["chunks"]
    if args.slice_chunks:
        chunks = chunks[:args.slice_chunks]
    if args.slice_tokens:
        chunks = [c[:args.slice_tokens] for c in chunks]
    corpus = dict(corpus, chunks=chunks, n_chunks=len(chunks),
                  seqlen=len(chunks[0]),
                  scored_positions=len(chunks) * (len(chunks[0]) - 1))
    top_id = max(max(c) for c in chunks)
    print(f"chunks={len(chunks)} seqlen={len(chunks[0])} "
          f"max_id={top_id}", flush=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.slice_chunks or args.slice_tokens:
        # A sliced input set is a new contract, never the frozen one.
        corpus = dict(corpus, contract_sha256="slice-test-only")
        corp_path = outdir / "corpus_slice.json"
        corp_path.write_text(json.dumps(corpus))
    else:
        corp_path = Path(args.corpus_contract)
    _commit("load", 0)
    t0 = time.time()
    teacher = fv._load_model(args.teacher, "bfloat16")
    teacher = teacher.to(args.device).eval()
    print(f"teacher load_s={time.time() - t0:.1f} "
          f"arch={type(teacher).__name__}", flush=True)
    vocab = int(getattr(getattr(teacher.config, "text_config",
                                teacher.config), "vocab_size"))
    assert top_id < min(corpus["tokenizer"]["vocab_size"], vocab), (
        f"corpus id {top_id} outside the compared vocabulary")
    _commit("load", 1)
    stems = {}
    for role, model in (("teacher", teacher),):
        for regime in ("prefill", "decode"):
            stem = outdir / f"{role}_{regime}"
            rc = fv.dump_payload(_ns(args, role, regime, str(stem),
                                     str(corp_path), args.teacher), model)
            assert rc == 0
            stems[(role, regime)] = str(stem) + ".meta.json"
    _commit("dump", 1)
    t0 = time.time()
    student, changed, total = build_fp8rtn_student(teacher, torch)
    print(f"fp8rtn build_s={time.time() - t0:.1f} "
          f"changed={changed}/{total}", flush=True)
    sdir = outdir / "student_fp8rtn"
    student.save_pretrained(str(sdir))
    (sdir / "fp8rtn_provenance.json").write_text(json.dumps({
        "method": "per-tensor E4M3 round-trip of every >1D weight, "
                  "deterministic, no training",
        "changed_elements": changed, "total_elements": total,
        "teacher": str(Path(args.teacher).resolve()),
        "label": "baseline screen, not a promoted rate"}, indent=1))
    del teacher
    for regime in ("prefill", "decode"):
        stem = outdir / f"student_{regime}"
        rc = fv.dump_payload(_ns(args, "student", regime, str(stem),
                                 str(corp_path), str(sdir)), student)
        assert rc == 0
        stems[("student", regime)] = str(stem) + ".meta.json"
    _commit("dump", 2)
    metas = {k: json.loads(Path(v).read_text()) for k, v in stems.items()}
    base = metas[("teacher", "prefill")]
    for k, m in metas.items():
        assert fv.histories_and_masks(base, m), \
            f"{k} drifts from the input set"
        assert (m.get("regime") or {})["name"] == k[1], k
    print("all four payloads share one input set, mask and vocabulary",
          flush=True)
    print(json.dumps({f"{k[0]}/{k[1]}": m["payload"]
                      for k, m in metas.items()}, indent=1), flush=True)
    return 0


def _ns(args, role, regime, stem, corp_path, checkpoint):
    return argparse.Namespace(
        checkpoint=checkpoint, corpus_contract=corp_path, role=role,
        teacher_label=args.teacher_label if role == "teacher" else None,
        out=stem, regime=regime, dtype="bfloat16", device=args.device)


if __name__ == "__main__":
    raise SystemExit(main())
