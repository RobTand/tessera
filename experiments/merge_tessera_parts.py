"""Merge the two halves of a shard-split Tessera export into one checkpoint.

`export_glm53_tessera.py --shards LO-HI` writes a self-consistent checkpoint
for its own subset: its shard files, an index covering only them, and a
`tessera_config.json` whose accounting counts only its units.  Splitting was
safe because input shards share no state -- a disjoint subset writes exactly
the files one box would -- and the merge is the same fact read backwards: the
shard *files* need no rewriting, only the two summaries that describe them.

Three things are checked rather than assumed, because a merge that silently
drops a shard produces a checkpoint that loads and is wrong:

  * **No shard is claimed twice.** Overlapping ranges would let one part's file
    overwrite the other's.
  * **The union covers the source's shard list exactly.** A missing input shard
    means a range was mistyped or a part died early, and the resulting index
    would name tensors no file holds.
  * **Every weight_map entry resolves to a file that exists**, after the move.

Accounting is summed, not recomputed: `quantized_bytes`/`quantized_params` are
the accountant's own totals from each half, and body bpp is re-derived from the
sum so it cannot drift from the bytes.  The rest of the config must be
*identical* between halves -- same grid, same code, same geometry, same
rotation, and the same **Hessian** where one shaped the bytes -- and the merge
refuses if it is not, since two halves encoded differently are two artifacts,
not one.  `check_configs` is that comparison, split out so the guard can be
tested field by field rather than only through a full merge.
"""
import argparse
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

#: Fields that define the ENCODING and must therefore be identical across every
#: part.  Dotted paths, because the exporter nests them: eight of the thirteen
#: names this tuple used to carry (``grid_digest``, ``code``, ``group``,
#: ``half``, ``container``, ``superblock``, ``encoder_profile``, ``arity``)
#: exist nowhere in the config the exporter writes, so they compared
#: ``None == None`` and passed vacuously.  ``grid.digest`` and ``conv_memory``
#: were among them -- precisely the two that catch encoder drift, which is the
#: failure this merge exists to prevent.
#:
#: Excluded on purpose: ``accounting``, ``plan`` and ``rungs_q256`` are per-part
#: by construction and are summed or unioned, not compared.
SHARED = (
    "quant_method", "container_version", "blob_suffix",
    "grid.digest", "grid.name", "grid.base", "grid.partition",
    "grid.arity", "grid.size", "grid.rate_cap",
    "conv_memory", "conv_generators", "trellis.span", "trellis.weighting",
    "body.kind", "body.window_bits", "body.seed", "body.sigma",
    "scale.group", "scale.half", "scale.refit", "scale.schedule", "scale.plane",
    "rotation", "with_diagonals", "tp_size",
    "source_model", "prismaquant_plan", "route_status",
    "requires_serve_flags", "inherits",
)

#: Fields that define the encoding when the exporter writes them, and that
#: earlier exporters did not write at all: compared like ``SHARED`` when the
#: first part carries them, refused when the parts disagree on whether they
#: exist, skipped (and said so) when no part has them.  ``wire.recipes`` is
#: the per-rung recipe table (body, span, plane, window table parameters per
#: q256 range) -- the flat ``body``/``scale.plane``/``trellis.span`` keys are
#: its projection and read ``per-rung`` when it varies, so two parts that
#: agree on the flat keys can still have been encoded differently below the
#: cap, and only the table can say.  ``encoder_fixture_id`` is which *encoder*
#: cut the bytes (``tessera.encoder_identity``, tessera#101): an encoder change
#: moves bytes at unchanged arguments and an unchanged profile id, so parts
#: built either side of one are two artifacts, not one, and nothing else in
#: this config can tell them apart.  The guard compares the stamped strings and
#: never computes an identity of its own -- only a process that is about to
#: encode pays for that.
SHARED_WHEN_WRITTEN = ("wire.recipes", "encoder_fixture_id")

#: The flat projections of the recipe table.  They describe the rungs a
#: part *used*, so two parts of one checkpoint may legitimately differ on
#: them (one part all at the cap, the other mixed) while carrying the same
#: table; when every part carries ``wire.recipes`` the table is compared and
#: these are not.
PROJECTED_BY_TABLE = ("trellis.span", "body.kind", "body.window_bits", "body.seed",
                      "body.sigma", "scale.plane", "scale.sigma")

#: The activation-aware encoding: which Hessian shaped the bytes, and what was
#: done with it.  Compared field by field so a refusal names the one that
#: differs, and every one of them is written by ``_write_config`` -- a name
#: here that the exporter does not write would compare ``_MISSING`` to
#: ``_MISSING`` and pass, which is how eight of ``SHARED``'s thirteen went
#: unenforced.  ``tests/test_merge_guard.py`` asserts each name resolves in a
#: config the exporter actually wrote, and gives each one its own failing case.
#:
#: The three ``hessian.*`` fields are the capture's identity (``HESSIAN_IDENTITY``
#: in ``tessera.export``): the sha of the calibration text, the fit token count
#: and the sha of the token ids.  The rest of the provenance (model, seqlen,
#: source split) rides along for an auditor and is not compared, because these
#: three already decide whether two captures are the same capture.
SHARED_ACTIVATION = (
    "activation_aware.ldlq_sigma",
    "activation_aware.ldlq_block",
    "activation_aware.refit_objective",
    "activation_aware.refit_objective_trailing",
    "activation_aware.refit_reach_floor",
    "activation_aware.refit_gauss_seidel",
    "activation_aware.hessian.text_sha256",
    "activation_aware.hessian.fit_tokens",
    "activation_aware.hessian.fit_ids_sha256",
)

_MISSING = object()


def dotted(config, path):
    """``config`` walked by a dotted path, or ``_MISSING``."""
    node = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def load(part):
    part = Path(part)
    index = json.loads((part / "model.safetensors.index.json").read_text())
    config = json.loads((part / "tessera_config.json").read_text())
    return part, index, config


def check_configs(parts):
    """Refuse unless every part was encoded identically; return the first config.

    ``parts`` is ``[(name, config), ...]``.  Raises ``SystemExit`` naming the
    single field that differs, because "the configs disagree" is not an
    actionable message when thirty fields are compared.

    Three classes of field, and the difference between them is who wrote them:

      * ``SHARED`` -- written by every exporter that has ever run.  Absent from
        the first part means the guard cannot do its job, so it refuses rather
        than passing; that is the bug that once left eight of thirteen names
        comparing ``None`` to ``None``.
      * ``SHARED_WHEN_WRITTEN`` -- written by later exporters only.  Compared
        when every part has them, refused when the parts disagree about
        whether they exist (different exporters), noted and skipped when none
        does.
      * ``SHARED_ACTIVATION`` -- the activation-aware block, which is
        ``null`` on a weights-only export and a dict otherwise.  Null in every
        part is a consistent weights-only merge; a mix of null and dict is two
        artifacts, one of which was shaped by a Hessian the other never saw.
    """
    names = [name for name, _ in parts]
    base = dict(parts[0][1])
    # A guard that cannot find the field it guards is a bug, not a pass.  The
    # old ``if field in base`` skipped absent names silently, which is how
    # eight of them went unenforced without anyone noticing.
    absent = [f for f in SHARED if dotted(base, f) is _MISSING]
    if absent:
        raise SystemExit(
            f"{names[0]} has no {absent} -- these fields define the "
            f"encoding and cannot be compared across parts, so the merge "
            f"cannot certify the parts were encoded identically. Either the "
            f"exporter stopped writing them or SHARED names them wrongly; "
            f"fix that rather than merging unchecked.")
    compared = list(SHARED)
    for field in SHARED_WHEN_WRITTEN:
        present = [dotted(config, field) is not _MISSING for _, config in parts]
        if all(present):
            compared.append(field)
            if field == "wire.recipes":
                compared = [f for f in compared if f not in PROJECTED_BY_TABLE]
        elif any(present):
            raise SystemExit(
                f"{field!r} is written by some parts and not others -- they were "
                f"built by different exporters; rebuild the older parts")
        else:
            unchecked = {
                "wire.recipes":
                    "the recipe is compared through its flat projection only",
                "encoder_fixture_id":
                    "whether one encoder cut both parts is unrecorded",
            }[field]
            print(f"note: no part carries {field!r} (written by later "
                  f"exporters); {unchecked}")

    # --- the activation-aware block ---------------------------------------
    written = [dotted(config, "activation_aware") for _, config in parts]
    if any(v is not _MISSING for v in written):
        stale = [n for n, v in zip(names, written) if v is _MISSING]
        if stale:
            raise SystemExit(
                f"{stale} carry no 'activation_aware' key while the others do -- "
                f"they were built by different exporters, and the older ones "
                f"cannot say whether a Hessian shaped their bytes; rebuild them")
        aware = [n for n, v in zip(names, written) if v]
        if aware and len(aware) != len(names):
            plain = [n for n, v in zip(names, written) if not v]
            raise SystemExit(
                f"{aware} were encoded activation-aware and {plain} were not -- "
                f"one half's bytes depend on a Hessian the other half never saw, "
                f"so they are two artifacts, not one")
        if aware:
            missing = [f for f in SHARED_ACTIVATION if dotted(base, f) is _MISSING]
            if missing:
                raise SystemExit(
                    f"{names[0]} is activation-aware but its block has no "
                    f"{missing} -- the merge cannot certify the parts were built "
                    f"against the same Hessian at the same settings, and a field "
                    f"absent from every part would compare equal and pass "
                    f"vacuously. Re-export with a current exporter.")
            compared.extend(SHARED_ACTIVATION)
    else:
        print("note: no part carries 'activation_aware' (written by later "
              "exporters); whether a Hessian shaped these bytes is unrecorded")

    for name, config in parts[1:]:
        for field in compared:
            if dotted(base, field) != dotted(config, field):
                raise SystemExit(
                    f"parts disagree on {field!r}: {dotted(base, field)!r} vs "
                    f"{dotted(config, field)!r} -- two halves encoded differently "
                    f"are two artifacts, not one")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", help="the part directories, in any order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="/mnt/shared/models/GLM-5.3-Flash-BF16")
    ap.add_argument("--move", action="store_true",
                    help="move shard files instead of copying (needs one filesystem)")
    args = ap.parse_args()

    loaded = [load(p) for p in args.parts]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- the three checks ------------------------------------------------
    seen, weight_map = {}, {}
    for part, index, _ in loaded:
        for key, shard in index["weight_map"].items():
            if shard in seen and seen[shard] != part:
                raise SystemExit(
                    f"shard {shard} is claimed by both {seen[shard].name} and "
                    f"{part.name}: the ranges overlap, and one part's file "
                    f"would overwrite the other's")
            seen[shard] = part
            if key in weight_map:
                raise SystemExit(f"tensor {key} appears in two parts")
            weight_map[key] = shard

    src_index = Path(args.source) / "model.safetensors.index.json"
    expected = set(json.loads(src_index.read_text())["weight_map"].values())
    got = set(seen)
    if got != expected:
        missing, extra = sorted(expected - got), sorted(got - expected)
        raise SystemExit(
            f"the parts cover {len(got)} of the source's {len(expected)} "
            f"shards. missing {missing[:5]}{'...' if len(missing) > 5 else ''}"
            f"  unexpected {extra[:5]}")

    # --- config: identical where it must be, summed where it adds --------
    base = check_configs([(part.name, config) for part, _, config in loaded])
    acct = {"quantized_params": 0, "quantized_bytes": 0, "passthrough_bytes": 0}
    plan, rungs = {}, set()
    for _, _, config in loaded:
        for key in acct:
            acct[key] += int(config["accounting"][key])
        plan.update(config.get("plan", {}))
        rungs.update(config.get("rungs_q256", []))
    bpp = Fraction(acct["quantized_bytes"] * 8, acct["quantized_params"])
    acct["body_bpp"] = float(bpp)
    acct["body_bpp_exact"] = [bpp.numerator, bpp.denominator]
    base["accounting"] = acct
    base["plan"] = plan
    base["rungs_q256"] = sorted(rungs)
    base["merged_from"] = [p.name for p, _, _ in loaded]

    # --- move the files ---------------------------------------------------
    move = shutil.move if args.move else shutil.copy2
    for shard, part in sorted(seen.items()):
        target = out / shard
        if target.exists() and target.resolve() == (part / shard).resolve():
            continue
        move(str(part / shard), str(target))
    for shard in sorted(seen):
        if not (out / shard).exists():
            raise SystemExit(f"{shard} is named by the index but absent from {out}")

    total = sum((out / s).stat().st_size for s in sorted(seen))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    (out / "tessera_config.json").write_text(json.dumps(base, indent=2))
    for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
        for aux in Path(args.source).glob(pattern):
            if aux.name == "model.safetensors.index.json":
                continue
            if not (out / aux.name).exists():
                shutil.copy2(aux, out / aux.name)

    gib = lambda b: b / 2 ** 30
    print(f"merged {len(loaded)} parts -> {out}")
    print(f"  shards           {len(seen)}   tensors {len(weight_map):,}")
    print(f"  quantized params {acct['quantized_params']:,}")
    print(f"  quantized bytes  {gib(acct['quantized_bytes']):.3f} GiB "
          f"= {acct['body_bpp']:.4f} bpp")
    print(f"  passthrough      {gib(acct['passthrough_bytes']):.3f} GiB")
    print(f"  TOTAL on disk    {gib(total):.3f} GiB")


if __name__ == "__main__":
    sys.exit(main())
