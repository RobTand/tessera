"""Merge the two halves of a shard-split Tessera export into one checkpoint.

`export_glm53_tessera.py --shards LO-HI` writes a self-consistent checkpoint
for its own subset: its shard files, an index covering only them, and a
`tessera_config.json` whose accounting counts only its units.  Splitting was
safe because input shards share no state -- a disjoint subset writes exactly
the files one box would -- and the merge is the same fact read backwards: the
shard *files* need no rewriting, only the two summaries that describe them.

Nothing below is assumed, because a merge that silently drops a shard, or
binds a stranger's shard, produces a checkpoint that loads and is wrong.  The
contract is the one ``tessera.serving_parts.merge_serving_parts`` already
holds its parts to (tessera#300), proved in this order and before a byte is
copied or a config written:

  * **The encoding is one encoding.** ``check_configs``: same grid, same
    code, same geometry, same rotation, and the same **Hessian** where one
    shaped the bytes -- two halves encoded differently are two artifacts.
  * **Every part was cut from ``--source``.** Each part stamps
    ``source_part_identity`` of what it read (config and auxiliary hashes, the
    whole tensor inventory, sha256 per shard read); the merge takes the same
    identity of ``--source`` once and proves each entry against it.  A part
    with no stamp -- an older exporter, or the in-memory ``export_checkpoint``
    -- is refused by name, not merged on the strength of its filenames.
  * **The parts were cut to one plan, and each fulfilled its slice.** Every
    part stamps the whole plan; they must agree.  For each part, the tensors
    it owns (the source inventory restricted to its shards) must appear in
    its index and its actual shard headers exactly as the plan says -- blob
    present and raw tensor absent for a planned name, raw for the rest, each
    in the shard it came from -- and every planned blob is parsed
    (``tessera.container.parse``, the reader's own fail-closed check) and its
    manifest must name that tensor at that rung and geometry.
  * **No shard is claimed twice, and the union is the source's shard list.**
    Both read off the content-verified inventory rather than a filename set.

Accounting is summed, not recomputed: `quantized_bytes`/`quantized_params` are
the accountant's own totals from each half, and body bpp is re-derived from the
sum so it cannot drift from the bytes.  `check_configs` is split out so the
encoding guard can be tested field by field rather than only through a full
merge; `check_assembly` is the source/plan proof, split out for the same
reason.
"""
import argparse
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Which fields must be identical across parts is the EXPORTER's list, read
#: from ``tessera.export`` rather than restated here.  A restatement is what
#: this guard has been wrong about twice: eight of its first thirteen names
#: existed nowhere in the config the exporter wrote, so they compared ``None``
#: to ``None`` and passed vacuously (fixed in ``317c882``), and three of the
#: replacements -- ``source_model``, ``prismaquant_plan``, ``inherits`` -- were
#: one GLM driver's ``extra_config``, so the guard refused every pair of parts
#: a plain ``export_checkpoint_streaming`` produced and blamed the exporter for
#: it (tessera#137).  ``tessera.export`` checks its declaration against the
#: config it just built on every export, so a name here is a name that was
#: written.
#:
#: Two subtractions, each a decision rather than a list:
#: ``SHARED_WHEN_WRITTEN`` is the pair older exporters legitimately lack, and
#: the activation block has its own class below.  Everything else the exporter
#: writes is compared here except ``CONFIG_PER_PART_FIELDS``: ``accounting``
#: and ``rungs_q256`` are summed or unioned, and ``plan`` and ``source`` are
#: the assembly contract ``check_assembly`` proves rather than compares.


def shared_fields():
    """The exporter's encoding fields: absent from a part means the guard is broken.

    Imported lazily because the serving-part merge this script also dispatches
    needs no tensor runtime (``tessera.serving_parts`` says so in its own
    docstring) and ``tessera.export`` pulls in torch.
    """
    from tessera.export import CONFIG_ENCODING_FIELDS

    return tuple(f for f in CONFIG_ENCODING_FIELDS if f not in SHARED_WHEN_WRITTEN)


def driver_fields(configs):
    """Top-level keys no exporter writes: a driver's ``extra_config``.

    Found by subtracting the exporter's own fields rather than by naming one
    driver's vocabulary, so a second driver's extras are guarded on the day it
    is written.  ``export_glm53_tessera.py`` contributes ``source_model``,
    ``prismaquant_plan`` and ``inherits``; the source model in particular is a
    genuine encoding fact, and two halves that disagree about it are two
    artifacts.  They are compared when every part carries them and refused when
    the parts disagree about whether they exist, which means two drivers -- the
    same rule ``SHARED_WHEN_WRITTEN`` states, for the same reason.
    """
    from tessera.export import (CONFIG_ACTIVATION_FIELD, CONFIG_ENCODING_FIELDS,
                                CONFIG_PER_PART_FIELDS)

    exporters = ({f.split(".")[0] for f in CONFIG_ENCODING_FIELDS}
                 | set(CONFIG_PER_PART_FIELDS) | {CONFIG_ACTIVATION_FIELD})
    return tuple(sorted(set().union(*(set(c) for c in configs)) - exporters))


#: Fields that define the encoding when the exporter writes them, and that
#: earlier exporters did not write at all: compared like :func:`shared_fields`
#: when the first part carries them, refused when the parts disagree on whether
#: they exist, skipped (and said so) when no part has them.  ``wire.recipes`` is
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
#:
#: ``schema_minor`` and ``tp_agnostic`` (tessera#328) are here for exactly the
#: same reason and NOT in :func:`shared_fields`: a part written before
#: 2026-09-05 carries ``tp_size: 1`` instead, and requiring the new names of
#: every part would make every part already on disk unmergeable -- the failure
#: mode #137 fixed for the driver fields.  Under this rule a set of legacy
#: parts merges unchanged (their ``tp_size`` is compared as a driver field --
#: no exporter writes it any more -- and they agree on it), a set of fresh
#: parts is compared on the new names, and a legacy part mixed with a fresh one
#: is refused as two exporters, which it is.
SHARED_WHEN_WRITTEN = ("wire.recipes", "encoder_fixture_id",
                       "schema_minor", "tp_agnostic")

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
#: ``_MISSING`` and pass, which is how eight of the guard's first thirteen went
#: unenforced.  ``tests/test_merge_guard.py`` asserts each name resolves in a
#: config the exporter actually wrote, and gives each one its own failing case.
#:
#: The ``hessian.*`` fields are the capture's identity.  The three token
#: fields (``HESSIAN_IDENTITY`` in ``tessera.export``) name the fit prefix --
#: the sha of the calibration text, the fit token count and the sha of the
#: token ids -- but they CANNOT decide whether two captures are the same
#: capture: ``capture_h_full.py`` hashes the ids flat and then reshapes them
#: to ``[-1, seqlen]``, so one prefix captured at 512 and at 1024 agrees on
#: all three while running different attention contexts and accumulating
#: different H (tessera#214).  So the guard also compares the model and the
#: sequence layout by name, and ``capture_sha256`` -- the exporter's sealed
#: digest of the actual H content plus that context
#: (``ActivationSource.capture_sha256``) -- which separates two captures even
#: when every recorded token field coincides.  A part whose block lacks any
#: of these refuses below ("Re-export with a current exporter"), because a
#: field absent from every part compares equal and passes vacuously.
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
    "activation_aware.hessian.model",
    "activation_aware.hessian.seqlen",
    "activation_aware.hessian.capture_sha256",
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

    Four classes of field, and the difference between them is who wrote them:

      * :func:`shared_fields` -- what the exporter declares it writes.  Absent
        from the first part means the guard cannot do its job, so it refuses
        rather than passing; that is the bug that once left eight of thirteen
        names comparing ``None`` to ``None``.
      * ``SHARED_WHEN_WRITTEN`` -- written by later exporters only.  Compared
        when every part has them, refused when the parts disagree about
        whether they exist (different exporters), noted and skipped when none
        does.
      * :func:`driver_fields` -- what a driver added through ``extra_config``,
        found by subtracting the exporter's own fields.  Same presence rule as
        ``SHARED_WHEN_WRITTEN``: a part that has one and a part that has not
        were built by different drivers.
      * ``SHARED_ACTIVATION`` -- the activation-aware block, which is
        ``null`` on a weights-only export and a dict otherwise.  Null in every
        part is a consistent weights-only merge; a mix of null and dict is two
        artifacts, one of which was shaped by a Hessian the other never saw.
    """
    names = [name for name, _ in parts]
    base = dict(parts[0][1])
    shared = shared_fields()
    # A guard that cannot find the field it guards is a bug, not a pass.  The
    # old ``if field in base`` skipped absent names silently, which is how
    # eight of them went unenforced without anyone noticing.
    absent = [f for f in shared if dotted(base, f) is _MISSING]
    if absent:
        raise SystemExit(
            f"{names[0]} has no {absent} -- these fields define the "
            f"encoding and cannot be compared across parts, so the merge "
            f"cannot certify the parts were encoded identically. Either the "
            f"exporter stopped writing them or tessera.export declares them "
            f"wrongly; fix that rather than merging unchecked.")
    compared = list(shared)
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
                "schema_minor":
                    "which container schema minor these bytes were written at "
                    "is unrecorded",
                "tp_agnostic":
                    "whether a rank can cut a shard out of these bytes is "
                    "unrecorded, so a loader refuses them above one rank",
            }[field]
            print(f"note: no part carries {field!r} (written by later "
                  f"exporters); {unchecked}")

    # --- a driver's own fields --------------------------------------------
    for field in driver_fields([config for _, config in parts]):
        present = [field in config for _, config in parts]
        if all(present):
            compared.append(field)
        else:
            without = [n for n, seen in zip(names, present) if not seen]
            raise SystemExit(
                f"{without} carry no {field!r} while the others do -- no "
                f"exporter writes that field, so the parts were built by "
                f"different drivers; rebuild them with one")

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


def _unsealed(name, stamp, schema):
    """Why a part's ``source`` block cannot bind it to ``--source``, or ``None``."""
    if stamp is _MISSING:
        return (f"{name} is unsealed: its tessera_config.json carries no 'source' "
                f"block, so nothing binds it to the checkpoint --source names -- it "
                f"was written by an exporter before tessera#300, and shard filenames "
                f"are not an identity. Re-export it with a current exporter "
                f"(export_checkpoint_streaming stamps the source it read).")
    if stamp is None:
        return (f"{name} carries 'source': null -- the in-memory export_checkpoint "
                f"wrote tensors it was handed and read no checkpoint, so nothing binds "
                f"it to --source. A shard-split part comes from "
                f"export_checkpoint_streaming; re-export it from the source.")
    if not isinstance(stamp, dict) or stamp.get("schema") != schema:
        return (f"{name} carries a 'source' block this merge does not read "
                f"({stamp.get('schema') if isinstance(stamp, dict) else type(stamp).__name__!r}; "
                f"this merge proves {schema!r}). Re-export it with a current exporter.")
    return None


def _difference(a, b, differ):
    """The keys two tensor mappings disagree on, for a refusal that names them;
    ``differ`` says what a changed value means (a rung, a shard)."""
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    moved = sorted(t for t in set(a) & set(b) if a[t] != b[t])
    return (f"{len(only_a) + len(only_b) + len(moved)} tensor(s): only in the first "
            f"{only_a[:3]}, only in the second {only_b[:3]}, {differ} {moved[:3]}")


def check_assembly(parts, source, blob_suffix, arity):
    """Prove every part against ``--source`` and against the one plan.

    ``parts`` is ``[(path, index, config), ...]``; ``blob_suffix`` and
    ``arity`` (the grid's) are the exporter's, encoding fields
    ``check_configs`` has already found equal across the parts.  A plan rung
    is a PER-POSITION rate and a manifest's ``root_q256`` the per-code one,
    ``arity`` positions to a code (``encode_linear_planes`` says why), so the
    blob is held to ``plan[name] * arity``.
    Returns ``(plan, identity, owner)``: the plan every part stamped, the
    ``source_part_identity`` of ``--source`` over all its shards (what the
    merged config records), and ``{shard: part path}``.  Raises ``SystemExit``
    naming the part, shard or tensor at fault, and reads nothing it is not
    proving -- so the cost is one pass over the source (the pass
    ``merge_serving_parts`` pays) plus one over the planned blobs.

    Order matters for the words a refusal uses: a part from another checkpoint
    is refused as that before its blobs are opened, and two plans are refused
    as two plans before either is proved fulfilled.
    """
    from safetensors import safe_open

    from tessera.container import parse
    from tessera.errors import TesseraError
    from tessera.serving_parts import (SOURCE_PART_SCHEMA, source_part_identity,
                                       tensor_names)

    # --- every part was cut from --source ------------------------------------
    for part, _, config in parts:
        why = _unsealed(part.name, config.get("source", _MISSING), SOURCE_PART_SCHEMA)
        if why:
            raise SystemExit(why)
    identity = source_part_identity(source)
    owner = {}
    for part, index, config in parts:
        stamp = config["source"]
        for field in ("config_sha256", "auxiliary_sha256"):
            if stamp.get(field) != identity[field]:
                raise SystemExit(
                    f"{part.name} was cut from a different checkpoint than {source}: "
                    f"its stamped {field} is {stamp.get(field)!r}, the source's is "
                    f"{identity[field]!r}")
        if stamp.get("tensors") != identity["tensors"]:
            raise SystemExit(
                f"{part.name} was cut from a checkpoint with a different tensor "
                f"inventory than {source}: "
                f"{_difference(stamp.get('tensors') or {}, identity['tensors'], 'in another shard')}")
        files = stamp.get("files") or {}
        for shard, digest in sorted(files.items()):
            if shard not in identity["files"]:
                raise SystemExit(
                    f"{part.name} read {shard}, which {source} does not hold")
            if digest != identity["files"][shard]:
                raise SystemExit(
                    f"{part.name} was cut from a different {shard} than {source} holds: "
                    f"the part read sha256 {digest}, the source's is "
                    f"{identity['files'][shard]} -- same filename, other bytes, so this "
                    f"is not a part of the checkpoint being published")
        claimed = set(index["weight_map"].values())
        if claimed != set(files):
            raise SystemExit(
                f"{part.name}'s index names shards {sorted(claimed - set(files))} its "
                f"source stamp did not read, or omits {sorted(set(files) - claimed)} "
                f"it did; the part's own index and receipt disagree")
        for shard in files:
            if shard in owner:
                raise SystemExit(
                    f"shard {shard} is claimed by both {owner[shard].name} and "
                    f"{part.name}: the ranges overlap, and one part's file "
                    f"would overwrite the other's")
            owner[shard] = part
    expected = set(identity["files"])
    if set(owner) != expected:
        missing, extra = sorted(expected - set(owner)), sorted(set(owner) - expected)
        raise SystemExit(
            f"the parts cover {len(owner)} of the source's {len(expected)} "
            f"shards. missing {missing[:5]}{'...' if len(missing) > 5 else ''}"
            f"  unexpected {extra[:5]}")

    # --- one plan, stamped whole by every part --------------------------------
    plans = [(part.name, config.get("plan")) for part, _, config in parts]
    for name, plan in plans:
        if not isinstance(plan, dict):
            raise SystemExit(f"{name} carries no plan dict; re-export it")
    first_name, plan = plans[0]
    for name, other in plans[1:]:
        if other != plan:
            raise SystemExit(
                f"parts were cut to different plans: {first_name} and {name} disagree "
                f"on {_difference(plan, other, 'at different rungs')} -- one checkpoint has one plan and "
                f"every part stamps the whole of it, so these are not parts of one "
                f"export; re-export them under one plan")

    # --- each part fulfilled its slice of it, in its actual output ------------
    for part, index, config in parts:
        files = config["source"]["files"]
        owned = {t: s for t, s in identity["tensors"].items() if s in files}
        slice_ = {t: q for t, q in plan.items() if t in owned}
        wanted = {(t + blob_suffix if t in slice_ else t): s for t, s in owned.items()}
        local = index["weight_map"]
        if local != wanted:
            raw = sorted(t for t in slice_ if t in local)
            encoded = sorted(t for t in owned if t not in slice_ and t + blob_suffix in local)
            absent = sorted(set(wanted) - set(local))
            extra = sorted(set(local) - set(wanted))
            moved = sorted(k for k in set(wanted) & set(local) if wanted[k] != local[k])
            raise SystemExit(
                f"{part.name} did not implement the plan for the tensors it owns: "
                f"planned but written raw {raw[:3]}, unplanned but written as blobs "
                f"{encoded[:3]}, owned but absent {absent[:3]}, present but unowned "
                f"{extra[:3]}, in another shard than the source's {moved[:3]}")
        for shard in sorted(files):
            actual = tensor_names(part / shard)
            in_shard = {k for k, s in wanted.items() if s == shard}
            if actual != in_shard:
                raise SystemExit(
                    f"{part.name}/{shard}: the index says {sorted(in_shard)[:3]}... "
                    f"but the file's header holds {sorted(actual)[:3]}... "
                    f"(missing {sorted(in_shard - actual)[:3]}, unexpected "
                    f"{sorted(actual - in_shard)[:3]})")
        if set(config.get("rungs_q256", [])) != set(slice_.values()):
            raise SystemExit(
                f"{part.name} records rungs_q256 {sorted(config.get('rungs_q256', []))} "
                f"but its slice of the plan uses {sorted(set(slice_.values()))}")
        for tensor, q256 in sorted(slice_.items()):
            key = tensor + blob_suffix
            with safe_open(str(part / owned[tensor]), framework="pt") as handle:
                blob = handle.get_tensor(key).numpy().tobytes()
            try:
                manifest = parse(blob).manifest
            except TesseraError as exc:
                raise SystemExit(
                    f"{part.name}: {key} is not a Tessera artifact this reader accepts "
                    f"({type(exc).__name__}: {exc}); nothing under that name can be "
                    f"published as {tensor}'s encoding") from exc
            branch = manifest.branch
            if branch.unit_id != tensor:
                raise SystemExit(
                    f"{part.name}: the blob under {key} is the encoding of "
                    f"{branch.unit_id!r}, not of {tensor}")
            if branch.root_q256 != q256 * arity:
                raise SystemExit(
                    f"{part.name}: {tensor} was encoded at q256 "
                    f"{branch.root_q256 / arity:g} per position ({branch.root_q256} "
                    f"per code) but the plan says {q256}; the config would price "
                    f"bytes that were never cut")
            with safe_open(str(source / owned[tensor]), framework="pt") as handle:
                shape = tuple(handle.get_slice(tensor).get_shape())
            geometry = (manifest.geometry.rows, manifest.geometry.columns)
            if geometry != shape:
                raise SystemExit(
                    f"{part.name}: {tensor}'s blob encodes a {geometry} unit but the "
                    f"source tensor is {shape}")
    return plan, identity, owner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", help="the part directories, in any order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="/mnt/shared/models/GLM-5.3-Flash-BF16")
    ap.add_argument("--move", action="store_true",
                    help="move shard files instead of copying (needs one filesystem)")
    args = ap.parse_args()

    if any((Path(p) / "tessera_part_config.json").exists() for p in args.parts):
        from tessera.serving_parts import merge_serving_parts
        try:
            manifest = merge_serving_parts(args.parts, Path(args.out), Path(args.source), move=args.move)
        except (ValueError, OSError, KeyError) as exc:
            raise SystemExit(f"serving-part merge refused: {exc}") from exc
        print(f"merged {len(args.parts)} serving parts -> {args.out}: "
              f"{manifest['totals']['modules']} modules, {manifest['totals']['wire_bytes']} wire bytes")
        return

    loaded = [load(p) for p in args.parts]
    out = Path(args.out)
    source = Path(args.source)

    # --- config: identical where it must be ------------------------------
    base = check_configs([(part.name, config) for part, _, config in loaded])

    # --- source and plan: proved, before anything is written --------------
    plan, identity, seen = check_assembly(loaded, source, base["blob_suffix"],
                                          int(base["grid"]["arity"]))
    weight_map = {}
    for part, index, _ in loaded:
        for key, shard in index["weight_map"].items():
            if key in weight_map:
                raise SystemExit(f"tensor {key} appears in two parts")
            weight_map[key] = shard

    # --- summed where it adds ---------------------------------------------
    acct = {"quantized_params": 0, "quantized_bytes": 0, "passthrough_bytes": 0}
    rungs = set()
    for _, _, config in loaded:
        for key in acct:
            acct[key] += int(config["accounting"][key])
        rungs.update(config["rungs_q256"])
    bpp = Fraction(acct["quantized_bytes"] * 8, acct["quantized_params"])
    acct["body_bpp"] = float(bpp)
    acct["body_bpp_exact"] = [bpp.numerator, bpp.denominator]
    base["accounting"] = acct
    base["plan"] = plan
    base["rungs_q256"] = sorted(rungs)
    # The merged checkpoint is bound to the whole source the parts were proved
    # against: the same block a part carries, with every shard's hash.
    base["source"] = identity
    base["merged_from"] = [p.name for p, _, _ in loaded]

    out.mkdir(parents=True, exist_ok=True)
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
