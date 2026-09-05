"""The legacy shard-split merge binds its parts to ``--source`` and to one plan.

``experiments/merge_tessera_parts.py`` assembles the parts a fleet of
``export_checkpoint_streaming`` runs wrote, one input shard per output shard.
Before tessera#300 it compared the SET of shard filenames against the source's
index, unioned the parts' ``plan`` dicts and copied the files: a part cut from
a different checkpoint with the same filenames merged, and a plan entry no
part had implemented was published over a raw bf16 tensor.  Both of codex's
static reproductions are below, plus the contract that replaces them -- the
one ``tessera.serving_parts.merge_serving_parts`` already has:

* every part carries the content identity of the source it read
  (``tessera_config.json`` ``source``: config, auxiliary files, whole tensor
  inventory, sha256 of every shard it read) and the merge proves each against
  the ``--source`` it publishes for;
* every part stamps the whole checkpoint's plan, the parts must agree on it,
  and each part's owned slice is proved fulfilled in its own output -- the
  blob present and the raw tensor absent in the header, and the blob's own
  manifest at the planned rung for that tensor -- before a byte is copied;
* a part with no binding (an older exporter, or the in-memory
  ``export_checkpoint``) is refused by name rather than merged unchecked.

tessera#337 is the layer under that.  Every check above is about what went
IN, or is name-, header- and manifest-shaped; none of them binds the bytes on
disk to the export that sealed them.  ``export_checkpoint_streaming`` used to
reuse a completed output directory, writing each shard over its old file and
replacing the index and config only after the last one, so a retry that failed
part way left NEW shards beside the OLD index, config and source seal -- and
that mixture merged clean and republished under the original checkpoint's
identity, decoding to ``[9.0, 1.03125]`` where its own published config prices
``[1.03125, 1.03125]``.  Two things close it, and both are below: a completed
output is immutable, so this exporter cannot make such a directory; and every
part stamps ``output`` (one sha256 per shard, taken as it wrote them), which
the merge verifies before assembly, so a directory mixed any other way is
refused by name.

Everything here runs on CPU with real encodes: the merge parses the blobs it
publishes, so a placeholder blob would be refused for the wrong reason.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.errors import GrammarError
from tessera.export import (BLOB_SUFFIX, export_checkpoint,
                            export_checkpoint_streaming, load_tessera_weight)
from tessera.serving_parts import (OUTPUT_PART_SCHEMA, SOURCE_PART_SCHEMA,
                                   output_part_identity, sha256_file,
                                   source_part_identity)

_spec = importlib.util.spec_from_file_location(
    "merge_tessera_parts",
    Path(__file__).resolve().parents[1] / "experiments" / "merge_tessera_parts.py",
)
merge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge)

K2 = tuple_grid(E2M1_GRID, 2)
Q256 = 896
FIRST = "model.layers.0.mlp.down_proj.weight"
SECOND = "model.layers.1.mlp.down_proj.weight"
NORM = "model.layers.1.norm.weight"


def _w(seed, rows=32, cols=256):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g).bfloat16()


def _write_source(root, tensors, hidden_size=32):
    """One shard per tensor, ``part-N.safetensors``, plus config and index --
    the layout codex's reproductions used."""
    root.mkdir(parents=True)
    weight_map = {}
    for i, (name, tensor) in enumerate(tensors.items(), start=1):
        shard = f"part-{i}.safetensors"
        save_file({name: tensor.contiguous()}, str(root / shard), metadata={"format": "pt"})
        weight_map[name] = shard
    (root / "config.json").write_text(json.dumps(
        {"architectures": ["Example"], "hidden_size": hidden_size}))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 0}, "weight_map": weight_map}))
    return weight_map


def _export(source, out, plan, shard):
    return export_checkpoint_streaming(
        source, out, plan, grid=K2, device="cpu", copy_aux=False,
        shard_filter={shard})


def _merge(parts, source, out):
    argv = ["merge", *map(str, parts), "--source", str(source), "--out", str(out)]
    with patch.object(sys, "argv", argv):
        merge.main()


def _config(part):
    return json.loads((Path(part) / "tessera_config.json").read_text())


def _rewrite_config(part, edit):
    config = _config(part)
    edit(config)
    (Path(part) / "tessera_config.json").write_text(json.dumps(config, indent=2))


def _reseal(part):
    """Re-stamp the ``output`` block over the shards the part now holds.

    Several cases below build a part whose OUTPUT does not implement the plan
    -- a raw tensor where a blob was planned, a blob under another tensor's
    name.  Those are obligations about honestly exported bytes, so the fixture
    seals what it holds; otherwise the merge would refuse each of them first
    as bytes replaced after the export (tessera#337), which is true of the
    fixture and not the rule under test.  Order is the point: the merge proves
    the bytes are the sealed bytes before it interprets them.
    """
    part = Path(part)
    index = json.loads((part / "model.safetensors.index.json").read_text())
    _rewrite_config(part, lambda c: c.__setitem__(
        "output", output_part_identity(part, set(index["weight_map"].values()))))


def _nothing_published(out):
    """A refusal happens before publication: no config, no index, no shard."""
    if not out.exists():
        return True
    return not any(out.iterdir())


def _two_shard_source(root, second=None):
    tensors = {FIRST: _w(1), SECOND: _w(2) if second is None else second}
    return _write_source(root, tensors), tensors


def _three_shard_source(root, seeds=(1, 2), norm=1.0):
    """Two planned shards and one the plan never names, so an assertion over
    the whole directory covers an encoded payload and a passthrough one."""
    tensors = {FIRST: _w(seeds[0]), SECOND: _w(seeds[1]),
               NORM: torch.full((32,), norm, dtype=torch.bfloat16)}
    return _write_source(root, tensors), tensors


SHARDS = ("part-1.safetensors", "part-2.safetensors", "part-3.safetensors")
PLANNED = {FIRST: Q256, SECOND: Q256}


def _export_all(source, out, plan=None):
    """An unfiltered part: every shard of the source, the way a one-box run
    and the failed-retry reproduction both go."""
    return export_checkpoint_streaming(
        source, out, PLANNED if plan is None else plan, grid=K2, device="cpu",
        copy_aux=False)


def _bytes_on_disk(path):
    return {p.name: sha256_file(p) for p in sorted(Path(path).iterdir()) if p.is_file()}


# --- codex's two static reproductions (tessera#300) -------------------------

def test_a_part_cut_from_another_checkpoint_refuses_before_publication(tmp_path):
    """Case 1: sources A and B share every filename; B's norm is nines.  Part 1
    from A and part 2 from B merged under ``--source A`` and published B's
    norm as A's checkpoint."""
    a, b = tmp_path / "source-a", tmp_path / "source-b"
    _write_source(a, {FIRST: _w(1), NORM: torch.ones(32, dtype=torch.bfloat16)})
    _write_source(b, {FIRST: _w(1), NORM: torch.full((32,), 9, dtype=torch.bfloat16)})
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    _export(a, pa, {FIRST: Q256}, "part-1.safetensors")
    _export(b, pb, {FIRST: Q256}, "part-2.safetensors")

    with pytest.raises(SystemExit) as refused:
        _merge([pa, pb], a, out)
    message = str(refused.value)
    assert "part-2.safetensors" in message and "part-b" in message
    assert _nothing_published(out)


def test_parts_cut_to_different_plans_refuse(tmp_path):
    """Case 2: part 1 exported under the whole plan, part 2 under an empty one.
    The merge unioned the plans and published ``plan[SECOND]`` over the raw
    bf16 tensor part 2 had passed through."""
    source = tmp_path / "source"
    _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, {}, "part-2.safetensors")

    with pytest.raises(SystemExit) as refused:
        _merge([pa, pb], source, out)
    assert "plan" in str(refused.value)
    assert _nothing_published(out)


# --- the assembly contract, obligation by obligation -------------------------

def test_a_planned_tensor_passed_through_raw_refuses(tmp_path):
    """Both parts carry the same plan, but part 2's output holds SECOND raw
    (the shape a stale or mis-run part has on disk).  The proof is against the
    part's actual header, not its config."""
    source = tmp_path / "source"
    _, tensors = _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, plan, "part-2.safetensors")
    save_file({SECOND: tensors[SECOND]}, str(pb / "part-2.safetensors"),
              metadata={"format": "pt"})
    (pb / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 0}, "weight_map": {SECOND: "part-2.safetensors"}}))
    _reseal(pb)

    with pytest.raises(SystemExit) as refused:
        _merge([pa, pb], source, out)
    assert SECOND in str(refused.value)
    assert _nothing_published(out)


def test_a_blob_at_another_rung_than_the_plan_refuses(tmp_path):
    """Part 2 encoded SECOND at 640 and then claims the plan's 896: the two
    configs agree, the header shows a blob, and only the blob's own manifest
    can say which rung was cut.  The merge reads it."""
    source = tmp_path / "source"
    _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, {FIRST: Q256, SECOND: 640}, "part-2.safetensors")
    _rewrite_config(pb, lambda c: (c.__setitem__("plan", dict(plan)),
                                   c.__setitem__("rungs_q256", [Q256])))

    with pytest.raises(SystemExit) as refused:
        _merge([pa, pb], source, out)
    message = str(refused.value)
    assert SECOND in message and "640" in message and str(Q256) in message
    assert _nothing_published(out)


def test_a_blob_for_another_tensor_under_a_planned_name_refuses(tmp_path):
    """Part 2's shard carries FIRST's blob under SECOND's name.  Same rung,
    same shape, right header -- the manifest's ``unit_id`` is the only thing
    that says whose weights these are."""
    source = tmp_path / "source"
    _two_shard_source(source, second=_w(3))
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, plan, "part-2.safetensors")
    with safe_open(str(pa / "part-1.safetensors"), framework="pt") as h:
        blob = h.get_tensor(FIRST + BLOB_SUFFIX)
    save_file({SECOND + BLOB_SUFFIX: blob}, str(pb / "part-2.safetensors"),
              metadata={"format": "pt"})
    _reseal(pb)

    with pytest.raises(SystemExit) as refused:
        _merge([pa, pb], source, out)
    message = str(refused.value)
    assert SECOND in message and FIRST in message
    assert _nothing_published(out)


def test_an_unsealed_legacy_part_refuses_by_name(tmp_path):
    """A part written before the stamp existed has no ``source`` block.  It is
    refused as unsealed -- not merged on the strength of its filenames -- and
    the message says what to do."""
    source = tmp_path / "source"
    _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, plan, "part-2.safetensors")
    _rewrite_config(pb, lambda c: c.pop("source"))

    with pytest.raises(SystemExit) as refused:
        _merge([pa, pb], source, out)
    message = str(refused.value)
    assert "part-b" in message and "source" in message and "re-export" in message.lower()
    assert _nothing_published(out)


def test_an_in_memory_export_is_not_a_shard_split_part(tmp_path):
    """``export_checkpoint`` writes tensors it was handed, not a checkpoint it
    read: its config says ``source: null`` and the merge refuses it, since
    there is nothing to bind it to."""
    source = tmp_path / "source"
    _, tensors = _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    pb.mkdir()
    export_checkpoint({SECOND: tensors[SECOND]}, {SECOND: Q256}, pb, grid=K2)
    (pb / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 0},
         "weight_map": {SECOND + BLOB_SUFFIX: "model.safetensors"}}))
    assert _config(pb)["source"] is None

    with pytest.raises(SystemExit) as refused:
        _merge([pa, pb], source, out)
    assert "part-b" in str(refused.value) and "source" in str(refused.value)
    assert _nothing_published(out)


def test_a_source_whose_shard_changed_since_the_export_refuses(tmp_path):
    """The stamp is of the bytes the part read; a shard rewritten afterwards
    (same tensors, other values) is a different checkpoint."""
    source = tmp_path / "source"
    _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, plan, "part-2.safetensors")
    save_file({FIRST: _w(9)}, str(source / "part-1.safetensors"), metadata={"format": "pt"})

    with pytest.raises(SystemExit) as refused:
        _merge([pa, pb], source, out)
    assert "part-1.safetensors" in str(refused.value)
    assert _nothing_published(out)


def test_valid_disjoint_parts_from_one_source_with_one_fulfilled_plan_merge(tmp_path):
    """The control: two parts, one source, one plan each part fulfilled for its
    own shard.  The merge publishes, the plan is the plan, and the merged
    checkpoint's ``source`` block is the whole-source identity."""
    source = tmp_path / "source"
    weight_map, tensors = _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, plan, "part-2.safetensors")

    _merge([pa, pb], source, out)

    config = _config(out)
    assert config["plan"] == plan
    assert config["rungs_q256"] == [Q256]
    assert config["merged_from"] == ["part-a", "part-b"]
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {n + BLOB_SUFFIX: s for n, s in weight_map.items()}
    assert config["source"] == source_part_identity(source)
    assert config["source"]["schema"] == SOURCE_PART_SCHEMA
    assert set(config["source"]["files"]) == set(weight_map.values())
    for name in plan:
        assert load_tessera_weight(out, name).shape == tensors[name].shape


# --- what the streaming exporter stamps ---------------------------------------

def test_the_exporter_stamps_the_source_it_read(tmp_path):
    source = tmp_path / "source"
    weight_map, _ = _two_shard_source(source)
    part = tmp_path / "part-a"
    _export(source, part, {FIRST: Q256, SECOND: Q256}, "part-1.safetensors")

    stamp = _config(part)["source"]
    assert stamp["schema"] == SOURCE_PART_SCHEMA
    assert stamp["config_sha256"] == sha256_file(source / "config.json")
    assert stamp["tensors"] == weight_map                       # the whole inventory
    assert stamp["files"] == {"part-1.safetensors":              # only what it read
                              sha256_file(source / "part-1.safetensors")}
    assert set(stamp["auxiliary_sha256"]) == {"config.json", "model.safetensors.index.json"}
    assert stamp == source_part_identity(source, {"part-1.safetensors"})


def test_a_plan_naming_a_tensor_in_no_shard_refuses_under_a_shard_filter(tmp_path):
    """A filtered run tolerates plan names that live in OTHER shards; a name
    that lives in no shard of the source is the mistyped plan the unfiltered
    path already refuses."""
    source = tmp_path / "source"
    _two_shard_source(source)
    with pytest.raises(KeyError, match="not present"):
        _export(source, tmp_path / "part", {FIRST: Q256, "model.nope.weight": Q256},
                "part-1.safetensors")


def test_a_source_index_that_misstates_its_shards_refuses_at_export(tmp_path):
    """The inventory a part stamps is the one the headers reproduce; an index
    naming a tensor no shard holds is refused before any encode."""
    source = tmp_path / "source"
    weight_map, _ = _two_shard_source(source)
    weight_map["model.ghost.weight"] = "part-1.safetensors"
    (source / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 0}, "weight_map": weight_map}))
    with pytest.raises(ValueError, match="index"):
        _export(source, tmp_path / "part", {FIRST: Q256}, "part-1.safetensors")


def test_a_driver_extra_config_cannot_overwrite_an_exporter_field(tmp_path):
    """``extra_config`` is a driver's own vocabulary; a key the exporter writes
    -- the source stamp above all -- is refused rather than silently replaced."""
    with pytest.raises(GrammarError, match="source"):
        export_checkpoint({FIRST: _w(1)}, {FIRST: Q256}, tmp_path, grid=K2,
                          extra_config={"source": {"schema": SOURCE_PART_SCHEMA}})


def test_the_seal_over_a_shard_is_the_file_digest(tmp_path):
    """``sha256_file`` is the modern path's digest; the stamp reuses it rather
    than defining a second one."""
    source = tmp_path / "source"
    _two_shard_source(source)
    stamp = source_part_identity(source, {"part-2.safetensors"})
    assert stamp["files"]["part-2.safetensors"] == hashlib.sha256(
        (source / "part-2.safetensors").read_bytes()).hexdigest()


# --- a completed output is immutable, and its bytes are sealed (tessera#337) ---

def test_a_failed_retry_into_a_completed_export_refuses_before_replacing_a_byte(tmp_path):
    """codex's reproduction, at its own boundary.

    Source A is exported successfully.  Source B carries the same tensor
    names, shapes and configuration at other values, and is re-exported into
    the SAME directory with a failure injected at its second unit -- the
    ordinary shape of a retry that dies part way.  The exporter used to
    overwrite shard by shard and replace the index and config only at the end,
    so this left B's first shard beside A's index, A's config and A's source
    seal: a checkpoint that loaded, verified against A, merged clean and
    decoded to ``[9.0, 1.03125]`` where the config it published prices
    ``[1.03125, 1.03125]``.

    A completed output is now immutable, so the refusal lands before the run
    reads a shard, encodes a unit or replaces a byte: the injected failure is
    never reached, and A's artifact is intact down to the digest of every file
    -- the two encoded shards and the passthrough one alike.
    """
    from tessera import export as export_module

    source_a, source_b = tmp_path / "source-a", tmp_path / "source-b"
    _three_shard_source(source_a, seeds=(1, 2), norm=1.0)
    _three_shard_source(source_b, seeds=(5, 6), norm=9.0)
    part = tmp_path / "part"
    _export_all(source_a, part)
    before = _bytes_on_disk(part)
    assert set(before) >= {*SHARDS, "tessera_config.json",
                           "model.safetensors.index.json"}

    encoded = []
    real = export_module.encode_linear

    def fail_on_the_second_unit(weight, **kwargs):
        encoded.append(kwargs.get("name"))
        if len(encoded) == 2:
            raise RuntimeError("injected second-unit failure during retry")
        return real(weight, **kwargs)

    with patch.object(export_module, "encode_linear", fail_on_the_second_unit):
        with pytest.raises(FileExistsError, match="tessera_config.json"):
            _export_all(source_b, part)

    assert encoded == []                      # refused before it encoded anything
    assert _bytes_on_disk(part) == before     # and before it replaced anything
    for name in PLANNED:                      # the artifact A sealed still decodes
        assert load_tessera_weight(part, name).shape == (32, 256)


def test_a_retry_into_an_output_no_run_ever_sealed_is_allowed_and_completes(tmp_path):
    """Immutability is of a COMPLETED export, not of any directory.

    A run that died before writing ``tessera_config.json`` never published an
    artifact -- there is nothing to preserve and nothing a reader or the merge
    would accept -- so a retry writes into it, replaces its leftovers and
    seals what it actually wrote.  Refusing here instead would strand every
    interrupted export behind a manual delete for no gain.
    """
    source = tmp_path / "source"
    _three_shard_source(source)
    part = tmp_path / "part"
    part.mkdir()
    leftovers = b"a shard from a run that died before it sealed anything"
    (part / "part-1.safetensors").write_bytes(leftovers)

    _export_all(source, part)

    stamp = _config(part)["output"]
    assert stamp["files"]["part-1.safetensors"] == sha256_file(part / "part-1.safetensors")
    assert stamp["files"]["part-1.safetensors"] != hashlib.sha256(leftovers).hexdigest()


def test_the_exporter_stamps_the_shards_it_wrote(tmp_path):
    """The receipt the merge proves the bytes by: one sha256 per shard this
    run wrote, encoded and passthrough alike, under its own schema."""
    source = tmp_path / "source"
    _three_shard_source(source)
    part = tmp_path / "part"
    _export_all(source, part)

    stamp = _config(part)["output"]
    assert stamp["schema"] == OUTPUT_PART_SCHEMA
    assert set(stamp["files"]) == set(SHARDS)
    assert stamp == output_part_identity(part, SHARDS)
    # part-3 holds NORM, which the plan never names: a passthrough shard is
    # bytes this export published and is sealed like any other.
    assert stamp["files"]["part-3.safetensors"] == sha256_file(part / "part-3.safetensors")


@pytest.mark.parametrize("swapped, payload", [("part-1.safetensors", "encoded"),
                                              ("part-3.safetensors", "passthrough")])
def test_a_mixed_output_directory_refuses_before_the_merge_publishes(tmp_path, swapped,
                                                                     payload):
    """The second layer, for a mixture this exporter can no longer make.

    One shard of a completed part is replaced by the same-named shard of
    another export of another source: the same tensor under the same name, at
    the same rung, of the same geometry, and (for the passthrough case) the
    same dtype and shape.  Every check the merge had -- output names, shard
    headers, ``unit_id``, ``root_q256``, geometry, and the part's source seal,
    which describes what went in and is untouched -- accepts it.  Only the
    output seal can say these are not the bytes that export wrote.
    """
    source_a, source_b = tmp_path / "source-a", tmp_path / "source-b"
    _three_shard_source(source_a, seeds=(1, 2), norm=1.0)
    _three_shard_source(source_b, seeds=(5, 6), norm=9.0)
    part_a, part_b = tmp_path / "part-a", tmp_path / "part-b"
    _export_all(source_a, part_a)
    _export_all(source_b, part_b)
    (part_a / swapped).write_bytes((part_b / swapped).read_bytes())

    out = tmp_path / "merged"
    with pytest.raises(SystemExit) as refused:
        _merge([part_a], source_a, out)
    message = str(refused.value)
    assert swapped in message and "337" in message, payload
    assert _nothing_published(out)


def test_a_part_with_no_output_seal_refuses_by_name(tmp_path):
    """A part written before the stamp existed cannot say which bytes its
    export wrote, and its ``source`` block cannot say it for it: that stamp is
    of the input and verifies whatever survived in the output directory.  So
    it is refused like an unsealed part, with the same answer -- re-export."""
    source = tmp_path / "source"
    _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, plan, "part-2.safetensors")
    _rewrite_config(pb, lambda c: c.pop("output"))

    with pytest.raises(SystemExit) as refused:
        _merge([pa, pb], source, out)
    message = str(refused.value)
    assert "part-b" in message and "output" in message and "re-export" in message.lower()
    assert _nothing_published(out)


def test_the_merged_checkpoint_seals_its_own_bytes(tmp_path):
    """The merged config's ``output`` is the union of the parts' seals -- the
    ones just verified, over files copied unchanged -- and not the first
    part's, which is the config the merge starts from."""
    source = tmp_path / "source"
    _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, plan, "part-2.safetensors")

    _merge([pa, pb], source, out)

    stamp = _config(out)["output"]
    assert stamp == output_part_identity(out, ["part-1.safetensors", "part-2.safetensors"])
    assert stamp != _config(pa)["output"]
    assert stamp["files"]["part-1.safetensors"] == _config(pa)["output"]["files"]["part-1.safetensors"]


def test_a_merge_that_dies_part_way_leaves_no_seal_over_the_shards_it_moved(tmp_path):
    """The exporter's window, one step later, in the merge's own publication.

    ``--out`` may already hold a complete checkpoint -- a re-run of the merge,
    or one of the parts, which the self-copy skip in ``main`` exists for.
    Shards land there one at a time and the index and config that describe
    them are written only after the last one, so a transfer that dies part way
    would leave new bytes under the old checkpoint's seal: exactly the mixture
    this file is about, published rather than merged.

    The seal cannot be preserved here -- the shards are being replaced in
    place -- so it is removed BEFORE the first transfer, and an unfinished
    merge leaves an unsealed directory the reader and this merge both refuse.
    """
    source = tmp_path / "source"
    _two_shard_source(source)
    pa, pb, out = tmp_path / "part-a", tmp_path / "part-b", tmp_path / "merged"
    plan = {FIRST: Q256, SECOND: Q256}
    _export(source, pa, plan, "part-1.safetensors")
    _export(source, pb, plan, "part-2.safetensors")
    _merge([pa, pb], source, out)
    assert (out / "tessera_config.json").exists()      # a complete checkpoint

    transfers = []
    real = merge.shutil.copy2

    def fail_on_the_second_transfer(src, dst):
        transfers.append(dst)
        if len(transfers) == 2:
            raise OSError("injected transfer failure part way through the merge")
        return real(src, dst)

    with patch.object(merge.shutil, "copy2", fail_on_the_second_transfer):
        with pytest.raises(OSError, match="injected"):
            _merge([pa, pb], source, out)

    assert len(transfers) == 2                          # it did replace bytes
    assert not (out / "tessera_config.json").exists()
    assert not (out / "model.safetensors.index.json").exists()
