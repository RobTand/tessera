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
from tessera.serving_parts import (SOURCE_PART_SCHEMA, sha256_file,
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


def _nothing_published(out):
    """A refusal happens before publication: no config, no index, no shard."""
    if not out.exists():
        return True
    return not any(out.iterdir())


def _two_shard_source(root, second=None):
    tensors = {FIRST: _w(1), SECOND: _w(2) if second is None else second}
    return _write_source(root, tensors), tensors


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
