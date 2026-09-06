"""A dense owner's roles are the runtime's output partitions (tessera#377).

RED FIRST.  LFM2.5-8B-A1B's ``short_conv.in_proj`` is ONE checkpoint tensor
that vLLM builds as ``MergedColumnParallelLinear(output_sizes=[dim] * 3)``.
The exporter's name rule (``dense_ownership.fused_module``) sees one tensor and
declared one role; ``sharding.plan_shard`` pairs the declared roles with
``output_partition_sizes`` by position and refused the load::

    The checkpoint declares 1 roles ['in_proj'] but the layer asks for 3 output
    partitions [2048, 2048, 2048]

The gate was right and stays.  The fix is on the producer: the construction
census records each Linear's ``output_sizes``, the contract carries it, and the
exporter derives its roles from it -- cutting a single source tensor by row in
partition order, exactly as ``MergedColumnParallelLinear.weight_loader`` does
with ``loaded_shard_id=None``.  A census that predates the field attests
nothing, and the manifest says so.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tessera.serving.contract import (
    construction_entry_from_receipt, load_serving_contract, output_partitions,
    validate_serving_contract)
from tessera.serving.dense_ownership import Member, partition_members

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "docs" / "measurements" / "construction"
LFM_RECEIPT = RECEIPTS / "lfm25-8b-a1b-eugr-0281rc1.json"
IN_PROJ = "model.layers.*.short_conv.in_proj"


def _receipt(name: str) -> dict:
    return json.loads((RECEIPTS / name).read_text())


OUT_PROJ = "model.layers.*.short_conv.out_proj"


def _lfm_receipt_with_output_sizes(sizes=(2048, 2048, 2048), out_proj=(2048,)) -> dict:
    """The committed LFM receipt, with the two ShortConv rows attested."""
    receipt = _receipt(LFM_RECEIPT.name)
    for row in receipt["linears"]:
        if row["prefix_pattern"] == IN_PROJ:
            row["output_sizes"] = list(sizes)
        elif row["prefix_pattern"] == OUT_PROJ:
            row["output_sizes"] = list(out_proj)
    return receipt


# --- the geometry rule --------------------------------------------------------

def test_one_source_tensor_over_several_partitions_is_cut_by_row_in_order():
    parts = partition_members("m.in_proj", ["m.in_proj.weight"], {"m.in_proj.weight": 6144},
                              [2048, 2048, 2048])
    assert parts == (
        Member("in_proj.0", "m.in_proj.weight", 0, 2048),
        Member("in_proj.1", "m.in_proj.weight", 2048, 2048),
        Member("in_proj.2", "m.in_proj.weight", 4096, 2048))


def test_as_many_partitions_as_tensors_keeps_the_tensors_as_roles():
    rows = {"a.q_proj.weight": 64, "a.k_proj.weight": 16, "a.v_proj.weight": 16}
    parts = partition_members("a.qkv_proj", list(rows), rows, [64, 16, 16])
    assert [(p.role, p.tensor, p.row_offset, p.rows) for p in parts] == [
        ("q_proj", "a.q_proj.weight", 0, 64), ("k_proj", "a.k_proj.weight", 0, 16),
        ("v_proj", "a.v_proj.weight", 0, 16)]


def test_unattested_geometry_declares_the_tensors_as_before():
    rows = {"a.q_proj.weight": 64, "a.k_proj.weight": 16, "a.v_proj.weight": 16}
    assert partition_members("a.qkv_proj", list(rows), rows, None) == \
        partition_members("a.qkv_proj", list(rows), rows, [64, 16, 16])
    assert partition_members("m.o_proj", ["m.o_proj.weight"], {"m.o_proj.weight": 64}, None) == (
        Member("o_proj", "m.o_proj.weight", 0, 64),)


def test_a_member_whose_rows_are_not_its_partition_is_refused():
    rows = {"a.q_proj.weight": 64, "a.k_proj.weight": 16, "a.v_proj.weight": 16}
    with pytest.raises(ValueError, match="a.k_proj.weight has 16 rows but the runtime builds"):
        partition_members("a.qkv_proj", list(rows), rows, [64, 32, 32])


def test_partitions_that_do_not_sum_to_the_tensor_are_refused():
    with pytest.raises(ValueError, match="sum to 6000"):
        partition_members("m.in_proj", ["m.in_proj.weight"], {"m.in_proj.weight": 6144},
                          [2000, 2000, 2000])


def test_several_tensors_cannot_be_recut_to_a_different_count():
    rows = {"a.q_proj.weight": 64, "a.k_proj.weight": 16, "a.v_proj.weight": 16}
    with pytest.raises(ValueError, match="cannot be paired"):
        partition_members("a.qkv_proj", list(rows), rows, [96])


# --- the census and the contract ----------------------------------------------

def test_the_committed_receipts_predate_the_field_and_attest_nothing():
    """Until a receipt is re-taken, ``output_partitions`` is None, not [rows]."""
    for path in sorted(RECEIPTS.glob("*.json")):
        entry = construction_entry_from_receipt(json.loads(path.read_text()))
        assert "output_sizes" not in entry, path.name
    entry = construction_entry_from_receipt(_receipt(LFM_RECEIPT.name))
    assert output_partitions(entry, "model.layers.0.conv.in_proj") is None


def test_a_receipt_that_records_output_sizes_attests_the_partitions_by_checkpoint_name():
    entry = construction_entry_from_receipt(_lfm_receipt_with_output_sizes())
    assert entry["output_sizes"] == {IN_PROJ: [2048, 2048, 2048], OUT_PROJ: [2048]}
    # The checkpoint spells it ``conv``; the runtime builds ``short_conv``.
    assert output_partitions(entry, "model.layers.0.conv.in_proj") == [2048, 2048, 2048]
    assert output_partitions(entry, "model.layers.0.conv.out_proj") == [2048]
    # A pattern the receipt did not attest is None, not [rows].
    assert output_partitions(entry, "model.layers.0.self_attn.out_proj") is None


def test_a_pattern_whose_members_disagree_on_geometry_attests_nothing():
    receipt = _lfm_receipt_with_output_sizes()
    row = next(r for r in receipt["linears"] if r["prefix_pattern"] == IN_PROJ)
    row.setdefault("disagreements", {})["output_sizes"] = ["model.layers.1.short_conv.in_proj"]
    entry = construction_entry_from_receipt(receipt)
    assert entry["output_sizes"] == {OUT_PROJ: [2048]}
    assert output_partitions(entry, "model.layers.0.conv.in_proj") is None


def _contract_with_lfm_entry(entry: dict) -> dict:
    contract = copy.deepcopy(load_serving_contract())
    rows = contract["construction"]["architectures"]
    for i, row in enumerate(rows):
        if row["architecture"] == entry["architecture"]:
            entry = dict(entry, receipt=row.get("receipt", "docs/measurements/construction/x.json"))
            rows[i] = entry
            break
    else:
        raise AssertionError("no LFM entry in the packaged contract")
    return contract


def test_the_validator_accepts_an_attested_table_and_refuses_one_a_gate_could_misread():
    good = construction_entry_from_receipt(_lfm_receipt_with_output_sizes())
    validate_serving_contract(_contract_with_lfm_entry(good))
    unknown = dict(good, output_sizes={"model.layers.*.no_such_proj": [1]})
    with pytest.raises(ValueError, match="walked no Linear"):
        validate_serving_contract(_contract_with_lfm_entry(unknown))
    empty = dict(good, output_sizes={IN_PROJ: []})
    with pytest.raises(ValueError, match="non-empty list of positive"):
        validate_serving_contract(_contract_with_lfm_entry(empty))
    disagreeing = dict(good, disagreements=[
        {"prefix_pattern": IN_PROJ, "fields": {"output_sizes": ["model.layers.1.short_conv.in_proj"]}}])
    disagreeing["offered"] = [p for p in disagreeing["offered"] if p != IN_PROJ]
    with pytest.raises(ValueError, match="members disagreed"):
        validate_serving_contract(_contract_with_lfm_entry(disagreeing))


# --- the export ---------------------------------------------------------------

torch = pytest.importorskip("torch")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "export_tessera_serving", ROOT / "experiments" / "export_tessera_serving.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)

LAYER = "model.layers.0."
IN_PROJ_TENSOR = LAYER + "conv.in_proj.weight"
OUT_PROJ_TENSOR = LAYER + "conv.out_proj.weight"


def _checkpoint(tmp_path: Path, rows: int = 96, cols: int = 32) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    torch.manual_seed(0)
    save_file({IN_PROJ_TENSOR: torch.randn(rows, cols, dtype=torch.bfloat16),
               OUT_PROJ_TENSOR: torch.randn(cols, cols, dtype=torch.bfloat16)},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({
        "architectures": ["Lfm2MoeForCausalLM"], "model_type": "lfm2_moe",
        "hidden_size": cols, "num_hidden_layers": 1}))
    return src


def _export(tmp_path, monkeypatch, entry, *extra):
    src = _checkpoint(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(exporter, "construction_entry", lambda architectures: entry)
    monkeypatch.setattr(
        "sys.argv", ["export_tessera_serving.py", str(src), str(out),
                     "--grid", "BF16", "--device", "cpu", "--no-verify", *extra])
    exporter.main()
    return out


def test_export_declares_the_attested_partitions_as_row_sliced_roles(tmp_path, monkeypatch):
    entry = construction_entry_from_receipt(_lfm_receipt_with_output_sizes((32, 32, 32), (32,)))
    out = _export(tmp_path, monkeypatch, entry)
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    scheme = next(g["scheme"] for g in config["config_groups"].values()
                  if g["targets"] == [LAYER + "conv.in_proj"])
    assert scheme["roles"] == [["in_proj.0", 32], ["in_proj.1", 32], ["in_proj.2", 32]]
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    module = manifest["modules"][LAYER + "conv.in_proj"]
    assert module["geometry_attested"] is True
    assert [(r["tensor"], r["role"], r["row_offset"], r["rows"], r["source_rows"])
            for r in module["roles"]] == [
        (IN_PROJ_TENSOR, "in_proj.0", 0, 32, 96), (IN_PROJ_TENSOR, "in_proj.1", 32, 32, 96),
        (IN_PROJ_TENSOR, "in_proj.2", 64, 32, 96)]
    assert manifest["modules"][LAYER + "conv.out_proj"]["roles"][0]["role"] == "out_proj"
    geometry = manifest["serving_gate"]["geometry"]
    assert geometry == {"attested_modules": 2, "unattested_modules": [],
                        "row_sliced_modules": [LAYER + "conv.in_proj"]}
    # Two owners, four roles: the manifest counts units as roles, and the
    # three row windows of one source tensor are three of them.
    assert manifest["totals"]["modules"] == 2 and manifest["totals"]["units"] == 4
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        keys = set(handle.keys())
    assert LAYER + "conv.in_proj.wire_bytes" in keys and IN_PROJ_TENSOR not in keys


def test_the_container_carries_three_members_that_decode_to_the_source_rows(tmp_path, monkeypatch):
    """What the runtime will parse: three members, in partition order, rows intact."""
    from tessera.fused import parse_fused
    entry = construction_entry_from_receipt(_lfm_receipt_with_output_sizes((32, 32, 32), (32,)))
    out = _export(tmp_path, monkeypatch, entry)
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        blob = bytes(handle.get_tensor(LAYER + "conv.in_proj.wire_bytes").numpy().tobytes())
    members = parse_fused(blob)
    assert [(m.name, m.rows) for m in members] == [("in_proj.0", 32), ("in_proj.1", 32), ("in_proj.2", 32)]


def test_an_unattested_census_declares_per_tensor_and_says_so(tmp_path, monkeypatch):
    entry = construction_entry_from_receipt(_receipt(LFM_RECEIPT.name))
    out = _export(tmp_path, monkeypatch, entry)
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    module = manifest["modules"][LAYER + "conv.in_proj"]
    assert module["geometry_attested"] is False
    assert [(r["role"], r["rows"]) for r in module["roles"]] == [("in_proj", 96)]
    geometry = manifest["serving_gate"]["geometry"]
    assert geometry["attested_modules"] == 0
    assert geometry["unattested_modules"] == [LAYER + "conv.in_proj", LAYER + "conv.out_proj"]
    assert geometry["row_sliced_modules"] == []


def test_a_checkpoint_that_disagrees_with_the_attested_geometry_is_refused_before_encoding(
        tmp_path, monkeypatch):
    entry = construction_entry_from_receipt(_lfm_receipt_with_output_sizes((40, 40, 40), (32,)))
    with pytest.raises(SystemExit) as excinfo:
        _export(tmp_path, monkeypatch, entry)
    message = str(excinfo.value)
    assert "conv.in_proj" in message and "sum to 120" in message
    assert not (tmp_path / "out" / "model.safetensors").exists()


def test_the_twin_refuses_a_row_sliced_module(tmp_path, monkeypatch):
    entry = construction_entry_from_receipt(_lfm_receipt_with_output_sizes((32, 32, 32), (32,)))
    with pytest.raises(SystemExit, match="--stock-twin was given"):
        _export(tmp_path, monkeypatch, entry, "--stock-twin", str(tmp_path / "twin"))
