"""The model-level walk: a plan in, a checkpoint out, the same weights back."""
from fractions import Fraction

import pytest
import torch

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.errors import GrammarError
from tessera.export import (
    encode_linear,
    export_checkpoint,
    load_tessera_weight,
    read_checkpoint_config,
)

K2 = tuple_grid(E2M1_GRID, 2)


def _w(rows=64, cols=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g).bfloat16()


def test_encode_linear_round_trips_through_its_own_bytes():
    unit = encode_linear(_w(), grid=K2, q256=896, name="l0")
    assert unit.exact_bytes > 0 and unit.params == 64 * 256


def test_declared_bpp_is_the_bytes_that_were_written():
    """The accountant divides real bytes by real params -- no estimate."""
    unit = encode_linear(_w(), grid=K2, q256=896, name="l0")
    assert unit.bpp == Fraction(unit.exact_bytes * 8, 64 * 256)


def test_checkpoint_round_trip_is_exact(tmp_path):
    tensors = {
        "layers.0.mlp.gate_proj.weight": _w(seed=1),
        "layers.0.mlp.up_proj.weight": _w(seed=2),
        "model.norm.weight": torch.ones(256).bfloat16(),
    }
    plan = {n: 896 for n in tensors if n.endswith("proj.weight")}
    report = export_checkpoint(tensors, plan, tmp_path, grid=K2)

    assert len(report.units) == 2
    assert report.quantized_params == 2 * 64 * 256
    for name in plan:
        got = load_tessera_weight(tmp_path, name)
        direct = encode_linear(tensors[name], grid=K2, q256=896, name=name)
        from tessera.unit_artifact import read_unit_artifact
        assert torch.equal(got, read_unit_artifact(direct.blob))


def test_passthrough_tensors_survive_verbatim(tmp_path):
    from safetensors import safe_open

    norm = torch.arange(8, dtype=torch.bfloat16)
    tensors = {"w": _w(), "model.norm.weight": norm}
    export_checkpoint(tensors, {"w": 896}, tmp_path, grid=K2)
    with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as h:
        assert torch.equal(h.get_tensor("model.norm.weight"), norm)
        # the quantized weight is NOT present under its own name, so a reader
        # that does not understand Tessera cannot load the blob as a weight
        assert "w" not in h.keys() and "w.tessera" in h.keys()


def test_config_declares_the_route_unbacked(tmp_path):
    """Principle 9: no runtime decodes this container, and a gate can read so."""
    export_checkpoint({"w": _w()}, {"w": 896}, tmp_path, grid=K2)
    config = read_checkpoint_config(tmp_path)
    assert config["route_status"] == "unbacked"
    assert config["grid"]["arity"] == 2


def test_body_bpp_excludes_passthrough_params(tmp_path):
    """principle 12: bpp is over quantizable params only."""
    tensors = {"w": _w(), "embed": torch.zeros(1000, 256).bfloat16()}
    report = export_checkpoint(tensors, {"w": 896}, tmp_path, grid=K2)
    assert report.quantized_params == 64 * 256
    assert report.passthrough_bytes == 1000 * 256 * 2
    assert float(report.body_bpp) < 8.0


def test_a_plan_that_names_a_missing_tensor_refuses(tmp_path):
    with pytest.raises(KeyError, match="not present"):
        export_checkpoint({"w": _w()}, {"nope": 896}, tmp_path, grid=K2)


def test_rows_indivisible_by_arity_refuse():
    with pytest.raises(GrammarError, match="arity"):
        encode_linear(_w(rows=63), grid=K2, q256=896, name="odd")


def test_r896_k2_declares_four_bits_per_parameter(tmp_path):
    """The arity factor is silent if wrong, so pin it to the number it means.

    ``R896`` is a PER-POSITION rate of 3.5 bits; the trellis spends 7 bits per
    code because a code spans two positions.  Passing the per-position number
    to the artifact builder yields a legal artifact declaring half its true
    rate.  On a unit large enough for the fixed forest planes to amortise, the
    body must land on 4.0 bpp -- 3.5 of payload plus the segment-2b scales.
    """
    report = export_checkpoint(
        {"w": _w(rows=512, cols=1024)}, {"w": 896}, tmp_path, grid=K2
    )
    assert 3.95 < float(report.body_bpp) < 4.10, float(report.body_bpp)


def test_config_declares_the_tp_degree_it_was_encoded_for(tmp_path):
    """A unit is a blob, not a sliceable tensor, so TP degree is baked in.

    The trellis runs down rows inside each column; a row-parallel split -- what
    a column-parallel Linear needs -- cuts it along its own state path. EXL3
    narrows tensor dims and stays TP-agnostic; Tessera cannot, so the artifact
    has to say which degree it was built for rather than fail obscurely at load.
    """
    export_checkpoint({"w": _w()}, {"w": 896}, tmp_path, grid=K2)
    assert read_checkpoint_config(tmp_path)["tp_size"] == 1


def test_a_sharded_export_reads_back_through_its_index(tmp_path):
    """The streaming exporter writes one shard per input shard plus an index.

    A reader that assumes the single-file ``model.safetensors`` layout can read
    back nothing this format exports at scale -- which is every checkpoint the
    streaming path exists for.  This is the only test that exercises the
    multi-shard read, so without it the sharded artifact is write-only.
    """
    import json

    from safetensors.torch import save_file

    from tessera.export import export_checkpoint_streaming

    src = tmp_path / "src"
    src.mkdir()
    tensors = {
        "layers.0.mlp.gate_proj.weight": _w(seed=1),
        "layers.1.mlp.gate_proj.weight": _w(seed=2),
    }
    weight_map = {}
    for i, (name, tensor) in enumerate(tensors.items(), start=1):
        shard = f"model-0000{i}-of-00002.safetensors"
        save_file({name: tensor}, str(src / shard), metadata={"format": "pt"})
        weight_map[name] = shard
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )

    out = tmp_path / "out"
    plan = {n: 896 for n in tensors}
    report = export_checkpoint_streaming(
        src, out, plan, grid=K2, device="cpu", copy_aux=False
    )
    assert len(report.units) == 2
    assert not (out / "model.safetensors").exists()  # sharded, not single-file

    from tessera.unit_artifact import read_unit_artifact

    for name in plan:
        direct = encode_linear(tensors[name], grid=K2, q256=896, name=name)
        assert torch.equal(
            load_tessera_weight(out, name), read_unit_artifact(direct.blob)
        )


def test_loading_an_absent_unit_names_the_index(tmp_path):
    export_checkpoint({"a.weight": _w()}, {"a.weight": 896}, tmp_path, grid=K2)
    with pytest.raises(KeyError):
        load_tessera_weight(tmp_path, "nope.weight")
