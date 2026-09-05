"""The standalone stock exporter joins the fused A-side scale too (#305).

``input_global_scale`` is capacity over amax.  vLLM's NVFP4 method reduces a
fused module's member scales with ``layer.input_global_scale.max()`` -- the
member with the SMALLEST calibrated range -- and then stores every group-16
block scale as ``e4m3(block_amax / 6 * scale)`` clamped at 448, so a value too
large for the tensor's true amax saturates and the wider members' peaks clip
silently.  A fused module's one GEMM quantises ONE input tensor for every
member, so the value that must be served is the MINIMUM member scale:
``tessera.fused.shared_input_global_scale``, the join's one home (PR #275, and
the declared one-bf16-ULP divergence bound of #283 / PR #284).

``experiments/export_tessera_serving.py`` and its stock twin were fixed there;
this separate sanctioned CLI was not.  It copied each donor member's scale
through unchanged and joined only the *weight* globals, so a donor whose
siblings differ at all -- inside the accepted bound -- served an activation
quantizer chosen by vLLM's max reduction rather than by Tessera's owned rule
(#305).  ``tests/test_export_input_scale_join.py`` covers the serving exporter
and its twin; ``tests/test_stock_shard_scheduling.py`` covers this exporter's
W4A16 arm, which has no donor input scales at all, so both could pass while
this path kept the old behaviour.

The join is resolved from the PLANNED fused roster before the output directory
exists, which is what makes it correct for siblings that straddle source
shards: an earlier output shard can no longer publish an unjoined sibling
before a later one arrives (#212's schedule, unchanged).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from tessera.errors import GrammarError  # noqa: E402
from tessera.fused import shared_input_global_scale  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "export_stock_compressed", ROOT / "experiments" / "export_stock_compressed.py")
stock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stock)

LAYER = "model.layers.0.self_attn."
Q, K, V = (f"{LAYER}{role}.weight" for role in ("q_proj", "k_proj", "v_proj"))
O = f"{LAYER}o_proj.weight"
MEMBERS = (Q, K, V)

#: One bf16 ULP at 4.0 -- inside ``FUSED_INPUT_SCALE_ULP``, so this spread is
#: one calibrated amax spelled twice and the join accepts it.
IN_BOUND = (4.0, 4.0 * (1.0 + 2.0 ** -8), 4.0)


def _weights():
    generator = torch.Generator().manual_seed(305)
    return {name: torch.randn(32, 32, generator=generator).bfloat16()
            for name in (Q, K, V, O)}


def _source(src: Path, shards: "dict[str, list[str]]") -> None:
    weights = _weights()
    src.mkdir(parents=True)
    weight_map = {}
    for shard, names in shards.items():
        save_file({n: weights[n].contiguous() for n in names},
                  str(src / shard), metadata={"format": "pt"})
        weight_map.update({n: shard for n in names})
    if len(shards) > 1:
        (src / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": 0}, "weight_map": weight_map}))
    (src / "config.json").write_text(json.dumps({"model_type": "test"}))


def _donor(path: Path, scales: "dict[str, float]") -> None:
    save_file({name[: -len(".weight")] + ".input_global_scale":
               torch.tensor([value], dtype=torch.float32)
               for name, value in scales.items()},
              str(path), metadata={"format": "pt"})


def _export(tmp_path, monkeypatch, *, scales, shards=None, out="out"):
    """Run the real W4A4 export over a donor with these member scales."""
    src = tmp_path / f"src-{out}"
    _source(src, shards or {"model.safetensors": [Q, K, V, O]})
    donor = tmp_path / f"donor-{out}.safetensors"
    _donor(donor, scales)
    destination = tmp_path / out
    monkeypatch.setattr(
        "sys.argv",
        ["export_stock_compressed.py", str(src), str(destination),
         "--activations", "w4a4", "--input-scales", str(donor),
         "--device", "cpu", "--no-verify"])
    stock.main()
    return destination


def _written_scales(out: Path, names=MEMBERS) -> "dict[str, float]":
    """Every member's ``input_global_scale`` as the checkpoint holds it."""
    index = out / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {}
        for path in out.glob("*.safetensors"):
            with safe_open(str(path), framework="pt") as handle:
                weight_map.update({name: path.name for name in handle.keys()})
    found = {}
    for name in names:
        key = name[: -len(".weight")] + ".input_global_scale"
        with safe_open(str(out / weight_map[key]), framework="pt") as handle:
            found[name] = float(handle.get_tensor(key).float().reshape(-1)[0])
    return found


def test_a_within_bound_donor_spread_is_joined_on_every_member(tmp_path, monkeypatch):
    """The #305 repro.  A donor whose q/k/v scales differ by one rounding: the
    exporter used to write all three through unchanged, leaving vLLM's
    ``.max()`` to pick 4.015625 -- the smallest calibrated range, the clipping
    side.  Every member must carry the shared helper's value instead, and the
    comparison is against the helper rather than a typed number."""
    scales = dict(zip(MEMBERS, IN_BOUND)) | {O: 4.0}
    out = _export(tmp_path, monkeypatch, scales=scales)
    joined = shared_input_global_scale(list(IN_BOUND), list(MEMBERS))
    written = _written_scales(out)
    assert written == {name: joined for name in MEMBERS}, written
    assert max(written.values()) == joined  # nothing left for vLLM to reduce


def test_an_already_unified_donor_is_unchanged(tmp_path, monkeypatch):
    """The control: a donor whose calibrator already wrote one amax on every
    sibling (PrismaQuant's own join, spread exactly 0) exports the value it
    arrived with."""
    scales = {name: 4.0 for name in MEMBERS} | {O: 4.0}
    out = _export(tmp_path, monkeypatch, scales=scales)
    assert _written_scales(out) == {name: 4.0 for name in MEMBERS}


def test_siblings_spanning_source_shards_all_carry_the_joined_value(tmp_path, monkeypatch):
    """The join is resolved from the planned roster before any shard is
    written, so the member that lands in the FIRST output shard carries the
    same joined value as the ones a later shard delivers.  Resolving at group
    completion instead would publish q's own scale before k and v arrived
    (#212's schedule is what makes the group wait)."""
    out = _export(tmp_path, monkeypatch,
                  scales=dict(zip(MEMBERS, IN_BOUND)) | {O: 4.0},
                  shards={"part-1.safetensors": [Q, O],
                          "part-2.safetensors": [K, V]})
    joined = shared_input_global_scale(list(IN_BOUND), list(MEMBERS))
    written = _written_scales(out)
    assert written == {name: joined for name in MEMBERS}, written
    weight_map = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert weight_map[Q[: -len(".weight")] + ".input_global_scale"] == "part-1.safetensors"


def test_an_unfused_linear_keeps_its_own_donor_scale(tmp_path, monkeypatch):
    """``o_proj`` is its own vLLM Linear with its own input tensor: the join is
    per fused module, not per layer, and a sibling's scale must not reach it."""
    scales = dict(zip(MEMBERS, IN_BOUND)) | {O: 2.5}
    out = _export(tmp_path, monkeypatch, scales=scales)
    assert _written_scales(out, (O,)) == {O: 2.5}


def test_a_donor_beyond_the_declared_bound_is_refused_before_any_output(
        tmp_path, monkeypatch):
    """A 2x spread is two calibrations, not one amax spelled twice, and a
    joined value would serve a distribution nobody measured.  The standalone
    exporter now refuses it by name where the serving exporter already does --
    before the donor's members can be written -- rather than exporting a
    checkpoint whose activation quantizer vLLM chooses."""
    with pytest.raises(GrammarError) as caught:
        _export(tmp_path, monkeypatch,
                scales={Q: 4.0, K: 2.0, V: 4.0, O: 4.0})
    message = str(caught.value)
    assert "input_global_scale" in message
    assert K[: -len(".weight")] in message
    assert "bf16" in message
    assert list((tmp_path / "out").glob("*.safetensors")) == []


def test_the_manifest_records_the_scale_that_was_written(tmp_path, monkeypatch):
    """Priced == written == served: the receipt states the A-side value each
    unit actually carries, so the join is readable off the manifest instead of
    being inferred from the donor."""
    out = _export(tmp_path, monkeypatch,
                  scales=dict(zip(MEMBERS, IN_BOUND)) | {O: 2.5})
    manifest = json.loads((out / "tessera_stock_manifest.json").read_text())
    joined = shared_input_global_scale(list(IN_BOUND), list(MEMBERS))
    recorded = {name: manifest["units"][name]["input_global_scale"]
                for name in (*MEMBERS, O)}
    assert recorded == {Q: joined, K: joined, V: joined, O: 2.5}, recorded
    assert recorded == {**_written_scales(out), O: 2.5}
