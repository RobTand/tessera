"""The stock exporter completes fused groups across source shards (#212).

``export_stock_compressed.py`` examined its NVFP4 fused groups after every
source shard and compared the partial group against the full q/k/v (gate/up)
roster, so a supported safetensors checkpoint failed export solely because its
fused siblings straddle source shards -- something a source index is entirely
free to do.  The main serving exporter already waits for pending members with
a weights cache; this holds the stock path to the same schedule:

* siblings split across shards produce EXACTLY the stock tensors the
  single-shard source produces, through one exact global share;
* the output index points at the shard the completed group was actually
  written to;
* a population that can never complete -- a sibling absent from the source or
  not planned NVFP4 -- is refused BEFORE the first encode, and a group left
  pending at end of input is refused there, never at a shard boundary.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "export_stock_compressed", ROOT / "experiments" / "export_stock_compressed.py")
stock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stock)

LAYER = "model.layers.0.self_attn."
Q, K, V = (f"{LAYER}{role}.weight" for role in ("q_proj", "k_proj", "v_proj"))
EMBED = "model.embed_tokens.weight"


def _tensors():
    generator = torch.Generator().manual_seed(29)
    return {name: torch.randn(32, 32, generator=generator).bfloat16()
            for name in (Q, K, V, EMBED)}


def _write(src: Path, shards: dict[str, list[str]], tensors) -> None:
    src.mkdir(parents=True)
    weight_map = {}
    for shard, names in shards.items():
        save_file({n: tensors[n].contiguous() for n in names}, str(src / shard),
                  metadata={"format": "pt"})
        weight_map.update({n: shard for n in names})
    if len(shards) > 1:
        (src / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": 0}, "weight_map": weight_map}))
    (src / "config.json").write_text(json.dumps({"model_type": "test"}))


def _export(tmp_path, monkeypatch, shards, tensors, out="out"):
    src = tmp_path / f"src-{out}"
    _write(src, shards, tensors)
    destination = tmp_path / out
    monkeypatch.setattr(
        "sys.argv",
        ["export_stock_compressed.py", str(src), str(destination),
         "--activations", "w4a16", "--device", "cpu", "--no-verify"])
    stock.main()
    return destination


def _stock_tensors(out: Path) -> dict[str, torch.Tensor]:
    index = out / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        shard = next(p.name for p in out.glob("*.safetensors"))
        with safe_open(str(out / shard), framework="pt") as handle:
            weight_map = {name: shard for name in handle.keys()}
    tensors = {}
    for name, shard in weight_map.items():
        with safe_open(str(out / shard), framework="pt") as handle:
            tensors[name] = handle.get_tensor(name)
    return tensors


def test_siblings_straddling_source_shards_export_the_single_shard_bytes(tmp_path, monkeypatch):
    """The #212 repro: q in part-1, k/v in part-2.  The export used to refuse
    at the first shard boundary ('fused group ... is incomplete on this
    shard'); it must instead wait, share one global over the completed group,
    and write bytes identical to the single-shard source's."""
    tensors = _tensors()
    split = _export(tmp_path, monkeypatch,
                    {"part-1.safetensors": [Q, EMBED], "part-2.safetensors": [K, V]},
                    tensors, out="split")
    single = _export(tmp_path, monkeypatch, {"model.safetensors": list(tensors)},
                     tensors, out="single")

    def comparable(tensor):
        return (tensor.view(torch.uint8)
                if tensor.dtype in (torch.float8_e4m3fn, torch.bfloat16) else tensor)

    split_tensors = _stock_tensors(split)
    single_tensors = _stock_tensors(single)
    assert set(split_tensors) == set(single_tensors)
    for name in sorted(single_tensors):
        assert torch.equal(comparable(split_tensors[name]),
                           comparable(single_tensors[name])), name

    manifest = json.loads((split / "tessera_stock_manifest.json").read_text())
    assert {Q, K, V} <= set(manifest["units"])
    assert all("shared_global_divisor" in manifest["units"][n] for n in (Q, K, V))


def test_the_index_points_at_the_shard_the_completed_group_was_written_to(tmp_path, monkeypatch):
    """A group completes when its LAST member arrives, so its tensors land in
    that output shard and the index must say so -- an index naming part-1
    would promise tensors part-1 does not hold."""
    out = _export(tmp_path, monkeypatch,
                  {"part-1.safetensors": [Q, EMBED], "part-2.safetensors": [K, V]},
                  _tensors(), out="indexed")
    weight_map = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    group_entries = {name: shard for name, shard in weight_map.items()
                     if name.startswith(LAYER)}
    assert group_entries, weight_map
    assert set(group_entries.values()) == {"part-2.safetensors"}
    with safe_open(str(out / "part-2.safetensors"), framework="pt") as handle:
        held = set(handle.keys())
    assert set(group_entries) <= held


def test_a_population_that_can_never_complete_is_refused_before_any_encode(tmp_path, monkeypatch):
    """v_proj missing from the source: refused up front, by name, not at a
    shard boundary after two encodes."""
    tensors = {name: value for name, value in _tensors().items() if name != V}
    with pytest.raises(SystemExit) as caught:
        _export(tmp_path, monkeypatch,
                {"part-1.safetensors": [Q, EMBED], "part-2.safetensors": [K]},
                tensors, out="incomplete")
    message = str(caught.value)
    assert "cannot share one weight_global_scale" in message and V in message
    assert not (tmp_path / "incomplete").exists()


def test_a_sibling_planned_out_of_the_nvfp4_kind_is_the_same_refusal(tmp_path, monkeypatch):
    """A BF16-planned sibling leaves the group unable to share one scale."""
    tensors = _tensors()
    src = tmp_path / "src-mixed"
    _write(src, {"model.safetensors": list(tensors)}, tensors)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({V: "BF16"}))
    monkeypatch.setattr(
        "sys.argv",
        ["export_stock_compressed.py", str(src), str(tmp_path / "mixed"),
         "--activations", "w4a16", "--device", "cpu", "--no-verify",
         "--plan-json", str(plan)])
    with pytest.raises(SystemExit) as caught:
        stock.main()
    message = str(caught.value)
    assert "cannot share one weight_global_scale" in message and V in message
