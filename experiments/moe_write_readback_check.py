"""The routed-MoE write half, run and then read back with the plugin's own readers.

CPU-only by construction.  It exercises the exporter's write path and the
reader's acceptance of what it wrote -- not the CUDA encoder and not the
fused-MoE kernel, which are what ``tests/test_export_moe_write.py``'s ``@cuda``
cases and the load probe cover.  Three legs, in order:

* **a refusal** -- one expert at a different shape, the one plan-time check
  that needs a whole checkpoint to provoke;
* **a control** -- the same checkpoint with no plan entry for the stack, which
  must leave the routed leaves at source precision and name the stack in
  ``ignore``;
* **the write, then the read-back** -- ``validate_tessera_moe_scheme`` (the
  function ``TesseraConfig`` calls), ``parse_tessera_expert_blob`` per
  container, ``moe_layout.unpack_moe_wires`` on padded rows, and
  ``prepare_tessera_moe_experts`` down to the stock per-channel FP8 stack.

The ``wire_stride`` claim is what this is really for: the blobs of one stack at
ONE shape and ONE rung differ in length, because the manifest's ``global_scale``
is an exact varint ratio whose width follows its value.  The run prints the
spread it measured, so the reason the sidecar declares a stride rather than a
byte count is a number here and not an argument.

    python experiments/moe_write_readback_check.py --work /home/rob/tmp/moe-check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import torch
from safetensors.torch import safe_open, save_file

HIDDEN, INTER, EXPERTS = 128, 64, 4
LAYER = "model.language_model.layers.1"
STACK = f"{LAYER}.mlp.experts"
GROUP_PROJECTIONS = {"w13": ("gate_proj", "up_proj"), "w2": ("down_proj",)}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _exporter():
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "export_tessera_serving", here / "export_tessera_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(generator, *, bad_geometry: bool) -> dict:
    def randn(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": randn(2 * HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.up_proj.weight": randn(2 * HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.down_proj.weight": randn(HIDDEN, 2 * HIDDEN),
        f"{LAYER}.mlp.shared_experts.gate_proj.weight": randn(INTER, HIDDEN),
        f"{LAYER}.mlp.shared_experts.up_proj.weight": randn(INTER, HIDDEN),
        f"{LAYER}.mlp.shared_experts.down_proj.weight": randn(HIDDEN, INTER),
        f"{LAYER}.mlp.gate.weight": randn(EXPERTS, HIDDEN),
        "lm_head.weight": randn(HIDDEN, HIDDEN),
        "model.embed_tokens.weight": randn(HIDDEN, HIDDEN),
    }
    for expert in range(EXPERTS):
        for projection in PROJECTIONS:
            shape = (HIDDEN, INTER) if projection == "down_proj" else (INTER, HIDDEN)
            tensors[f"{STACK}.{expert}.{projection}.weight"] = randn(*shape)
    if bad_geometry:
        tensors[f"{STACK}.3.gate_proj.weight"] = randn(2 * INTER, HIDDEN)
    return tensors


def _export(export, work: pathlib.Path, name: str, tensors: dict, plan):
    case = work / name
    source = case / "source"
    source.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(source / "model.safetensors"), metadata={"format": "pt"})
    (source / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": HIDDEN, "moe_intermediate_size": INTER,
                        "num_hidden_layers": 2, "n_routed_experts": EXPERTS}}))
    argv = ["export", str(source), str(case / "out"), "--grid", "E4M3", "--q256", "1024",
            "--device", "cpu", "--no-verify", "--passthrough-unrouted"]
    if plan is not None:
        (case / "plan.json").write_text(json.dumps(plan))
        argv += ["--plan-json", str(case / "plan.json")]
    saved, sys.argv = sys.argv, argv
    try:
        export.main()
    finally:
        sys.argv = saved
    return case / "out"


def _value_check(prepared, source: pathlib.Path) -> None:
    """The decoded tile IS the expert that went in, not merely well-formed bytes.

    Shapes and a clean parse cannot tell a correct tile from a transposed or
    interleaved one -- both are the right size.  A wrong orientation, a wrong
    gate/up split or a wrong expert row would put the relative error near
    ``sqrt(2)`` on independent weights, so the number below separates "the
    layout is right" from "the layout is plausible".
    """
    with safe_open(str(source / "model.safetensors"), framework="pt") as handle:
        original = {n: handle.get_tensor(n) for n in handle.keys() if n.startswith(STACK + ".")}

    def leg(label, tile, scale, keys):
        got = tile.to(torch.float32) * scale.to(torch.float32)
        want = torch.cat([original[k].to(torch.float32) for k in keys], dim=0)
        assert got.shape == want.shape, (got.shape, want.shape)
        return label, (got - want).norm().item() / want.norm().item()

    worst = 0.0
    for expert in range(EXPERTS):
        for label, rel in (
            leg(f"expert {expert} w13", prepared.w13_weight[expert],
                prepared.w13_weight_scale[expert],
                [f"{STACK}.{expert}.gate_proj.weight", f"{STACK}.{expert}.up_proj.weight"]),
            leg(f"expert {expert} w2", prepared.w2_weight[expert],
                prepared.w2_weight_scale[expert], [f"{STACK}.{expert}.down_proj.weight"]),
        ):
            worst = max(worst, rel)
    assert worst < 0.15, f"the decoded tile is not the source expert: rel_err {worst}"
    assert prepared.w13_weight[0].to(torch.float32).abs().sum() > 0, "expert 0 decoded to zeros"
    assert not torch.equal(prepared.w13_weight[0], prepared.w13_weight[1]), \
        "every expert decoded to the same tile"
    print(f"  value check: worst relative error against the source experts {worst:.6f} "
          "(a transposed, interleaved or misrouted tile sits near 1.41)")


def _read_back(out: pathlib.Path, source: pathlib.Path) -> None:
    from tessera.moe_layout import MoePacked, unpack_moe_wires
    from tessera.serving.moe_route import prepare_tessera_moe_experts
    from tessera.serving.scheme import (MOE_GROUPS, expert_role_declarations,
                                        parse_tessera_expert_blob, validate_tessera_moe_scheme)

    config = json.loads((out / "config.json").read_text())["quantization_config"]
    groups = [g for g in config["config_groups"].values() if g["targets"] == [STACK]]
    assert len(groups) == 1, sorted(config["config_groups"])
    assert groups[0]["format"] == "TESSERA", groups[0]["format"]
    assert STACK not in config["ignore"], config["ignore"]
    assert not [i for i in config["ignore"] if i.startswith(STACK + ".")], config["ignore"]
    scheme = groups[0]["scheme"]
    print("  config_groups: one TESSERA entry on the stack, and it is not in ignore")
    print("  scheme:", json.dumps(scheme, sort_keys=True))

    declared = validate_tessera_moe_scheme(scheme, STACK)
    print("  validate_tessera_moe_scheme accepted:",
          {k: declared[k] for k in ("family", "structure", "experts",
                                    "hidden_size", "intermediate_size")})

    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        names = sorted(handle.keys())
        wires = {n: handle.get_tensor(n) for n in names if n.startswith(STACK + ".")}
    want = sorted(f"{STACK}.{e}.{p}.wire" for e in range(EXPERTS) for p in PROJECTIONS)
    assert sorted(wires) == want, (sorted(wires), want)
    assert all(t.dtype == torch.uint8 and t.ndim == 1 for t in wires.values())
    assert not [n for n in names if n.startswith(STACK + ".") and n.endswith(".weight")]
    print(f"  tensors: {len(wires)} 1-D uint8 .wire under the stack, no source .weight left")

    blob = {n: bytes(t.numpy().tobytes()) for n, t in wires.items()}
    for group in MOE_GROUPS:
        spec = declared["groups"][group]
        roles = expert_role_declarations(spec)
        lengths = []
        for expert in range(EXPERTS):
            for role, projection in zip(roles, GROUP_PROJECTIONS[group]):
                raw = blob[f"{STACK}.{expert}.{projection}.wire"]
                lengths.append(len(raw))
                (name, unit), = parse_tessera_expert_blob(raw, role, STACK)
                assert name == projection, (name, projection)
                assert unit.manifest.geometry.columns == spec["columns"]
                assert unit.grid.name == spec["grid"] and unit.body.name == spec["body"]
        assert max(lengths) == spec["wire_stride"], (group, max(lengths), spec["wire_stride"])
        print(f"  {group}: {len(lengths)} containers parsed against the declared role; "
              f"lengths {min(lengths)}..{max(lengths)} (spread {max(lengths) - min(lengths)} "
              f"bytes at one shape and rung), wire_stride={spec['wire_stride']} == max")

    stride13 = declared["groups"]["w13"]["wire_stride"]
    stride2 = declared["groups"]["w2"]["wire_stride"]
    w13 = torch.zeros(EXPERTS, 2, stride13, dtype=torch.uint8)
    w13_len = torch.zeros(EXPERTS, 2, dtype=torch.long)
    w2 = torch.zeros(EXPERTS, stride2, dtype=torch.uint8)
    w2_len = torch.zeros(EXPERTS, dtype=torch.long)
    for expert in range(EXPERTS):
        for index, projection in enumerate(GROUP_PROJECTIONS["w13"]):
            row = wires[f"{STACK}.{expert}.{projection}.wire"]
            w13[expert, index, :row.numel()] = row
            w13_len[expert, index] = row.numel()
        row = wires[f"{STACK}.{expert}.down_proj.wire"]
        w2[expert, :row.numel()] = row
        w2_len[expert] = row.numel()
    back13, back2 = unpack_moe_wires(MoePacked(w13_wire=w13, w13_wire_len=w13_len,
                                               w2_wire=w2, w2_wire_len=w2_len))
    for expert in range(EXPERTS):
        assert back13[expert][0] == blob[f"{STACK}.{expert}.gate_proj.wire"]
        assert back13[expert][1] == blob[f"{STACK}.{expert}.up_proj.wire"]
        assert back2[expert] == blob[f"{STACK}.{expert}.down_proj.wire"]
    print("  unpack_moe_wires: padded rows + lengths round-trip to the written blobs, byte for byte")

    prepared = prepare_tessera_moe_experts(
        {"w13": [[back13[e][0], back13[e][1]] for e in range(EXPERTS)],
         "w2": [[back2[e]] for e in range(EXPERTS)]}, declared, STACK, device="cpu")
    print("  prepare_tessera_moe_experts(cpu): w13", list(prepared.w13_weight.shape),
          prepared.w13_weight.dtype, "w2", list(prepared.w2_weight.shape),
          "scales", list(prepared.w13_weight_scale.shape), list(prepared.w2_weight_scale.shape))
    _value_check(prepared, source)

    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    routed = manifest["routed_moe"]
    assert routed["disposition"] == "quantized", routed
    print("  manifest routed_moe:", {k: routed[k] for k in sorted(routed) if k != "reason"})
    record = manifest["modules"][STACK]
    print("  manifest stack record:", {k: record[k] for k in sorted(record) if k != "roles"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    export = _exporter()
    generator = torch.Generator().manual_seed(11)
    plan = {STACK: {"grid": "E4M3", "q256": 1024}}

    print("[1/3] one expert at a different shape is refused before any encode")
    try:
        _export(export, args.work, "geometry", _checkpoint(generator, bad_geometry=True), plan)
    except SystemExit as exc:
        print(f"  refused: {exc}")
    else:
        raise AssertionError("a stack whose experts disagree about shape was NOT refused")

    print("[2/3] control: no plan entry for the stack, so the stack is untouched")
    out = _export(export, args.work, "control", _checkpoint(generator, bad_geometry=False), None)
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    assert STACK in config["ignore"], sorted(config["ignore"])
    assert STACK not in {t for g in config["config_groups"].values() for t in g["targets"]}
    with safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        names = set(handle.keys())
    assert not [n for n in names if n.endswith(".wire") and n.startswith(STACK + ".")]
    assert f"{STACK}.0.gate_proj.weight" in names
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["routed_moe"]["disposition"] == "passed_through_bf16", manifest["routed_moe"]
    print("  stack named in ignore, no .wire tensors, source weights kept, "
          f"disposition={manifest['routed_moe']['disposition']}")

    print("[3/3] the write half, then the read-back")
    out = _export(export, args.work, "written", _checkpoint(generator, bad_geometry=False), plan)
    _read_back(out, out.parent / "source")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
