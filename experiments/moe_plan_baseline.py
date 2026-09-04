#!/usr/bin/env python
"""Plan baseline for the exporter's routed-MoE classification (#5).

``export_tessera_serving.py`` decides at PLAN time which tensors are dense
Linears and which belong to a routed-MoE layer.  A wrong answer there is
silent and expensive: an expert leaf planned as a dense Linear is encoded --
for hours on a real model -- into a checkpoint that declares modules the
runtime never builds, and a packed expert stack the planner cannot see leaves
the MoE module out of ``ignore``, so the plugin refuses the whole layer at
load.  Neither is visible in a diff; both are visible in the artifact.  So this
harness runs the exporter END TO END on the CPU, at toy shapes, over every
expert layout a checkpoint can carry, and digests what it wrote:

    python experiments/moe_plan_baseline.py before.json     # at HEAD
    ...apply the change...
    python experiments/moe_plan_baseline.py after.json
    python experiments/moe_plan_baseline.py --diff before.json after.json

Three kinds of row per case:

* ``classify/<case>`` -- what ``quantizable`` put in each bucket, as sorted
  name lists.  The cheapest view of the plan.
* ``export/<case>/{quantization_config,tensors,manifest}`` -- sha256 over the
  artifact the DEFAULT plan writes: the config groups and ignore list, every
  tensor's name and bytes, and the manifest minus its volatile fields.  A
  control case must not move at all; a case the change fixes moves in exactly
  the tensors that stop being dense.
* ``plan/<case>/<kind>`` -- what happens when a ``--plan-json`` names an
  expert leaf, a packed stack or the router.  The encoder is replaced by a
  sentinel, so the row reads either the refusal the exporter raised or
  ``ENCODE REACHED <tensor>`` -- the plan got as far as asking the encoder for
  that tensor, which is the mis-plan in one line.

Two rows are REAL, one per source layout, and they are digests rather than
name lists (``_real_buckets``): ``classify/Qwen3.8-Flash-Next`` is the
transformers-5 PACKED layout with no ``.weight`` suffix, and
``classify/GLM-5.3-Flash-4layer`` the UNPACKED per-expert 2-D leaves under
``model.language_model.layers.N.`` -- 2592 of them, the model #5 names.  Both
are classified by reading shapes; nothing is encoded or written for either.

The layouts, with the runtime file that builds each MoE module at
``<moe>.experts`` on the pinned build (``prismaquant/glm53-mia-sm121:
487ecf187``): GLM-5.3-Flash unpacked (``models/glm5next/nvidia/model.py:239``);
Mixtral unpacked ``block_sparse_moe.experts.E.w{1,3,2}`` (``mixtral.py:128,
:252``); LFM2-MoE unpacked ``feed_forward.experts.E.w{1,3,2}``
(``lfm2_moe.py:152,:301``); transformers-5 packed ``mlp.experts.gate_up_proj``
/ ``down_proj`` with no suffix, the Qwen3.8-Flash-Next layout on disk
(``qwen3_moe.py:208``, ``qwen3_next.py:179``); gpt-oss packed with biases
(``gpt_oss.py:221``).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "export_tessera_serving", ROOT / "experiments" / "export_tessera_serving.py")
export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export)

#: Real checkpoints on this box, classified read-only.  Two layouts, because
#: one of each is what the classification has to get right: Qwen3.8-Flash-Next
#: carries transformers-5 PACKED expert stacks with no ``.weight`` suffix, and
#: GLM-5.3-Flash-4layer -- the model #5 names, and the one an expert route will
#: first serve -- carries the UNPACKED per-expert 2-D leaves under a
#: ``model.language_model.layers.N.`` prefix.  A toy fixture can stage either
#: shape; only the checkpoint can say that this is the shape it has.
REAL = (Path("/mnt/shared/models/Qwen3.8-Flash-Next"),
        Path("/mnt/shared/models/GLM-5.3-Flash-4layer"))
HIDDEN, INTER = 64, 32          # every unit is a whole number of 32-row tuples, K % 16 == 0
PACKED_INTER = 16               # 2 * 16 != 64, so a packed stack's orientation is decidable
GRID, Q256 = "E4M3", 1024


def _seeded(case: str, name: str, *shape):
    seed = int.from_bytes(hashlib.sha256(f"{case}:{name}".encode()).digest()[:4], "little")
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _attention(t, case, P, layer):
    for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
        n = f"{P}.{layer}.self_attn.{role}.weight"
        t[n] = _seeded(case, n, HIDDEN, HIDDEN)


def _dense_mlp(t, case, P, layer, owner="mlp"):
    t[f"{P}.{layer}.{owner}.gate_proj.weight"] = _seeded(case, f"g{layer}", 3 * INTER, HIDDEN)
    t[f"{P}.{layer}.{owner}.up_proj.weight"] = _seeded(case, f"u{layer}", 3 * INTER, HIDDEN)
    t[f"{P}.{layer}.{owner}.down_proj.weight"] = _seeded(case, f"d{layer}", HIDDEN, 3 * INTER)


def _head(t, case):
    t["lm_head.weight"] = _seeded(case, "lm_head", HIDDEN, HIDDEN)
    t["model.embed_tokens.weight"] = _seeded(case, "embed", HIDDEN, HIDDEN)


def _unpacked(t, case, moe, experts, names=("gate_proj", "up_proj", "down_proj")):
    gate, up, down = names
    for e in range(experts):
        t[f"{moe}.experts.{e}.{gate}.weight"] = _seeded(case, f"e{e}g", INTER, HIDDEN)
        t[f"{moe}.experts.{e}.{up}.weight"] = _seeded(case, f"e{e}u", INTER, HIDDEN)
        t[f"{moe}.experts.{e}.{down}.weight"] = _seeded(case, f"e{e}d", HIDDEN, INTER)


def case_dense(case):
    t, P = {}, "model.layers"
    for layer in (0, 1):
        _attention(t, case, P, layer)
        _dense_mlp(t, case, P, layer)
    _head(t, case)
    cfg = {"architectures": ["Qwen3ForCausalLM"], "hidden_size": HIDDEN,
           "intermediate_size": 3 * INTER, "num_hidden_layers": 2}
    return t, cfg, {}


def case_glm_unpacked(case):
    """The 4-layer GLM checkpoint in miniature: the fixture of
    tests/test_export_moe_layouts.py, with a router wide enough to plan."""
    t, P, E = {}, "model.language_model.layers", 4
    _attention(t, case, P, 0)
    _dense_mlp(t, case, P, 0)
    t[f"{P}.0.self_attn.k_conv1d.weight"] = _seeded(case, "conv", 2 * HIDDEN, 1, 4)
    _attention(t, case, P, 1)
    moe = f"{P}.1.mlp"
    _unpacked(t, case, moe, E)
    _dense_mlp(t, case, P, 1, owner="mlp.shared_experts")
    t[f"{moe}.gate.weight"] = _seeded(case, "router", E, HIDDEN)
    t[f"{moe}.gate.e_score_correction_bias"] = torch.zeros(E)
    t["model.visual.blocks.0.mlp.gate_proj.weight"] = _seeded(case, "vis", HIDDEN, HIDDEN)
    _head(t, case)
    cfg = {"architectures": ["Glm5NextForConditionalGeneration"],
           "text_config": {"hidden_size": HIDDEN, "moe_intermediate_size": INTER,
                           "num_hidden_layers": 2, "n_routed_experts": E}}
    plans = {"expert": f"{moe}.experts.0.gate_proj.weight", "router": f"{moe}.gate.weight"}
    return t, cfg, plans


def case_mixtral_unpacked(case):
    """``block_sparse_moe.experts.E.w{1,3,2}`` and a router of 32 rows -- wide
    enough that ``rows % 32 == 0`` lets it into an E4M3 plan."""
    t, P, E = {}, "model.layers", 32
    _attention(t, case, P, 0)
    _dense_mlp(t, case, P, 0)
    _attention(t, case, P, 1)
    moe = f"{P}.1.block_sparse_moe"
    _unpacked(t, case, moe, E, names=("w1", "w3", "w2"))
    t[f"{moe}.gate.weight"] = _seeded(case, "router", E, HIDDEN)
    _head(t, case)
    cfg = {"architectures": ["MixtralForCausalLM"], "hidden_size": HIDDEN,
           "intermediate_size": INTER, "num_local_experts": E, "num_hidden_layers": 2}
    plans = {"expert": f"{moe}.experts.0.w1.weight", "router": f"{moe}.gate.weight"}
    return t, cfg, plans


def case_lfm2_unpacked(case):
    t, P, E = {}, "model.layers", 32
    _attention(t, case, P, 0)
    _dense_mlp(t, case, P, 0, owner="feed_forward")
    _attention(t, case, P, 1)
    moe = f"{P}.1.feed_forward"
    _unpacked(t, case, moe, E, names=("w1", "w3", "w2"))
    t[f"{moe}.gate.weight"] = _seeded(case, "router", E, HIDDEN)
    t[f"{moe}.expert_bias"] = torch.zeros(E)
    _head(t, case)
    cfg = {"architectures": ["Lfm2MoeForCausalLM"], "hidden_size": HIDDEN,
           "moe_intermediate_size": INTER, "num_experts": E, "num_hidden_layers": 2}
    plans = {"expert": f"{moe}.experts.0.w1.weight", "router": f"{moe}.gate.weight"}
    return t, cfg, plans


def _packed_qwen(t, case, moe, E, suffix):
    t[f"{moe}.experts.gate_up_proj{suffix}"] = _seeded(case, "gu", E, 2 * PACKED_INTER, HIDDEN)
    t[f"{moe}.experts.down_proj{suffix}"] = _seeded(case, "dn", E, HIDDEN, PACKED_INTER)


def case_qwen38_packed_nosuffix(case, suffix=""):
    """The Qwen3.8-Flash-Next layout on disk: transformers-5 packed experts as
    bare parameters (no ``.weight``), a router, a shared expert and its gate."""
    t, P, E = {}, "model.language_model.layers", 4
    _attention(t, case, P, 0)
    _dense_mlp(t, case, P, 0)
    _attention(t, case, P, 1)
    moe = f"{P}.1.mlp"
    _packed_qwen(t, case, moe, E, suffix)
    t[f"{moe}.gate.weight"] = _seeded(case, "router", E, HIDDEN)
    _dense_mlp(t, case, P, 1, owner="mlp.shared_expert")
    t[f"{moe}.shared_expert_gate.weight"] = _seeded(case, "seg", 1, HIDDEN)
    _head(t, case)
    cfg = {"architectures": ["Qwen3NextForCausalLM"],
           "text_config": {"hidden_size": HIDDEN, "moe_intermediate_size": PACKED_INTER,
                           "num_experts": E, "num_hidden_layers": 2}}
    plans = {"packed": f"{moe}.experts.gate_up_proj{suffix}", "router": f"{moe}.gate.weight"}
    return t, cfg, plans


def case_qwen38_packed_weight_suffix(case):
    """The same stack with the ``.weight`` suffix the old tests assumed."""
    return case_qwen38_packed_nosuffix(case, suffix=".weight")


def case_gptoss_packed(case):
    t, P, E = {}, "model.layers", 4
    _attention(t, case, P, 0)
    _attention(t, case, P, 1)
    for layer in (0, 1):
        moe = f"{P}.{layer}.mlp"
        t[f"{moe}.experts.gate_up_proj"] = _seeded(case, f"gu{layer}", E, HIDDEN, 2 * PACKED_INTER)
        t[f"{moe}.experts.gate_up_proj_bias"] = torch.zeros(E, 2 * PACKED_INTER)
        t[f"{moe}.experts.down_proj"] = _seeded(case, f"dn{layer}", E, PACKED_INTER, HIDDEN)
        t[f"{moe}.experts.down_proj_bias"] = torch.zeros(E, HIDDEN)
        t[f"{moe}.router.weight"] = _seeded(case, f"router{layer}", E, HIDDEN)
        t[f"{moe}.router.bias"] = torch.zeros(E)
    _head(t, case)
    cfg = {"architectures": ["GptOssForCausalLM"], "hidden_size": HIDDEN,
           "intermediate_size": PACKED_INTER, "num_local_experts": E, "num_hidden_layers": 2}
    plans = {"packed": f"{P}.1.mlp.experts.gate_up_proj", "router": f"{P}.1.mlp.router.weight"}
    return t, cfg, plans


CASES = {
    "dense": case_dense,
    "glm_unpacked": case_glm_unpacked,
    "mixtral_unpacked": case_mixtral_unpacked,
    "lfm2_unpacked": case_lfm2_unpacked,
    "qwen38_packed_nosuffix": case_qwen38_packed_nosuffix,
    "qwen38_packed_weight_suffix": case_qwen38_packed_weight_suffix,
    "gptoss_packed": case_gptoss_packed,
}


def _write(root: Path, case: str):
    tensors, cfg, plans = CASES[case](case)
    src = root / case / "src"
    src.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps(cfg))
    return src, plans


def _buckets(result) -> dict:
    """``quantizable``'s answer as sorted name lists, whatever its return type."""
    fields = getattr(result, "_fields", None)
    if fields:
        out = {}
        for f in fields:
            if f == "shards":
                continue
            value = getattr(result, f)
            out[f] = sorted(value) if isinstance(value, (dict, set, list, tuple)) else value
        return out
    _shards, shapes, packed, routed = result
    return {"shapes": sorted(shapes), "packed": sorted(packed), "routed": sorted(routed)}


def _real_buckets(result, src: Path) -> dict:
    """``_buckets`` as a DIGEST, for a checkpoint with thousands of tensors.

    A real MoE layer contributes 864 routed names per layer, and a baseline
    file that inlines them is unreadable in a diff for no gain: what a
    classification change moves is which bucket a name is in, and a per-bucket
    count plus a sha over the sorted names moves whenever that does.  The first
    and last name of each bucket are kept because they are what says *which*
    layout was classified when the count is right and the sha is not.

    A packed stack's ORIENTATION is recorded beside the names, because it is
    the other thing the planner decides about a real checkpoint and the one
    that transposes every expert in silence when it is wrong.  It is read the
    way the exporter reads it -- off ``config.json`` -- and a refusal is
    recorded as the refusal, since "this checkpoint cannot be oriented" is an
    answer about it and not a harness failure.
    """
    shapes = getattr(result, "expert_shapes", None)
    if shapes is None:
        shapes = result[2]
    config = json.loads((src / "config.json").read_text())
    out = {}
    for bucket, names in _buckets(result).items():
        names = sorted(names)
        out[bucket] = {"count": len(names),
                       "sha256": _sha("\n".join(names).encode()),
                       "first": names[0] if names else None,
                       "last": names[-1] if names else None}
    orientations = {}
    for name in sorted(shapes):
        try:
            orientations[name] = export.packed_expert_orientation(name, shapes[name], config)
        except SystemExit as exc:                       # the refusal IS the record
            orientations[name] = f"REFUSED: {exc}"
    if orientations:
        first = sorted(orientations)[0]
        out["packed_orientation"] = {"distinct": sorted(set(orientations.values())),
                                     "example": {first: orientations[first]}}
    return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _run_main(argv: list[str]):
    """Run the exporter in-process; return (exit message or None, stdout)."""
    out = io.StringIO()
    old = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(out):
            export.main()
        return None, out.getvalue()
    except SystemExit as exc:
        return str(exc), out.getvalue()
    finally:
        sys.argv = old


def _digest_export(outdir: Path) -> dict:
    cfg = json.loads((outdir / "config.json").read_text())["quantization_config"]
    tensors = {}
    for shard in sorted(outdir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as h:
            for k in sorted(h.keys()):
                t = h.get_tensor(k).contiguous()
                tensors[k] = _sha(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.uint8
                                  else t.numpy().tobytes())
    manifest = json.loads((outdir / "tessera_serving_manifest.json").read_text())
    for volatile in ("written", "git", "source", "plan_json", "stock_twin", "input_scales_from"):
        manifest.pop(volatile, None)
    return {
        "quantization_config": _sha(_canonical(cfg)),
        "declared": sorted(t for g in cfg["config_groups"].values() for t in g["targets"]),
        "ignore": sorted(cfg["ignore"]),
        "tensors": _sha(_canonical(tensors)),
        "tensor_names": sorted(tensors),
        "manifest": _sha(_canonical(manifest)),
    }


class _EncodeReached(Exception):
    pass


def _plan_row(src: Path, outdir: Path, tensor: str) -> str:
    """Name one tensor in a plan and report the refusal, or the encoder call."""
    plan = outdir.parent / f"plan_{_sha(tensor.encode())[:8]}.json"
    plan.write_text(json.dumps({tensor: {"grid": GRID, "q256": Q256}}))
    real = export.encode_linear_planes

    def sentinel(weight, *, name, **_kw):
        raise _EncodeReached(name)

    export.encode_linear_planes = sentinel
    try:
        try:
            message, _stdout = _run_main(["export", str(src), str(outdir), "--grid", GRID,
                                          "--q256", str(Q256), "--plan-json", str(plan),
                                          "--device", "cpu"])
        except _EncodeReached as reached:
            return f"ENCODE REACHED {reached}"
    finally:
        export.encode_linear_planes = real
        shutil.rmtree(outdir, ignore_errors=True)
    return f"REFUSED: {message}" if message else "EXPORTED WITHOUT ENCODING ANYTHING"


def run(root: Path, cases: list[str], *, export_rows: bool) -> dict:
    rows: dict = {}
    for case in cases:
        src, plans = _write(root, case)
        rows[f"classify/{case}"] = _buckets(export.quantizable(src))
        for kind, tensor in plans.items():
            rows[f"plan/{case}/{kind}"] = _plan_row(src, root / case / f"plan_{kind}", tensor)
        if export_rows:
            outdir = root / case / "out"
            message, stdout = _run_main(["export", str(src), str(outdir), "--grid", GRID,
                                         "--q256", str(Q256), "--device", "cpu", "--no-verify"])
            if message:
                rows[f"export/{case}"] = f"REFUSED: {message}"
            else:
                rows[f"export/{case}"] = _digest_export(outdir)
                rows[f"export/{case}/stdout_moe_lines"] = [
                    line.strip() for line in stdout.splitlines()
                    if "expert" in line or "router" in line or "MoE" in line]
            print(f"  {case}: {'refused' if message else 'exported'}", flush=True)
    for real in REAL:
        if not real.exists():
            continue
        rows[f"classify/{real.name}"] = _real_buckets(export.quantizable(real), real)
        print(f"  classified the real {real.name} (read-only)", flush=True)
    return rows


def _flatten(rows: dict, prefix="") -> dict:
    flat = {}
    for k, v in rows.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "/"))
        else:
            flat[key] = v
    return flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", type=Path)
    ap.add_argument("--diff", nargs=2, type=Path)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--no-export", action="store_true",
                    help="classification and plan rows only (no CPU encode)")
    ap.add_argument("--work", type=Path, default=None,
                    help="where the toy checkpoints and exports are written "
                         "(default: a fresh directory under $TMPDIR)")
    args = ap.parse_args()
    if args.diff:
        before, after = (_flatten(json.loads(p.read_text())) for p in args.diff)
        keys = sorted(set(before) | set(after))
        changed = [k for k in keys if before.get(k) != after.get(k)]
        for k in changed:
            b, a = before.get(k, "<absent>"), after.get(k, "<absent>")
            print(f"CHANGED {k}\n  before: {str(b)[:400]}\n  after:  {str(a)[:400]}")
        print(f"{len(changed)} changed of {len(keys)}")
        return 1 if changed else 0
    if args.out is None:
        ap.error("an output path, or --diff before after")
    if "TMPDIR" not in os.environ:
        raise SystemExit("set TMPDIR (never /tmp; see CLAUDE.md)")
    root = args.work or Path(tempfile.mkdtemp(prefix="moe_plan_baseline_"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"work: {root}", flush=True)
    rows = run(root, args.cases.split(","), export_rows=not args.no_export)
    args.out.write_text(json.dumps(rows, indent=1, sort_keys=True))
    print(f"wrote {args.out}: {len(_flatten(rows))} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
