"""Device crosscheck: the dense A8/A16 routes at PREFILL shapes, TP1 and TP2.

WHAT THIS DRIVES.  The production load path of the two dense window routes --
``lane.build_tessera_method`` -> ``create_weights`` -> ``process_weights_after_
loading`` -> ``apply`` -- on a real container and a real rank geometry, and it
compares the served output against the RETAINED reference: the materialising
preparation (``prepare_tessera_fp8_module`` / ``prepare_tessera_bf16_module``,
which decodes through ``tessera.decode.materialize_fp8`` /
``materialize_bf16``) whose tile and row scale are multiplied by a stock
matmul.  That reference is the arm the native lane replaced, so agreement is
what makes the replacement a substitution rather than a second renderer.

WHY PREFILL IS THE SHAPE.  The dense routes serve one packed GEMM at every M,
but the arm that was retired for this regime was the prefill fallback: past
``GEMV_MAX_M`` (8) the streamed lane decoded a whole ``[rows, columns]`` tile
per forward and called ``torch._scaled_mm`` / ``torch.mm``.  So the M set walks
the boundaries rather than the middle: the empty batch, one row, the GEMV
limit and the row past it (8, 9), the ``tl.dot`` tile (17), a ``block_m``
multiple (64), a shape that divides nothing (129) and a prefill-sized 512.

WHAT THE DISPATCH PROBE IS FOR.  Numerical agreement alone cannot tell "the
native GEMM ran" from "a materialiser ran and agreed", and before the
substitution the streamed route DID materialise.  So around every ``apply``
the probe counts the packed bundle's launches (``PreparedWindowGemm.__call__``,
at least one per role) and makes every materialising entry point raise if it is
entered: ``PreparedWindow.decode``, ``materialize_fp8``/``materialize_bf16``,
``torch._scaled_mm`` and ``torch.mm``.  An arm fails if anything in the second
set is called, whatever its numbers.

SHAPE VALIDITY.  Two refusal arms hand a real wire to a declaration that does
not match it (the other family, and a wrong rung).  Both must refuse by name at
the load, in this build: an unsupported case fails closed rather than being
served through an expansion nobody declared.

RUN IT.  vLLM work is exempt from PrismaBuild, and this drives the real vLLM
route on a device, so it runs directly in the pinned image (the image is the
one the mixed-fixture serve and the A16 TP1 run used):

  docker run --rm --gpus all --user 1000:1000 -e HOME=/tmp \
    -e PYTHONPATH=/work/src:/work/tests \
    -v <source snapshot>:/work:ro -v /mnt/shared:/mnt/shared:ro \
    --entrypoint python3 <image a5424378…> \
    /work/tests/native_dense_prefill_cross_check.py

``--preflight`` needs no device and no vLLM: it resolves the fixtures, checks
each declaration with ``validate_tessera_scheme`` and builds every rank's plan,
which is the plumbing a device run would otherwise fail on first.  That mode
is CPU work and goes through PrismaBuild like every other CPU action.

BOUNDS.  The fixtures are two small dense modules (the widest is 4096x4096),
so the arms are bounded by construction; ``--m`` caps the batch and the whole
run is one bounded process.  Tolerances are the chosen screens the route tests
use, not composed bounds, and they are printed with every arm.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

#: The two dense fixtures, by family.  Each names a config group and the
#: container tensor beside it; the paths are box artifacts with an override, so
#: a box that keeps them elsewhere says so instead of skipping silently.
FIXTURES = {
    "TESSERA_FP8": {
        "env": "TESSERA_A8_DENSE_FIXTURE",
        "default": "/mnt/shared/tessera-runs/derivatives/"
                   "mixedA4A8A16-layers0-4-20260916",
        "modules": (
            {"group": "tessera_model_language_model_layers_3_mlp_shared_experts_gate_up_proj",
             "tensor": "model.language_model.layers.3.mlp.shared_experts.gate_up_proj.wire_bytes",
             "parallel": "column"},
            {"group": "tessera_model_language_model_layers_3_mlp_shared_experts_down_proj",
             "tensor": "model.language_model.layers.3.mlp.shared_experts.down_proj.wire_bytes",
             "parallel": "row"},
        ),
    },
    "TESSERA_BF16": {
        "env": "TESSERA_A16_DENSE_FIXTURE",
        "default": "/mnt/shared/tessera-runs/bf16/qwen0.6b-bf16-r7-plugin",
        "modules": (
            {"group": "tessera_model_layers_0_mlp_gate_up_proj",
             "tensor": "model.layers.0.mlp.gate_up_proj.wire_bytes",
             "parallel": "column"},
            {"group": "tessera_model_layers_0_mlp_down_proj",
             "tensor": "model.layers.0.mlp.down_proj.wire_bytes",
             "parallel": "row"},
        ),
    },
}

#: The M set the prefill arms walk.  See the module docstring for the shape
#: each value is at.
DEFAULT_M = (0, 1, 8, 9, 17, 64, 129, 512)

#: The chosen screens the route tests use, per family, against the reference
#: product's own magnitude: the FP8 route's served error screen, and the
#: module screen ``test_serving_native_window`` holds the BF16 lane to.  They
#: are SCREENS, not composed bounds, and every arm prints which one it used.
BF16_SCREEN = {"fixed": 5.0e-3, "of_max_abs": 1.0e-2}
FP8_SCREEN = {"fixed": 0.0, "of_max_abs": 0.0, "relative": 8.0e-3}


def _failure(message: str):
    raise SystemExit(f"native_dense_prefill_cross_check: {message}")


def _fixture_root(spec) -> Path:
    import os

    root = Path(os.environ.get(spec["env"]) or spec["default"])
    if not (root / "config.json").is_file():
        _failure(
            f"{spec['env']} / its documented default resolves to {root}, which has no "
            "config.json; the dense fixture this crosscheck prices against is missing"
        )
    return root


def _read_wire(root: Path, tensor: str) -> bytes:
    """The container bytes for ``tensor``, from an index or a single shard."""
    from safetensors import safe_open

    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        matches = [key for key in weight_map if key == tensor or key.endswith(tensor)]
        if len(matches) != 1:
            _failure(f"{tensor} matches {len(matches)} entries in {index}")
        key = matches[0]
        with safe_open(str(root / weight_map[key]), framework="pt") as handle:
            return bytes(handle.get_tensor(key).detach().cpu().numpy().tobytes())
    shard = root / "model.safetensors"
    if not shard.is_file():
        _failure(f"{root} carries neither {index.name} nor {shard.name}")
    with safe_open(str(shard), framework="pt") as handle:
        matches = [key for key in handle.keys() if key == tensor or key.endswith(tensor)]
        if len(matches) != 1:
            _failure(f"{tensor} matches {len(matches)} tensors in {shard}")
        key = matches[0]
        return bytes(handle.get_tensor(key).detach().cpu().numpy().tobytes())


def _declared(root: Path, group: str, blob: bytes, prefix: str):
    """The config group's scheme, validated, with the blob's own byte count."""
    from tessera.serving.scheme import validate_tessera_scheme

    config = json.loads((root / "config.json").read_text())
    groups = config["quantization_config"]["config_groups"]
    if group not in groups:
        _failure(f"{group} is not a config group of {root}")
    scheme = dict(groups[group]["scheme"])
    if int(scheme.get("wire_bytes", len(blob))) != len(blob):
        _failure(
            f"{group}: the sidecar declares {scheme.get('wire_bytes')} bytes and the "
            f"container carries {len(blob)}"
        )
    scheme["wire_bytes"] = len(blob)
    declared = validate_tessera_scheme(scheme, prefix)
    return scheme, declared


def _plan(declared, parallel: str, tp_rank: int, tp_size: int):
    """This rank's ``ShardPlan``, in the numbering ``create_weights`` uses."""
    from tessera.serving.sharding import plan_shard

    roles = [(str(name), int(rows)) for name, rows in declared["roles"]]
    columns = int(declared["columns"])
    rows = sum(r for _, r in roles)
    if tp_size == 1:
        return plan_shard("crosscheck", roles=roles, columns=columns,
                          out_partitions=[r for _, r in roles], in_size=columns,
                          tp_rank=0, tp_size=1, input_size=columns, output_size=rows)
    if parallel == "column":
        # Column-parallel: the OUTPUT row axis is cut, the input width is whole.
        return plan_shard("crosscheck", roles=roles, columns=columns,
                          out_partitions=[r // tp_size for _, r in roles],
                          in_size=columns, tp_rank=tp_rank, tp_size=tp_size,
                          input_size=columns, output_size=rows)
    # Row-parallel: the INPUT width is cut, the output row axis is whole.
    return plan_shard("crosscheck", roles=roles, columns=columns,
                      out_partitions=[r for _, r in roles],
                      in_size=columns // tp_size, tp_rank=tp_rank, tp_size=tp_size,
                      input_size=columns, output_size=rows)


def _create_weight_args(declared, parallel: str, tp_size: int):
    """The four numbers a vLLM ``LinearBase`` hands ``create_weights``."""
    roles = [(str(name), int(rows)) for name, rows in declared["roles"]]
    columns = int(declared["columns"])
    rows = sum(r for _, r in roles)
    if tp_size == 1:
        return dict(input_size_per_partition=columns,
                    output_partition_sizes=[r for _, r in roles],
                    input_size=columns, output_size=rows)
    if parallel == "column":
        return dict(input_size_per_partition=columns,
                    output_partition_sizes=[r // tp_size for _, r in roles],
                    input_size=columns, output_size=rows)
    return dict(input_size_per_partition=columns // tp_size,
                output_partition_sizes=[r for _, r in roles],
                input_size=columns, output_size=rows)


def _layer(tp_rank: int, tp_size: int):
    """A ``LinearBase`` stand-in: the rank's own TP coordinates and nothing else.

    The same seam ``tests/test_serving_fp8_route.py`` drives the route through,
    and the same one the window-MoE stock crosscheck uses: the route reads
    ``tp_rank``/``tp_size``, registers one parameter and two buffers, and
    deletes the parameter when it is spent.
    """
    import torch

    class _Layer(torch.nn.Module):
        def __init__(self, rank: int, size: int):
            super().__init__()
            self.tp_rank = int(rank)
            self.tp_size = int(size)

    return _Layer(tp_rank, tp_size)


@contextlib.contextmanager
def _dispatch_probe():
    """Count the packed lane; make every materialising entry point raise."""
    import torch

    import tessera.decode as decode
    from tessera import window_gemm as wg
    from tessera.serving import window as reference_window

    record = {"native_calls": 0, "materialiser_calls": []}
    patches = []

    def _patch(owner, name, replacement):
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        patches.append((owner, name, original))

    original_call = wg.PreparedWindowGemm.__call__

    def counted(self, *args, **kwargs):
        record["native_calls"] += 1
        return original_call(self, *args, **kwargs)

    def refuse(what):
        def _refuse(*_args, **_kwargs):
            record["materialiser_calls"].append(what)
            raise AssertionError(f"the route entered the materialising path: {what}")
        return _refuse

    try:
        _patch(wg.PreparedWindowGemm, "__call__", counted)
        _patch(reference_window.PreparedWindow, "decode", refuse("PreparedWindow.decode"))
        _patch(decode, "materialize_fp8", refuse("materialize_fp8"))
        _patch(decode, "materialize_bf16", refuse("materialize_bf16"))
        if hasattr(torch, "_scaled_mm"):
            _patch(torch, "_scaled_mm", refuse("torch._scaled_mm"))
        _patch(torch, "mm", refuse("torch.mm"))
        yield record
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def _reference_product(family: str, blob: bytes, scheme, plan, x):
    """The retired arm's product: the reference decoder's tile and row scale."""
    import torch

    from tessera.serving import bf16_route, fp8_route
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    from tessera.serving.sharding import shard_parsed_roles

    parsed = parse_tessera_blob_for_scheme(blob, scheme, "crosscheck", device="cuda")
    roles = shard_parsed_roles(parsed, plan)
    if family == "TESSERA_FP8":
        from tessera.serving import native_ops

        module = fp8_route.prepare_tessera_fp8_module(roles, device="cuda")
        # The FP8 reference's ``decode`` is the BYTE tile (its docstring's
        # claim is byte identity with ``materialize_stock``), so the values are
        # the E4M3 view of it, exactly as the route's own ``_scaled_mm`` takes.
        tile = module.decode().view(torch.float8_e4m3fn).to(torch.float32)
        a_q, a_scale = native_ops.native_fp8_quant(x.contiguous())
        left = a_q.to(torch.float32) * a_scale.to(torch.float32)
    else:
        module = bf16_route.prepare_tessera_bf16_module(roles, device="cuda")
        tile = module.decode().to(torch.float32)
        left = x.to(torch.float32)
    scale = module.row_scale().to(torch.float32).reshape(-1, 1)
    return (left @ (tile * scale).t()).to(torch.bfloat16)


def _drive_route(family: str, scheme, declared, blob: bytes, parallel: str,
                 tp_rank: int, tp_size: int, x, mode: str):
    """The production load path, then one forward, under the dispatch probe."""
    import torch

    from tessera.serving import lane as serving_lane

    serving_lane.reset_for_tests()
    prefix = f"crosscheck.{family}.tp{tp_size}.rank{tp_rank}"
    method = serving_lane.build_tessera_method(scheme, prefix, mode)
    layer = _layer(tp_rank, tp_size)
    method.create_weights(layer, params_dtype=torch.bfloat16,
                          **_create_weight_args(declared, parallel, tp_size))
    layer.wire_bytes.data = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    method.process_weights_after_loading(layer)
    with _dispatch_probe() as probe:
        got = method.apply(layer, x)
    return got, layer, probe


def _arm(family: str, module: dict, declared, scheme, blob: bytes, parallel: str,
         tp_rank: int, tp_size: int, m: int, mode: str, report: dict):
    import torch

    plan = _plan(declared, parallel, tp_rank, tp_size)
    columns = int(declared["columns"])
    x_full = torch.randn(max(m, 1), columns, dtype=torch.bfloat16, device="cuda",
                         generator=torch.Generator(device="cuda").manual_seed(1000 + m))
    if parallel == "row" and tp_size > 1:
        role = next(iter(declared["roles"]))[0]
        lo, hi = plan.role(str(role)).lo, plan.role(str(role)).hi
        x = x_full[:, lo:hi].contiguous()
    else:
        x = x_full
    x = x[:m].contiguous()
    want = _reference_product(family, blob, scheme, plan, x)
    got, layer, probe = _drive_route(family, scheme, declared, blob, parallel,
                                     tp_rank, tp_size, x, mode)
    entry = {
        "family": family, "group": module["group"], "mode": mode,
        "tp_rank": tp_rank, "tp_size": tp_size, "parallel": parallel, "m": m,
        "shape": [int(v) for v in got.shape],
        "native_calls": probe["native_calls"],
        "materialiser_calls": probe["materialiser_calls"],
    }
    if m == 0:
        entry["passed"] = (
            tuple(got.shape) == tuple(want.shape)
            and not probe["materialiser_calls"]
            and probe["native_calls"] > 0
        )
        report["arms"].append(entry)
        return entry["passed"]
    diff = (got.float() - want.float()).abs()
    magnitude = want.float().abs().max().clamp_min(1e-9)
    screen = (FP8_SCREEN if family == "TESSERA_FP8" else BF16_SCREEN)
    limit = screen["fixed"] + screen["of_max_abs"] * float(magnitude)
    if screen.get("relative"):
        limit = max(limit, screen["relative"] * float(magnitude))
    error = float(diff.max())
    entry.update({
        "max_abs": error, "max_over_mag": float(error / float(magnitude)),
        "tolerance": limit, "tolerance_derivation": "chosen screen, not a composed bound",
    })
    # The load path's own claims, read off the prepared layer: the weights stay
    # packed, no decoded tile is registered, and the record names the native op.
    native = getattr(layer, "tessera_native", None)
    packed_ok = native is not None and native.packed_bytes() < (
        int(declared["rows"]) * int(declared["columns"]) * 2)
    absent = [name for name in ("weight_fp8", "weight_bf16", "tessera_prepared",
                               "tessera_gemv")
              if hasattr(layer, name)]
    entry.update({
        "packed_bytes": None if native is None else int(native.packed_bytes()),
        "materialising_attributes": absent,
        "decoder": getattr(layer, "tessera_decoder", None),
    })
    passed = (
        error <= limit
        and not probe["materialiser_calls"]
        and probe["native_calls"] > 0
        and packed_ok
        and not absent
    )
    entry["passed"] = passed
    report["arms"].append(entry)
    return passed


def _refusal_arms(family: str, module: dict, declared, blob: bytes, report: dict):
    """A real wire under a declaration that does not match it must refuse."""
    from tessera.serving.scheme import parse_compact_blob_for_scheme

    other = "TESSERA_BF16" if family == "TESSERA_FP8" else "TESSERA_FP8"
    cases = {
        "family": dict(declared, family=other,
                       grid=("BF16" if other == "TESSERA_BF16" else "E4M3"),
                       q256=(1792 if other == "TESSERA_BF16" else 1024)),
        "rung": dict(declared, q256=(896 if family == "TESSERA_FP8" else 1024)),
    }
    ok = True
    for label, wrong in cases.items():
        entry = {"family": family, "group": module["group"], "refusal": label}
        try:
            parse_compact_blob_for_scheme(blob, wrong, "crosscheck", device="cpu")
        except ValueError as exc:
            message = str(exc)
            entry["refused"] = True
            entry["message"] = message[:400]
            # A refusal for the RIGHT reason: the reader compares the wire's
            # own sidecar facts against the declaration and says which differ.
            # A bare "invalid scheme" would pass an existence check and still
            # leave the mismatch unstated.
            entry["named_the_mismatch"] = "sidecar scheme declares" in message
            ok = ok and entry["named_the_mismatch"]
        else:
            entry["refused"] = False
            entry["message"] = "accepted a wire the declaration does not match"
            ok = False
        report["refusals"].append(entry)
    return ok


def _preflight(selected) -> dict:
    """Resolve and check everything a device run would fail on first.

    The refusal arms are CPU work -- the reader compares the wire's own sidecar
    against the declaration and needs no device -- so they run here as well:
    this mode is the receipt for "an unsupported declaration fails closed by
    name", and the device run is what remains for the numerics.
    """
    out = {"fixtures": [], "refusals": [], "ok": True}
    for family, spec in selected.items():
        root = _fixture_root(spec)
        for module in spec["modules"]:
            blob = _read_wire(root, module["tensor"])
            _, declared = _declared(root, module["group"], blob, module["group"])
            if declared["family"] != family:
                _failure(f"{module['group']} declares {declared['family']}, not {family}")
            out["ok"] = _refusal_arms(family, module, declared, blob, out) and out["ok"]
            plans = []
            for parallel, tp_rank, tp_size in (
                    (module["parallel"], 0, 1),
                    (module["parallel"], 0, 2),
                    (module["parallel"], 1, 2)):
                plan = _plan(declared, parallel, tp_rank, tp_size)
                plans.append({"tp_rank": tp_rank, "tp_size": tp_size,
                              "shard_rows": int(plan.shard_rows),
                              "shard_columns": int(plan.shard_columns)})
            out["fixtures"].append({
                "family": family, "root": str(root), "group": module["group"],
                "tensor": module["tensor"], "bytes": len(blob),
                "rows": int(declared["rows"]), "columns": int(declared["columns"]),
                "q256": int(declared["q256"]), "parallel": module["parallel"],
                "plans": plans,
            })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", action="append", choices=sorted(FIXTURES),
                        help="restrict to a family; repeatable (default: both)")
    parser.add_argument("--m", type=int, action="append",
                        help=f"batch sizes (default: {' '.join(map(str, DEFAULT_M))})")
    parser.add_argument("--mode", default="streamed", choices=("streamed", "resident"),
                        help="the declared residency (default: streamed)")
    parser.add_argument("--tp", type=int, default=2, choices=(1, 2),
                        help="the widest world size to drive (default: 2)")
    parser.add_argument("--preflight", action="store_true",
                        help="resolve the fixtures and build every plan; no device needed")
    parser.add_argument("--json", help="write the report here as well as to stdout")
    args = parser.parse_args(argv)

    selected = {family: spec for family, spec in FIXTURES.items()
                if not args.family or family in args.family}
    if args.preflight:
        report = _preflight(selected)
    else:
        import torch

        if not torch.cuda.is_available():
            _failure("no CUDA device; this crosscheck drives the route on a device")
        report = {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "mode": args.mode,
            "arms": [],
            "refusals": [],
        }
        try:
            import vllm

            report["vllm"] = getattr(vllm, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            report["vllm"] = f"unavailable: {type(exc).__name__}: {exc}"
        ok = True
        for family, spec in selected.items():
            root = _fixture_root(spec)
            for module in spec["modules"]:
                blob = _read_wire(root, module["tensor"])
                scheme, declared = _declared(root, module["group"], blob, module["group"])
                ok = _refusal_arms(family, module, declared, blob, report) and ok
                for tp_size in ((1,) if args.tp == 1 else (1, 2)):
                    for tp_rank in range(tp_size):
                        for m in (args.m or DEFAULT_M):
                            ok = _arm(family, module, declared, scheme, blob,
                                      module["parallel"], tp_rank, tp_size, m,
                                      args.mode, report) and ok
        report["all_arms_passed"] = bool(ok)
    report["preflight"] = bool(args.preflight)
    text = json.dumps(report, indent=1, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    if args.preflight:
        return 0 if report["ok"] else 1
    return 0 if report.get("all_arms_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
