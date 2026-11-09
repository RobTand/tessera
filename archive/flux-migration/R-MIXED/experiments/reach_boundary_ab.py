"""Issue #115: price upward landing for unraised CHANNEL boundary rows.

Issue #87 makes the reach raise exact: rows selected by its ``over`` mask are
stepped to the next fp16 word when round-to-nearest lands below ``amax/reach``.
The ordinary RMS start is deliberately unchanged.  A very small boundary set
therefore remains: an *unraised* row can clear the reach in float arithmetic
and then have its scale word round below the same bound.

This is the measured A/B for whether to widen that policy:

* ``nearest_unraised`` is the current encoder.  Raised rows land upward;
  unraised rows keep their ordinary round-to-nearest RMS word.
* ``upward_unraised`` starts from exactly that result and steps any remaining
  word below ``amax/reach`` upward.  It changes no field width or rate.

The population is derived from every 2-D Linear weight in Qwen3-0.6B rather
than copied from a roster.  First, every tensor is screened with the exact
initial-scale arithmetic.  Then every tensor containing at least one affected
unraised row is encoded as A, B, A-repeat at E4M3 q256=1024 with the shipping
recipe and default four scale refits.  The repeat must be byte-identical.

The receipt separates three different claims:

* initial representability: rows and source values beyond ``reach * scale``;
* final artifact saturation: source values beyond the *finished* row scale and
  emitted grid values equal to the table reach;
* reconstruction: plain and diagonal-H weighted relative error, both weight-
  space screens.  This is not a served BF16-teacher KL result.

It also recomputes the behaviour-derived ``encoder_fixture_id`` under both
arms.  That is load-bearing: the boundary is rare enough that merely including
"an unraised row" in a fixture does not prove the fixture reaches this branch.

Typical fleet invocation (all execution goes through PrismaBuild)::

    pbrun --gpu --demand gpu=1,cpu=2,mem_gb=24 --timeout-s 10800 \
      --wait-s 14400 --cwd "$PWD" -- \
      /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python \
      experiments/reach_boundary_ab.py \
      --out experiments/results/ts115_boundary_ab_qwen06b.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera import encoder_identity as ei  # noqa: E402
from tessera import scale_channel as sc  # noqa: E402
from tessera.alphabet import E4M3_GRID  # noqa: E402
from tessera.container import parse  # noqa: E402
from tessera.encode import grid_vector_table, window_table  # noqa: E402
from tessera.export import (  # noqa: E402
    DEFAULT_SCALE_REFIT,
    E4M3_RECIPE,
    encode_linear_planes,
)
from tessera.unit_artifact import read_unit_artifact  # noqa: E402

from bf16_route_weight_space import DENSE_H, DENSE_SRC, open_all  # noqa: E402


ARM_A = "nearest_unraised"
ARM_B = "upward_unraised"
ARM_REPEAT = "nearest_unraised_repeat"
CURRENT_INITIAL = sc.initial_channel_scale


def _wide_initial_channel_scale(work, sigma, reach=None):
    """Arm B: current #87 landing plus the remaining unraised boundary."""
    stored, effective, global_scale = CURRENT_INITIAL(work, sigma, reach=reach)
    if reach is None:
        return stored, effective, global_scale
    floor = work.float().abs().amax(dim=1) / float(reach)
    stored, effective = sc._bump_below_floor(
        stored, effective, floor, global_scale,
    )
    return stored, effective, global_scale


def _set_arm(arm: str) -> None:
    sc.initial_channel_scale = (
        _wide_initial_channel_scale if arm == ARM_B else CURRENT_INITIAL
    )


def _identity_for(arm: str) -> str:
    """Compute, rather than assume, whether #101 can see this byte move."""
    _set_arm(arm)
    ei._MEMO.clear()
    return ei.encoder_fixture_id().hex()


def _install_identity(arm: str, identities: dict[str, str]) -> None:
    """Make artifact stamping agree with the arm whose encode follows."""
    _set_arm(arm)
    key = ARM_B if arm == ARM_B else ARM_A
    ei._MEMO[:] = [bytes.fromhex(identities[key])]


def _linear_names(index) -> list[str]:
    """Every transformer-block 2-D Linear, derived from the checkpoint."""
    names = []
    for key, handle in index.items():
        if not key.endswith(".weight") or ".layers." not in key:
            continue
        if "norm" in key or "embed" in key:
            continue
        if len(handle.get_slice(key).get_shape()) != 2:
            continue
        names.append(key[:-len(".weight")])
    return sorted(names)


def _reach(device: torch.device) -> tuple[float, float]:
    sigma = (
        float(sc.default_channel_sigma(E4M3_GRID))
        if E4M3_RECIPE.channel_sigma is None
        else float(E4M3_RECIPE.channel_sigma)
    )
    table = window_table(
        E4M3_GRID,
        E4M3_RECIPE.window_bits,
        sigma=sigma,
        seed=E4M3_RECIPE.window_seed,
        half=16,
        device=device,
    )
    reach = float(
        grid_vector_table(E4M3_GRID, device=device)[table.long()].abs().max()
    )
    return reach, sigma


def _weighted_sums(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    h: "torch.Tensor | None",
    rows: "torch.Tensor | None" = None,
) -> dict[str, float]:
    if rows is not None:
        weight = weight[rows]
        reconstruction = reconstruction[rows]
    error2 = (reconstruction.float() - weight.float()).pow(2)
    weight2 = weight.float().pow(2)
    out = {
        "wt_num": float(error2.sum()),
        "wt_den": float(weight2.sum()),
    }
    if h is not None:
        hv = h.float().to(weight.device)
        out.update({
            "h_num": float((error2.sum(dim=0) * hv).sum()),
            "h_den": float((weight2.sum(dim=0) * hv).sum()),
        })
    return out


def _relative(sums: dict[str, float], prefix: str) -> float:
    den = sums[prefix + "_den"]
    return math.sqrt(sums[prefix + "_num"] / den) if den else 0.0


def _initial_pair(
    weight: torch.Tensor,
    h: "torch.Tensor | None",
    reach: float,
    sigma: float,
) -> tuple[dict, torch.Tensor]:
    w = weight.float()
    rms = w.pow(2).mean(dim=1).sqrt()
    amax = w.abs().amax(dim=1)
    floor = amax / reach
    over = amax * sigma > reach * rms

    _set_arm(ARM_A)
    stored_a, effective_a, global_a = sc.initial_channel_scale(
        w, sigma, reach=reach,
    )
    _set_arm(ARM_B)
    stored_b, effective_b, global_b = sc.initial_channel_scale(
        w, sigma, reach=reach,
    )
    if global_a != global_b:
        raise RuntimeError(
            f"the two initial arms changed the shared global: {global_a} != {global_b}"
        )

    boundary = (~over) & (effective_a < floor)
    moved = stored_a != stored_b
    if not torch.equal(moved, boundary):
        raise RuntimeError(
            "arm B moved rows other than the unraised words arm A landed below reach: "
            f"moved={int(moved.sum())}, boundary={int(boundary.sum())}"
        )
    if bool((effective_b < floor).any()):
        raise RuntimeError(
            f"arm B still left {int((effective_b < floor).sum())} row(s) below reach"
        )

    limit_a = reach * effective_a[:, None]
    limit_b = reach * effective_b[:, None]
    past_a = w.abs() > limit_a
    past_b = w.abs() > limit_b
    affected_values = []
    for row in torch.nonzero(boundary).flatten().cpu().tolist():
        columns = torch.nonzero(past_a[row]).flatten().cpu().tolist()
        affected_values.append({
            "row": row,
            "scale_word_a": float(stored_a[row]),
            "scale_word_b": float(stored_b[row]),
            "scale_word_bits_a": hex(
                int(stored_a[row:row + 1].contiguous().view(torch.int16).item())
                & 0xFFFF
            ),
            "scale_word_bits_b": hex(
                int(stored_b[row:row + 1].contiguous().view(torch.int16).item())
                & 0xFFFF
            ),
            "effective_scale_a": float(effective_a[row]),
            "effective_scale_b": float(effective_b[row]),
            "required_scale": float(floor[row]),
            "relative_scale_inflation": float(
                effective_b[row] / effective_a[row] - 1.0
            ),
            "values": [{
                "column": column,
                "source_value": float(w[row, column]),
                "abs_source_value": float(w[row, column].abs()),
                "represented_reach_a": float(limit_a[row, 0]),
                "represented_reach_b": float(limit_b[row, 0]),
                "excess_a": float(w[row, column].abs() - limit_a[row, 0]),
                "past_reach_b": bool(past_b[row, column]),
            } for column in columns],
        })

    def arm(effective: torch.Tensor) -> dict:
        limit = reach * effective[:, None]
        excess = (w.abs() - limit).clamp_min(0.0)
        values = w.abs() > limit
        sums = {
            "wt_num": float(excess.pow(2).sum()),
            "wt_den": float(w.pow(2).sum()),
        }
        if h is not None:
            hv = h.float().to(w.device)
            sums.update({
                "h_num": float((excess.pow(2).sum(dim=0) * hv).sum()),
                "h_den": float((w.pow(2).sum(dim=0) * hv).sum()),
            })
        return {
            "rows_past_reach": int(values.any(dim=1).sum()),
            "values_past_reach": int(values.sum()),
            "clip_relative_wt": _relative(sums, "wt"),
            "clip_relative_h": _relative(sums, "h") if "h_num" in sums else None,
            "clip_sums": sums,
        }

    rel = effective_b[moved] / effective_a[moved] - 1.0
    return ({
        "rows": int(w.shape[0]),
        "columns": int(w.shape[1]),
        "rows_raised": int(over.sum()),
        "rows_unraised": int((~over).sum()),
        "affected_unraised_rows": int(boundary.sum()),
        "affected_row_indices": torch.nonzero(boundary).flatten().cpu().tolist(),
        "affected_values": affected_values,
        "stored_words_moved": int(moved.sum()),
        "stored_word_steps": (
            sorted(set(
                (stored_b[moved].contiguous().view(torch.int16)
                 - stored_a[moved].contiguous().view(torch.int16)).cpu().tolist()
            )) if bool(moved.any()) else []
        ),
        "global_scale": float(global_a),
        "scale_inflation_min": float(rel.min()) if rel.numel() else 0.0,
        "scale_inflation_mean": float(rel.mean()) if rel.numel() else 0.0,
        "scale_inflation_max": float(rel.max()) if rel.numel() else 0.0,
        ARM_A: arm(effective_a),
        ARM_B: arm(effective_b),
    }, boundary)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_provenance(path: Path) -> dict:
    """Local HF revision/etag plus the small config's content identity."""
    metadata = path / ".cache/huggingface/download/model.safetensors.metadata"
    lines = metadata.read_text().splitlines() if metadata.exists() else []
    config = path / "config.json"
    return {
        "path": str(path),
        "huggingface_revision": lines[0] if lines else None,
        "model_safetensors_etag": lines[1] if len(lines) > 1 else None,
        "config_sha256": _file_sha(config) if config.exists() else None,
    }


def _encode_arm(
    arm: str,
    identities: dict[str, str],
    weight: torch.Tensor,
    h: "torch.Tensor | None",
    boundary: torch.Tensor,
    name: str,
    q256: int,
    reach: float,
    scale_refit: int,
    refit_reach_floor: bool,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    _install_identity(arm, identities)
    started = time.time()
    exported, unit, _forests = encode_linear_planes(
        weight,
        grid=E4M3_GRID,
        q256=q256,
        name=name,
        verify=True,
        scale_refit=scale_refit,
        refit_reach_floor=refit_reach_floor,
    )
    reconstruction = read_unit_artifact(exported.blob, device=weight.device)
    effective = unit.scale_rows.float().to(weight.device) * unit.scale_global
    values_past = weight.abs() > reach * effective[:, None]
    emitted = grid_vector_table(E4M3_GRID, device=weight.device)[
        unit.codes.long()
    ].squeeze(-1)
    emitted_at_reach = emitted.abs() == reach
    finished = time.time()
    full = _weighted_sums(weight, reconstruction, h)
    selected = _weighted_sums(weight, reconstruction, h, boundary)
    manifest = parse(exported.blob).manifest
    record = {
        "container_bytes": len(exported.blob),
        "priced_bytes": int(exported.exact_bytes),
        "bpp": float(exported.bpp),
        "artifact_sha256": _sha(exported.blob),
        "payload_sha256": manifest.payload_digest.hex(),
        "encoder_profile_id": manifest.encoder_profile_id.hex(),
        "encoder_fixture_id": (
            manifest.encoder_fixture_id or ei.UNTAGGED_ENCODER_ID
        ).hex(),
        "started_unix": started,
        "finished_unix": finished,
        "seconds": finished - started,
        "rows_past_final_reach": int(values_past.any(dim=1).sum()),
        "values_past_final_reach": int(values_past.sum()),
        "emitted_values_at_reach": int(emitted_at_reach.sum()),
        "values_past_and_emitted_at_reach": int(
            (values_past & emitted_at_reach).sum()
        ),
        "wt": _relative(full, "wt"),
        "h": _relative(full, "h") if "h_num" in full else None,
        "boundary_wt": _relative(selected, "wt"),
        "boundary_h": _relative(selected, "h") if "h_num" in selected else None,
        "score_sums": full,
        "boundary_score_sums": selected,
    }
    return record, unit.scale_rows.detach().clone(), unit.codes.detach().clone(), reconstruction


def _summarise_encoded(
    records: dict[str, dict], aggregate_sums: dict[str, dict[str, float]]
) -> dict:
    a, b = records[ARM_A], records[ARM_B]
    ratios = {
        "wt_ratio_b_over_a": b["wt"] / a["wt"] if a["wt"] else 1.0,
        "h_ratio_b_over_a": (
            b["h"] / a["h"] if a["h"] not in (None, 0.0) else None
        ),
        "boundary_wt_ratio_b_over_a": (
            b["boundary_wt"] / a["boundary_wt"]
            if a["boundary_wt"] else 1.0
        ),
        "boundary_h_ratio_b_over_a": (
            b["boundary_h"] / a["boundary_h"]
            if a["boundary_h"] not in (None, 0.0) else None
        ),
    }
    for arm in (ARM_A, ARM_B):
        sums = records[arm]["score_sums"]
        for key, value in sums.items():
            aggregate_sums[arm][key] = aggregate_sums[arm].get(key, 0.0) + value
    return ratios


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")


def main() -> None:
    run_started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DENSE_SRC)
    parser.add_argument("--hessian", default=DENSE_H)
    parser.add_argument("--q256", type=int, default=1024)
    parser.add_argument("--scale-refit", type=int, default=DEFAULT_SCALE_REFIT)
    parser.add_argument("--refit-reach-floor", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="debug only: screen the first N derived tensors")
    parser.add_argument("--census-only", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch reports no CUDA device")
    index = open_all(args.model)
    names = _linear_names(index)
    if args.limit:
        names = names[:args.limit]
    h_all = torch.load(args.hessian, map_location="cpu", weights_only=False)
    reach, sigma = _reach(device)

    identities = {
        ARM_A: _identity_for(ARM_A),
        ARM_B: _identity_for(ARM_B),
    }
    identities[ARM_REPEAT] = _identity_for(ARM_A)
    if identities[ARM_REPEAT] != identities[ARM_A]:
        raise RuntimeError("the A encoder identity changed on its repeat")

    document = {
        "schema": "tessera.issue115.reach_boundary_ab.v1",
        "claim": "weight-space screen; not served BF16-teacher KL",
        "args": vars(args),
        "environment": {
            "host": socket.gethostname(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "run_started_unix": run_started,
        },
        "recipe": {
            "grid": E4M3_GRID.name,
            "q256": args.q256,
            "window_bits": E4M3_RECIPE.window_bits,
            "window_seed": E4M3_RECIPE.window_seed,
            "channel_sigma": sigma,
            "reach_grid_units": reach,
            "scale_refit": args.scale_refit,
            "refit_reach_floor": args.refit_reach_floor,
        },
        "encoder_identities": {
            **identities,
            "b_moves_identity": identities[ARM_B] != identities[ARM_A],
        },
        "population": {
            "model": _model_provenance(Path(args.model)),
            "hessian_path": args.hessian,
            "hessian_sha256": _file_sha(Path(args.hessian)),
            "selection": "every transformer-block 2-D Linear weight",
            "tensors_screened": len(names),
            "rows_screened": 0,
            "values_screened": 0,
            "hessian_tensors": 0,
            "hessian_tensor_names": [],
            "affected_tensors": 0,
            "affected_unraised_rows": 0,
            "affected_source_values": 0,
            "affected_hessian_tensor_names": [],
        },
        "tensors": {},
    }
    output = Path(args.out)
    print(
        f"#115 A/B: {len(names)} derived tensors, E4M3 q{args.q256}, "
        f"L={E4M3_RECIPE.window_bits}, reach={reach:g}, device={device}",
        flush=True,
    )
    print(
        "encoder identity: "
        f"A={identities[ARM_A][:16]} B={identities[ARM_B][:16]} "
        f"moved={identities[ARM_B] != identities[ARM_A]}",
        flush=True,
    )

    affected: list[str] = []
    screen_sums = {
        ARM_A: {"wt_num": 0.0, "wt_den": 0.0, "h_num": 0.0, "h_den": 0.0},
        ARM_B: {"wt_num": 0.0, "wt_den": 0.0, "h_num": 0.0, "h_den": 0.0},
    }
    screen_counts = {
        ARM_A: {"rows_past_reach": 0, "values_past_reach": 0},
        ARM_B: {"rows_past_reach": 0, "values_past_reach": 0},
    }
    initial_inflations: list[float] = []
    boundaries: dict[str, list[int]] = {}
    for number, name in enumerate(names, 1):
        source = index[name + ".weight"].get_tensor(name + ".weight")
        source_dtype = str(source.dtype)
        w = source.to(device).float()
        h = h_all.get(name) if isinstance(h_all, dict) else None
        h = None if h is None or h.numel() != w.shape[1] else h
        initial, boundary = _initial_pair(w, h, reach, sigma)
        population = document["population"]
        population["rows_screened"] += int(w.shape[0])
        population["values_screened"] += int(w.numel())
        population["hessian_tensors"] += int(h is not None)
        if h is not None:
            population["hessian_tensor_names"].append(name)
        population["affected_unraised_rows"] += initial["affected_unraised_rows"]
        population["affected_source_values"] += initial[ARM_A]["values_past_reach"]
        for arm in (ARM_A, ARM_B):
            sums = initial[arm]["clip_sums"]
            for key, value in sums.items():
                screen_sums[arm][key] += value
            for key in ("rows_past_reach", "values_past_reach"):
                screen_counts[arm][key] += initial[arm][key]
        if initial["affected_unraised_rows"]:
            affected.append(name)
            if h is not None:
                population["affected_hessian_tensor_names"].append(name)
            boundaries[name] = initial["affected_row_indices"]
            initial_inflations.extend(
                row["relative_scale_inflation"]
                for row in initial["affected_values"]
            )
            document["tensors"][name] = {
                "shape": list(w.shape),
                "source_dtype": source_dtype,
                "initial": initial,
            }
            print(
                f"screen {number:>3}/{len(names)} {name}: "
                f"{initial['affected_unraised_rows']} row(s), "
                f"{initial[ARM_A]['values_past_reach']} value(s), "
                f"inflation <= {initial['scale_inflation_max']:.6e}",
                flush=True,
            )
        del source, w
        if device.type == "cuda":
            torch.cuda.empty_cache()

    document["population"]["affected_tensors"] = len(affected)
    document["population"]["affected_tensor_names"] = affected
    document["environment"]["screen_finished_unix"] = time.time()
    document["initial_aggregate"] = {
        ARM_A: {
            **screen_counts[ARM_A],
            "clip_relative_wt": _relative(screen_sums[ARM_A], "wt"),
            "clip_relative_h": (
                _relative(screen_sums[ARM_A], "h")
                if document["population"]["hessian_tensors"] else None
            ),
            "clip_sums": screen_sums[ARM_A],
        },
        ARM_B: {
            **screen_counts[ARM_B],
            "clip_relative_wt": _relative(screen_sums[ARM_B], "wt"),
            "clip_relative_h": (
                _relative(screen_sums[ARM_B], "h")
                if document["population"]["hessian_tensors"] else None
            ),
            "clip_sums": screen_sums[ARM_B],
        },
        "scale_inflation_min": min(initial_inflations) if initial_inflations else 0.0,
        "scale_inflation_mean": (
            statistics.fmean(initial_inflations) if initial_inflations else 0.0
        ),
        "scale_inflation_max": max(initial_inflations) if initial_inflations else 0.0,
    }
    _write(output, document)
    print(
        f"screened {document['population']['rows_screened']} rows / "
        f"{document['population']['values_screened']} values: "
        f"{len(affected)} tensors, "
        f"{document['population']['affected_unraised_rows']} affected rows, "
        f"{document['population']['affected_source_values']} values past reach",
        flush=True,
    )
    if args.census_only:
        print(f"wrote census-only {output}", flush=True)
        return

    aggregate_sums = {ARM_A: {}, ARM_B: {}}
    tensor_wt_ratios: list[float] = []
    tensor_h_ratios: list[float] = []
    rows_moved = codes_moved = reconstruction_values_moved = 0
    payloads_moved = 0
    container_bytes = {ARM_A: 0, ARM_B: 0}
    priced_bytes = {ARM_A: 0, ARM_B: 0}
    final_counts = {
        ARM_A: {
            "rows_past_final_reach": 0,
            "values_past_final_reach": 0,
            "emitted_values_at_reach": 0,
            "values_past_and_emitted_at_reach": 0,
        },
        ARM_B: {
            "rows_past_final_reach": 0,
            "values_past_final_reach": 0,
            "emitted_values_at_reach": 0,
            "values_past_and_emitted_at_reach": 0,
        },
    }
    for number, name in enumerate(affected, 1):
        w = index[name + ".weight"].get_tensor(name + ".weight").to(device).float()
        h = h_all.get(name) if isinstance(h_all, dict) else None
        h = None if h is None or h.numel() != w.shape[1] else h
        boundary = torch.zeros(w.shape[0], dtype=torch.bool, device=device)
        boundary[torch.tensor(boundaries[name], device=device)] = True
        records = {}
        planes = {}
        codes = {}
        reconstructions = {}
        for arm in (ARM_A, ARM_B, ARM_REPEAT):
            record, plane, body_codes, reconstruction = _encode_arm(
                arm,
                identities,
                w,
                h,
                boundary,
                name,
                args.q256,
                reach,
                args.scale_refit,
                args.refit_reach_floor,
            )
            records[arm] = record
            planes[arm] = plane
            codes[arm] = body_codes
            reconstructions[arm] = reconstruction
        if records[ARM_A]["artifact_sha256"] != records[ARM_REPEAT]["artifact_sha256"]:
            raise RuntimeError(f"{name}: A repeat is not byte-identical")
        if records[ARM_A]["container_bytes"] != records[ARM_B]["container_bytes"]:
            raise RuntimeError(
                f"{name}: matched-rate arms differ in bytes: "
                f"{records[ARM_A]['container_bytes']} != "
                f"{records[ARM_B]['container_bytes']}"
            )
        if records[ARM_A]["priced_bytes"] != records[ARM_B]["priced_bytes"]:
            raise RuntimeError(
                f"{name}: matched-rate arms differ in priced bytes: "
                f"{records[ARM_A]['priced_bytes']} != "
                f"{records[ARM_B]['priced_bytes']}"
            )
        if records[ARM_A]["bpp"] != records[ARM_B]["bpp"]:
            raise RuntimeError(
                f"{name}: matched-rate arms differ in bpp: "
                f"{records[ARM_A]['bpp']} != {records[ARM_B]['bpp']}"
            )
        pair = _summarise_encoded(records, aggregate_sums)
        moved_rows = int((planes[ARM_A] != planes[ARM_B]).sum())
        moved_codes = int((codes[ARM_A] != codes[ARM_B]).sum())
        moved_reconstruction = int(
            (reconstructions[ARM_A] != reconstructions[ARM_B]).sum()
        )
        final_rel = (
            planes[ARM_B].float() / planes[ARM_A].float() - 1.0
        )
        moved_final = planes[ARM_A] != planes[ARM_B]
        pair.update({
            "matched_bytes": True,
            "payload_moved": (
                records[ARM_A]["payload_sha256"] != records[ARM_B]["payload_sha256"]
            ),
            "final_row_words_moved": moved_rows,
            "final_row_words_up": int((final_rel > 0).sum()),
            "final_row_words_down": int((final_rel < 0).sum()),
            "final_row_indices_moved": (
                torch.nonzero(moved_final).flatten().cpu().tolist()
            ),
            "final_row_details": [{
                "row": row,
                "scale_word_a": float(planes[ARM_A][row]),
                "scale_word_b": float(planes[ARM_B][row]),
                "scale_word_bits_a": hex(
                    int(planes[ARM_A][row:row + 1].contiguous()
                        .view(torch.int16).item()) & 0xFFFF
                ),
                "scale_word_bits_b": hex(
                    int(planes[ARM_B][row:row + 1].contiguous()
                        .view(torch.int16).item()) & 0xFFFF
                ),
                "relative_delta": float(final_rel[row]),
            } for row in torch.nonzero(moved_final).flatten().cpu().tolist()],
            "final_scale_max_abs_relative_delta": (
                float(final_rel[moved_final].abs().max())
                if bool(moved_final.any()) else 0.0
            ),
            "body_codes_moved": moved_codes,
            "body_codes_moved_fraction": moved_codes / codes[ARM_A].numel(),
            "reconstruction_values_moved": moved_reconstruction,
            "reconstruction_delta_over_weight": float(
                (reconstructions[ARM_B] - reconstructions[ARM_A]).norm() / w.norm()
            ),
        })
        document["tensors"][name]["encodes"] = records
        document["tensors"][name]["pair"] = pair
        rows_moved += moved_rows
        codes_moved += moved_codes
        reconstruction_values_moved += moved_reconstruction
        payloads_moved += int(pair["payload_moved"])
        for arm in (ARM_A, ARM_B):
            container_bytes[arm] += records[arm]["container_bytes"]
            priced_bytes[arm] += records[arm]["priced_bytes"]
            for key in final_counts[arm]:
                final_counts[arm][key] += records[arm][key]
        tensor_wt_ratios.append(pair["wt_ratio_b_over_a"])
        if pair["h_ratio_b_over_a"] is not None:
            tensor_h_ratios.append(pair["h_ratio_b_over_a"])
        _write(output, document)
        print(
            f"encode {number:>2}/{len(affected)} {name}: "
            f"bytes={records[ARM_A]['container_bytes']} matched; "
            f"rows!={moved_rows} codes!={moved_codes} "
            f"past={records[ARM_A]['values_past_final_reach']}->"
            f"{records[ARM_B]['values_past_final_reach']} "
            f"wt={pair['wt_ratio_b_over_a']:.9f}x "
            + (
                f"h={pair['h_ratio_b_over_a']:.9f}x"
                if pair["h_ratio_b_over_a"] is not None else "h=n/a"
            ),
            flush=True,
        )
        del w, planes, codes, reconstructions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = {}
    for arm in (ARM_A, ARM_B):
        sums = aggregate_sums[arm]
        aggregate[arm] = {
            "wt": _relative(sums, "wt"),
            "h": _relative(sums, "h") if "h_num" in sums else None,
            "score_sums": sums,
        }
    aggregate.update({
        "scope": "affected tensors only; unaffected initial planes are identical by construction",
        "wire_bytes_matched_every_tensor": True,
        "container_bytes": container_bytes,
        "priced_bytes": priced_bytes,
        "payloads_moved": payloads_moved,
        "final_reach_counts": final_counts,
        "final_row_words_moved": rows_moved,
        "body_codes_moved": codes_moved,
        "reconstruction_values_moved": reconstruction_values_moved,
        "wt_ratio_b_over_a": (
            aggregate[ARM_B]["wt"] / aggregate[ARM_A]["wt"]
            if aggregate[ARM_A]["wt"] else 1.0
        ),
        "h_ratio_b_over_a": (
            aggregate[ARM_B]["h"] / aggregate[ARM_A]["h"]
            if aggregate[ARM_A]["h"] not in (None, 0.0) else None
        ),
        "h_scope": {
            "kind": "diagonal activation-second-moment screen",
            "tensors": document["population"]["affected_hessian_tensor_names"],
            "served_bf16_teacher_kl": False,
        },
        "tensor_wt_ratio_mean": statistics.fmean(tensor_wt_ratios),
        "tensor_wt_ratio_median": statistics.median(tensor_wt_ratios),
        "tensor_wt_better": sum(r < 1.0 for r in tensor_wt_ratios),
        "tensor_wt_worse": sum(r > 1.0 for r in tensor_wt_ratios),
        "tensor_wt_equal": sum(r == 1.0 for r in tensor_wt_ratios),
        "tensor_h_ratio_mean": (
            statistics.fmean(tensor_h_ratios) if tensor_h_ratios else None
        ),
        "tensor_h_ratio_median": (
            statistics.median(tensor_h_ratios) if tensor_h_ratios else None
        ),
        "tensor_h_better": sum(r < 1.0 for r in tensor_h_ratios),
        "tensor_h_worse": sum(r > 1.0 for r in tensor_h_ratios),
        "tensor_h_equal": sum(r == 1.0 for r in tensor_h_ratios),
    })
    document["encode_aggregate"] = aggregate
    document["environment"]["run_finished_unix"] = time.time()
    _write(output, document)
    print(
        "aggregate affected tensors: "
        f"wt={aggregate['wt_ratio_b_over_a']:.9f}x "
        + (
            f"h={aggregate['h_ratio_b_over_a']:.9f}x "
            if aggregate["h_ratio_b_over_a"] is not None else "h=n/a "
        )
        + f"rows!={rows_moved} codes!={codes_moved}; wrote {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
