#!/usr/bin/env python3
"""Replay a served route census against the ``lane_eligibility`` table (#111).

WHY THIS EXISTS.  Deriving a cell's ``executes`` from ``scheme.ROUTE_LAUNCHES``
proves the DOCUMENT agrees with the CODE.  Only a serve proves the code agrees
with the MACHINE, and the serve that matters here already happened: the R1024
census (#104) recorded ``tessera_window_gemv::gemv`` on 112 of 112 modules in
the decode regime on a document that published the materialised FP8 pair.  This
replays that receipt through ``census.cell_launch_agreement`` -- the same join
``tools/tessera_route_census.py`` now writes into every new receipt -- so the
evidence #111 was filed on is reproducible without a GPU and without a re-serve.

It also replays the PRE-#111 claim against the same records, because a check
that only ever passes is not a check: put the old value back and the records it
was filed on refuse it, once per module.

    python experiments/ts111_replay_cell_agreement.py \
        --census /home/rob/tessera-runs/ts104/census-R1024-readable.json \
        --checkpoint /mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024

The rung per module is the one fact a route record does not carry, so it is
read from the checkpoint's own ``config_groups`` -- the same place the census
tool reads it from during a serve.  Nothing here imports torch or vLLM.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tessera.serving.census import cell_launch_agreement          # noqa: E402
from tessera.serving.contract import (                            # noqa: E402
    CENSUS_PHASE_REGIMES, PAYLOAD_FAMILY_BY_ROUTE, load_serving_contract)


def rungs_by_module(checkpoint: pathlib.Path) -> dict:
    """``module -> q256``, from the checkpoint's own sidecar.

    A fused module's ``q256`` is a per-role list since contract v6; a group
    whose members disagree resolves to no rung, so its modules land in the
    honest ``unattested`` bucket rather than borrowing one member's cell.
    """
    config = json.loads((checkpoint / "config.json").read_text())
    out = {}
    for group in config["quantization_config"]["config_groups"].values():
        scheme = group.get("scheme", {})
        value = scheme.get("q256")
        if isinstance(value, list):
            distinct = {int(v) for v in value}
            value = distinct.pop() if len(distinct) == 1 else None
        for target in group.get("targets", []):
            out[target] = None if value is None else int(value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True, type=pathlib.Path,
                    help="a tessera.serving.route_census/1 receipt")
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path,
                    help="the checkpoint it was served from (read for its rungs)")
    args = ap.parse_args()

    receipt = json.loads(args.census.read_text())
    if receipt.get("compiled"):
        print("this receipt is COMPILED: one graph serves every M and stamps both "
              "launches as one `a+b` pair, which no cell publishes. The join is "
              "eager-only and there is nothing here to replay.")
        return 2
    records = receipt["records"]
    cells = load_serving_contract()["lane_eligibility"]["cells"]
    platform = "sm_{}{}".format(*receipt["device"]["capability"])
    rungs = rungs_by_module(args.checkpoint)

    print(f"receipt      : {args.census}")
    print(f"checkpoint   : {receipt['checkpoint']}")
    print(f"mode / eager : {receipt['env'].get('TESSERA_SERVE_MODE')} / "
          f"compiled = {receipt.get('compiled')}")
    print(f"device       : {receipt['device']['name']} ({platform})")
    print(f"modules      : { {p: len(r) for p, r in sorted(records.items())} }")

    print("\n=== the shipped table ===")
    block, problems = cell_launch_agreement(
        records, cells=cells, phase_regimes=CENSUS_PHASE_REGIMES, platform=platform,
        rungs_by_module=rungs, families_by_route=PAYLOAD_FAMILY_BY_ROUTE)
    print(json.dumps(block, indent=1))
    print(f"problems: {problems}")

    # THE NEGATIVE CONTROL.  Put the pre-#111 value back -- the decode regime
    # publishing the materialised FP8 pair -- and the records this was all
    # filed on refuse it, once per module.
    stale = json.loads(json.dumps(cells))
    hits = 0
    for cell in stale:
        if cell["family"] == "TESSERA_E4M3_K1" and cell["regime"] == "decode":
            cell["executes"] = [{"symbol": "torch._scaled_mm", "decoder": "torch_window"}]
            hits += 1
    assert hits, "no E4M3 decode cell to make stale; the table moved under this replay"
    print("\n=== the pre-#111 claim (decode = the materialised pair), same records ===")
    stale_block, stale_problems = cell_launch_agreement(
        records, cells=stale, phase_regimes=CENSUS_PHASE_REGIMES, platform=platform,
        rungs_by_module=rungs, families_by_route=PAYLOAD_FAMILY_BY_ROUTE)
    print(f"agrees: {stale_block['agrees']}  problems: {len(stale_problems)}")
    print(f"first : {stale_problems[0] if stale_problems else '(none -- the control failed)'}")

    ok = block["agrees"] is True and not problems and stale_problems
    print(f"\nVERDICT: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
