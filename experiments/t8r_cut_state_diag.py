#!/usr/bin/env python3
"""Where does a TP row cut's window start state go wrong?  One expert
projection of a real wire, four spellings of the state immediately before
local row ``r0``:

* ``ref``      -- the reference decoder's replay of the FULL column
                  (``decode.replay_window`` over the parsed unit's expanded
                  ``body_bits``), row ``r0 - 1``;
* ``cut``      -- ``compact_prep._window_cut_state`` without scratch;
* ``prep``     -- ``prepare_window_compact(...).initial_state`` without scratch;
* ``prep_s``   -- the same with a caller-owned scratch dict (the routed
                  intake's spelling), fresh and then reused.

Also: the kernel-decoded codes of the cut unit's first rows against the
reference codes (``kernel_window_gemv.decode_codes`` ignores the start state,
so rows below ceil(L/R) are expected to differ there; rows past it must not).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench_routed_load import MOE_UNITS, _load_wires, _schemes, _target  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--experts", default="0,1")
    ap.add_argument("--r0", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from tessera import compact_prep as cp
    from tessera.decode import replay_window
    from tessera.serving.moe_route import _packed_group_shard_plan
    from tessera.serving.scheme import (expert_role_declarations, parse_compact_tessera_expert_blob,
                                        parse_tessera_expert_blob)

    device = torch.device("cuda")
    data = Path(args.data)
    experts = [int(e) for e in args.experts.split(",")]
    n_exp = max(experts) + 1
    wires = _load_wires(data, [args.layer], n_exp)[args.layer]
    declared = _schemes(data, [args.layer], n_exp)[args.layer]
    target = _target(args.layer)
    out = []
    scratch = {}
    for e in experts:
        for group, index, proj in MOE_UNITS:
            if proj == "down_proj":
                continue
            role = expert_role_declarations(declared["groups"][group])[index]
            blob = wires[e][proj].contiguous().numpy().tobytes()
            pu = parse_tessera_expert_blob(blob, role, f"{target} {proj} {e}", device=device)[0][1]
            unit = pu.unit
            L = int(unit.window_bits)
            rates = torch.tensor(unit.rates, device=device)
            ref = torch.zeros(unit.body_bits.shape[1], dtype=torch.int64, device=device)
            for present in sorted(set(unit.rates)):
                which = torch.nonzero(rates == present).squeeze(1)
                st = replay_window(unit.body_bits[:, which], L, present, None)
                ref[which] = st[args.r0 - 1]
            wire = parse_compact_tessera_expert_blob(blob, role, f"{target} {proj} {e}", device=device)[0][1]
            md = wire.metadata
            cols = int(md.columns) if hasattr(md, "columns") else int(unit.body_bits.shape[1])
            r1 = args.r0 + 1024
            cut = cp._window_cut_state(md, args.r0, 0, cols, device, None)
            prep = cp.prepare_window_compact(wire, rows=(args.r0, r1), device=device).initial_state
            prep_s = cp.prepare_window_compact(wire, rows=(args.r0, r1), device=device,
                                               scratch=scratch).initial_state
            row = {"expert": e, "proj": proj, "rates": sorted(set(int(r) for r in unit.rates)),
                   "window_bits": L, "md_rates_eq_unit_rates": list(md.rates) == list(unit.rates),
                   "shard_state_is_none": getattr(md, "shard_state", None) is None,
                   "unit_initial_state_is_none": getattr(unit, "initial_state", None) is None,
                   "release_entries": int(unit.release_index.numel()),
                   "cut_vs_ref_mismatch": int((cut.long() != ref).sum()),
                   "prep_vs_ref_mismatch": int((prep.long() != ref).sum()),
                   "prep_scratch_vs_ref_mismatch": int((prep_s.long() != ref).sum()),
                   "cols": cols}
            # the expanded body bits vs the packed plane's fields at the tail rows
            packed = cp._plane_u8(md.chunks[cp.PlaneKind.BODY], device, None, "body")
            rt = tuple(int(r) for r in md.rates)
            starts, cur = [], 0
            for c in range(cols):
                starts.append(cur)
                cur += rt[c] * int(md.rows)
            starts = torch.tensor(starts, dtype=torch.int64, device=device)
            bad_fields = 0
            for rr in range(args.r0 - 5, args.r0 + 2):
                for present in sorted(set(rt)):
                    which = torch.tensor([c for c in range(cols) if rt[c] == present], device=device)
                    f = cp.gather_packed_fields(packed, starts[which] + rr * present, present)
                    bad_fields += int((f != unit.body_bits[rr, which].long()).sum())
            row["packed_vs_expanded_body_field_mismatch_rows_r0-5..r0+1"] = bad_fields
            row["md_rows"] = int(md.rows)
            if row["cut_vs_ref_mismatch"]:
                i = int(torch.nonzero(cut.long() != ref)[0])
                row["first_bad_col"] = {"col": i, "rate": int(unit.rates[i]), "cut": int(cut[i]),
                                        "ref": int(ref[i])}
            out.append(row)
            print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
