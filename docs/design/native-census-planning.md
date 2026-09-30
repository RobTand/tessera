# Native census planning

CPU slice for tessera#689. GPU census acceptance remains held by the coordinator.

`tools/plan_native_census.py` builds `tessera.native_census_plan.v1` from an
explicit JSON array of requested scopes. It never initializes CUDA, loads
weights, executes a kernel, measures timing, publishes a cell, or submits work.
It refuses unknown fields and invalid selectors instead of guessing defaults.

```json
[
  {
    "route": "TESSERA_FP8",
    "grid": "E4M3",
    "q256": 880,
    "structure": "dense",
    "mode": "resident",
    "execution_mode": "eager",
    "regime": "decode",
    "tp_degree": 1,
    "requested_platform": "sm_121",
    "shape": {"M": 1, "N": 128, "K": 128}
  }
]
```

```sh
python tools/plan_native_census.py --requests requests.json --output plan.json
```

N/K are requested **rank-local** dimensions; the planner does not derive TP
geometry. Routed scopes also require `shape.experts` and `shape.topk`.
The M-to-regime rule and admissible launch pairs come from the shared dispatch
registry. No symbol roster or special list of rungs is copied into this tool.

Each row ID binds the complete requested scope, the packaged contract's raw
SHA-256, and the dispatch registry's canonical SHA-256. Request order does not
change the plan; duplicate scopes refuse. An existing output file is not
replaced. The advertised dense reader range is reported using the reader's
own range/step rule, and is explicitly **not routed admission**.

## What this does not prove

Every plan says `status: not_executed`, `gpu_executed: false`,
`qualification: not_measured`, and `measurement: null`. These are planning
records, not `lane_eligibility` cells or timing receipts. The requested platform
is not an observed device; no supplied image label is accepted as identity.
Admissible pairs describe registry alternatives, not launches observed on a
particular shape or prepared bundle.

The later GPU arm must load the actual encoded wire, bind its bytes and runtime
identity, verify geometry/native availability, and collect real route-census
evidence. New rungs need their own qualification. A CPU plan cannot extend a
reader range, promote a cell, establish graph replay, or certify served quality.
