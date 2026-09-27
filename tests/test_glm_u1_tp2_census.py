"""The all-E2M1 GLM stub, served at a tensor-parallel world of two, natively.

``TESSERA_E2M1_K2`` keeps ``max_world_size: 2`` in contract v39.  Its world-size
receipt (``glm53_a4_stub_tp2_sm121``, v29) was taken on the materialising
``torch_materialize_stock`` route v39 withdrew, and the native span-2 decoder
has a rank-cut admission that decode does not have
(``lane_planes.require_native_select_plane_admission``: a rank's rows must be a
whole number of select columns, ``arity * 8 * span`` = 32 for E2M1x2).  So the
v29 receipt alone did not show that the grouped native route admits and serves
a two-rank cut of these wires.

This receipt does.  Stub D of the U1 set (eight GLM-5.3-Flash layers, every
Tessera module E2M1) was censused at TP 2 across sparky and sparklina (ray
executor, eager, resident) on the GLM serving image, 2026-09-27 01:10Z-01:12Z,
with the census tool at 62f9e1ce5.  It is committed byte for byte beside the
stub's ``config.json`` (which the TP1 receipt of the same stub also cites).

What it pins:

1. the receipt is the one cited: sha256, verdict ``served`` with no problems,
   the image, eager, the same checkpoint config, and a world of two that
   answered as two ranks on two hosts;
2. every module of every rank, in both phases, ran the two native A4 launches
   -- no materialising fallback on either rank;
3. each rank's shape is the TP1 shape of the same module cut in half along
   exactly one axis, and both axes are cut for both structures, so the receipt
   exercises both cuts; every rank-local extent is a multiple of the
   admission's 32, and the live serve admitted it (a refused cut would not
   have reached the native launch on that rank);
4. the contract still says 2 for the unit, with both loader axes sharded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tessera.serving.contract import load_serving_contract

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TP2 = RESULTS / "glm53_u1_stub_d_tp2_eager_census.json"
TP1 = RESULTS / "glm53_u1_stub_d_tp1_eager_census.json"
CONFIG = RESULTS / "glm53_u1_stub_d_config.json"
TP2_SHA256 = "1a286a81d36b891f8839669551d96bbbddfbbc29b7a8bea07f31c99bf18e2824"
IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5")
A4_LAUNCH = {"dense": ("tessera.kernel_a4.a4_span2_gemm", "native_span2_gemm"),
             "moe": ("tessera.kernel_a4.a4_span2_grouped_gemm", "native_span2_grouped")}
#: ``arity * 8 * span`` for E2M1x2 (arity 2, span 2): the native span-2
#: select plane packs eight super-symbols to a byte with no per-column offset.
SELECT_COLUMN_ROWS = 2 * 8 * 2


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _nk(shape):
    fields = dict((f[0], int(f[1:])) for f in shape.split(":"))
    return fields["N"], fields["K"]


def test_the_committed_receipt_is_the_two_rank_serve_it_says_it_is():
    raw = TP2.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TP2_SHA256
    receipt = json.loads(raw)
    assert receipt["verdict"] == "served" and receipt["problems"] == []
    assert receipt["runtime"] == {"execution_mode": "eager", "image": IMAGE}
    assert receipt["checkpoint_sidecars"]["config.json"] == \
        hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert receipt["env"]["TESSERA_SERVE_MODE"] == "resident"
    assert receipt["topology"]["requested_tensor_parallel_size"] == 2
    assert receipt["topology"]["observed_world_size"] == 2
    assert receipt["topology"]["distributed_executor_backend"] == "ray"
    ranks = receipt["ranks"]
    assert sorted(r["rank"] for r in ranks) == [0, 1]
    assert sorted(r["node"] for r in ranks) == ["sparklina", "sparky"]
    for rank in ranks:
        assert rank["world_size"] == 2
        assert rank["runtime_image"] == IMAGE
        assert rank["platform_token"] == "sm_121"
        assert rank["lane_refusals"] == {}


def test_every_module_on_every_rank_ran_the_native_a4_launches():
    receipt = _load(TP2)
    for rank in receipt["ranks"]:
        for phase, records in rank["records"].items():
            assert len(records) == 21, (rank["rank"], phase)
            for name, record in records.items():
                assert (record["symbol"], record["decoder"]) == A4_LAUNCH[record["kind"]], (
                    rank["rank"], phase, name)
                assert record["contract"] == "e2m1_group16_ue4m3_static"
                assert record["state"] == "served"
    for phase, histogram in receipt["histogram"].items():
        assert histogram["tessera_modules"] == 42, phase  # 21 per rank
        assert histogram["other_route_modules"] == 0, phase


def test_each_rank_is_the_tp1_module_cut_in_half_on_one_axis_and_admits_the_cut():
    tp1 = {name: _nk(rec["shape"]) for name, rec in _load(TP1)["records"]["decode"].items()}
    axes_by_kind: dict = {}
    for rank in _load(TP2)["ranks"]:
        for name, record in rank["records"]["decode"].items():
            n1, k1 = tp1[name]
            n2, k2 = _nk(record["shape"])
            if (n2, k2) == (n1 // 2, k1):
                axis = "column"
            elif (n2, k2) == (n1, k1 // 2):
                axis = "row"
            else:
                raise AssertionError(f"{name}: TP1 N{n1}:K{k1} -> rank N{n2}:K{k2}")
            axes_by_kind.setdefault(record["kind"], set()).add(axis)
            # Both local extents, so the check holds whichever axis the wire
            # lays its rows along.
            assert n2 % SELECT_COLUMN_ROWS == 0 and k2 % SELECT_COLUMN_ROWS == 0, (
                name, n2, k2)
    assert axes_by_kind == {"dense": {"row", "column"}, "moe": {"row", "column"}}


def test_the_contract_keeps_e2m1_at_two_with_both_axes_sharded():
    units = {u["unit"]: u for u in load_serving_contract()["tensor_parallel"]["units"]}
    unit = units["TESSERA_E2M1_K2"]
    assert unit["max_world_size"] == 2
    assert {axis: v["status"] for axis, v in unit["loader_axes"].items()} == {
        "row": "sharded", "column": "sharded"}
