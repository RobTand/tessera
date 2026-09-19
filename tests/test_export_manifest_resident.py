"""The export manifest prices every resident row (tessera#557).

On the attested 2026-09-18 full-engine capture
(``frontier-qwen3-0.6b-20260918-399``) the dense startup check refused on two
layer-0 units: the ledger charged 20,484 resident bytes the manifest did not
price -- the BF16 unit's ``row_scale`` buffer (16,384 B), the NVFP4 unit's
``trellis_input_global_scale`` (4 B), and the three memoised trellis tables
the NVFP4 load pins (2048 + 1024 + 1024 B).  The byte figures below are that
ledger's, read from its ``dense_startup_check.units[*].resident_rows``; the
shapes are the serving load's own (``bf16_route`` registers ``[rows]`` fp32,
``nvfp4_route`` registers one fp32 scalar, ``decode._replay_tables`` builds
``[4, points]`` + two ``[2, states]`` int64).  No figure here is derived from
the code under test: the ledger is the receipt, the routes are the mechanism,
and the manifest is what must agree with both.
"""
import pytest

from experiments.full_engine_ownership import dense_startup_check
from tessera.serving_parts import dense_resident_bytes_resident_mode

#: ``g:model.layers.0.self_attn.qkv_proj``: BF16, 4096 rows x 1024 cols.
QKV_ROWS, QKV_COLS = 4096, 1024
#: Its ledger rows: the decoded tile and the fp32-per-row scale.
QKV_TILE, QKV_SCALE = 8388608, 16384
#: ``g:model.layers.0.mlp.gate_up_proj``: NVFP4, 6144 rows x 1024 cols.
GATE_ROWS, GATE_COLS = 6144, 1024
#: Its ledger rows: nibbles, block scales, the A-side scalar, and the three
#: load-pinned trellis tables.
GATE_NIBBLES, GATE_SCALES = 3145728, 393216
GATE_A_SCALE = 4
GATE_TABLES = (2048, 1024, 1024)


def test_bf16_manifest_prices_the_row_scale():
    """The BF16 figure is the tile plus one fp32 per row, not the tile alone."""
    assert dense_resident_bytes_resident_mode("TESSERA_BF16", QKV_ROWS, QKV_COLS) == (
        QKV_TILE + QKV_SCALE
    )


def test_nvfp4_manifest_prices_the_a_side_scale_and_the_trellis_tables():
    """The NVFP4 figure is planes plus the scalar plus the pinned tables."""
    assert dense_resident_bytes_resident_mode(
        "TESSERA_NVFP4", GATE_ROWS, GATE_COLS, trellis_table_bytes=sum(GATE_TABLES)
    ) == GATE_NIBBLES + GATE_SCALES + GATE_A_SCALE + sum(GATE_TABLES)


def test_fp8_manifest_accounting_is_unchanged():
    """The FP8 route already keeps exactly tile plus row scales (tessera#557
    found all 110 FP8 units agreeing); this pins that accounting in place."""
    assert dense_resident_bytes_resident_mode("TESSERA_FP8", 4096, 1024) == (
        4096 * 1024 + 4096 * 4
    )


def test_unknown_family_is_refused_not_priced_zero():
    """A family without an accounting rule must not silently price nothing."""
    with pytest.raises(ValueError, match="no resident-mode accounting"):
        dense_resident_bytes_resident_mode("TESSERA_MYSTERY", 8, 8)


def test_replay_table_bytes_matches_the_attested_load_tables():
    """The helper measures the capture's trellis: E2M1x2 at q896 serves rate
    7 (``q256 * arity / 256``), whose ``[4, 2**(R-1)]`` subset table is
    ``[4, 64]`` int64 and whose memory-6 code transition tables are
    ``[2, 64]`` int64 -- exactly the ledger's three unowned rows."""
    pytest.importorskip("torch")
    import torch

    from tessera.alphabet import E2M1_GRID, tuple_grid, build_forest
    from tessera.decode import _replay_tables, replay_table_bytes
    from tessera.export import DEFAULT_CODE

    forest = build_forest(7, grid=tuple_grid(E2M1_GRID, 2))
    subsets, table_next, table_sub = _replay_tables(forest, DEFAULT_CODE, "cpu")
    assert [(t.shape, t.dtype) for t in (subsets, table_next, table_sub)] == [
        (torch.Size([4, 64]), torch.int64),
        (torch.Size([2, 64]), torch.int64),
        (torch.Size([2, 64]), torch.int64),
    ]
    assert [t.numel() * t.element_size()
            for t in (subsets, table_next, table_sub)] == list(GATE_TABLES)
    assert replay_table_bytes(forest, DEFAULT_CODE) == sum(GATE_TABLES)


def _capture_rows():
    """The six disagreeing allocation rows, as ``(unit, bytes)``."""
    rows = [("qkv", QKV_TILE), ("qkv", QKV_SCALE)]
    rows += [("gate", b) for b in
             (GATE_NIBBLES, GATE_SCALES, *GATE_TABLES, GATE_A_SCALE)]
    return rows


def _check(manifest_qkv, manifest_gate):
    rows = [{"bytes": size, "allocation_id": f"test:{index}",
             "allocate_index": index, "free_completed_index": None}
            for index, (_unit, size) in enumerate(_capture_rows())]
    units = [unit for unit, _size in _capture_rows()]
    views = [{"class": "candidate", "unit": unit} for unit in units]
    dense = {
        "schema": "tessera.full_engine_dense_startup_observation.v1",
        "memory_allocated_bytes": sum(row["bytes"] for row in rows),
        "units": {
            "qkv": {"family": "TESSERA_BF16",
                    "manifest_resident_bytes_resident_mode": manifest_qkv},
            "gate": {"family": "TESSERA_NVFP4",
                     "manifest_resident_bytes_resident_mode": manifest_gate},
        },
    }
    return dense_startup_check(rows, views, dense, ready_index=len(rows))


def test_a_manifest_missing_the_scale_rows_refuses_naming_them():
    """The issue's red-first refusal: the old figures disagree on both units
    by exactly the unpriced rows, and the check names the bytes."""
    check = _check(QKV_TILE, GATE_NIBBLES + GATE_SCALES)
    assert check["closed"] is False
    assert check["units_disagreeing"] == ["gate", "qkv"]
    assert check["manifest_unpriced_resident_bytes"] == QKV_SCALE + sum(GATE_TABLES) + GATE_A_SCALE
    assert check["units"]["qkv"]["difference_bytes"] == QKV_SCALE
    assert check["units"]["gate"]["difference_bytes"] == sum(GATE_TABLES) + GATE_A_SCALE


def test_the_fixed_manifest_closes_the_capture_check():
    """The same rows against the priced manifest: exact per-unit agreement."""
    check = _check(QKV_TILE + QKV_SCALE,
                   GATE_NIBBLES + GATE_SCALES + GATE_A_SCALE + sum(GATE_TABLES))
    assert check["closed"] is True
    assert check["units_disagreeing"] == []
    assert check["manifest_unpriced_resident_bytes"] == 0
