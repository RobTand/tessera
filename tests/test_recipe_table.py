"""The config carries the recipe per rung, and a checkpoint replays at its own meaning.

``wire_recipe`` is rung-independent today (TCQ everywhere); the E4M3 and
E2M1x2 sub-cap flips are gated on the encoder.  These tests exercise the
machinery those flips land on -- a resolver that varies with the rung, a
config table that records it, a replay that reads it back per unit -- by
patching ``wire_recipe`` rather than waiting for the flip.
"""

import pytest
import torch

from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.export import (
    PER_RUNG,
    TCQ_RECIPE,
    RecipeRange,
    WireRecipe,
    encode_linear,
    encode_settings_from_config,
    export_checkpoint,
    load_tessera_weight,
    read_checkpoint_config,
    recipe_at,
    recipe_table,
    rung_ceiling,
    wire_recipe,
)
from tessera.grammar import GrammarError
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.unit_artifact import parse, read_unit_artifact

K2 = tuple_grid(E2M1_GRID, 2, "coset")
WINDOW = BodyKind.WINDOW
CAP_Q256 = 7 * 256 // 2          # the E2M1x2 coset trellis cap, 3.5 b/wt payload

WINDOW_LUT = WireRecipe(WINDOW, 1, ScalePlaneKind.LUT, window_bits=8, window_seed=3)
WINDOW_CHANNEL = WireRecipe(WINDOW, 1, ScalePlaneKind.CHANNEL, window_bits=8)


def _sub_cap_window(grid, q256=None):
    """The E2M1x2 flip the doc names: window below the cap, TCQ at it."""
    if grid == K2 and q256 is not None and q256 < CAP_Q256:
        return WINDOW_LUT
    return TCQ_RECIPE


def _weights(rows=64, cols=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g)


def test_the_table_today_is_one_tcq_range_to_the_window_ceiling():
    for grid, ceiling in ((E2M1_GRID, 1024), (K2, 1024), (E4M3_GRID, 2048)):
        assert rung_ceiling(grid) == ceiling
        table = recipe_table(grid)
        assert table == (RecipeRange(1, ceiling, TCQ_RECIPE),)
        assert recipe_at(table, ceiling) == TCQ_RECIPE
        with pytest.raises(GrammarError, match="outside"):
            recipe_at(table, ceiling + 1)


def test_the_table_records_a_rung_dependent_recipe_as_ranges():
    table = recipe_table(K2, _sub_cap_window)
    assert table == (RecipeRange(1, CAP_Q256 - 1, WINDOW_LUT),
                     RecipeRange(CAP_Q256, 1024, TCQ_RECIPE))
    for entry in table:
        assert RecipeRange.from_config(entry.to_config()) == entry
    assert WireRecipe.from_config(WINDOW_CHANNEL.to_config()) == WINDOW_CHANNEL
    with pytest.raises(GrammarError, match="body kind"):
        WireRecipe.from_config({**WINDOW_LUT.to_config(), "body": "hash"})
    with pytest.raises(GrammarError, match="scale plane"):
        WireRecipe.from_config({**WINDOW_LUT.to_config(), "plane": "rank1"})


def test_a_mixed_plan_exports_each_unit_at_its_rung_and_replays_it(tmp_path, monkeypatch):
    monkeypatch.setattr("tessera.export.wire_recipe", _sub_cap_window)
    assert wire_recipe(K2, 640) == TCQ_RECIPE          # the import is the unpatched name
    tensors = {"low": _weights(seed=1), "cap": _weights(seed=2)}
    plan = {"low": 640, "cap": CAP_Q256}
    report = export_checkpoint(tensors, plan, tmp_path, grid=K2, scale_refit=1)
    bodies = {u.name: parse(u.blob).manifest.body for u in report.units}
    assert bodies == {"low": WINDOW, "cap": BodyKind.TCQ}
    config = read_checkpoint_config(tmp_path)
    # the flat keys are projections and say so when they cannot be one value
    assert config["body"]["kind"] == PER_RUNG
    assert config["body"]["window_bits"] is None and config["trellis"]["span"] is None
    assert config["scale"]["plane"] == "lut16"          # uniform across the table
    assert [(e["q256_lo"], e["q256_hi"], e["body"]) for e in config["wire"]["recipes"]] == \
        [(1, CAP_Q256 - 1, "window"), (CAP_Q256, 1024, "tcq")]
    # replay resolves per rung, and refuses to guess without one
    with pytest.raises(GrammarError, match="varies with the rung"):
        encode_settings_from_config(config)
    for name, q in plan.items():
        settings = encode_settings_from_config(config, q)
        assert settings["body"] is bodies[name]
        replay = encode_linear(tensors[name], grid=K2, q256=q, verify=False, **settings)
        assert torch.equal(load_tessera_weight(tmp_path, name), read_unit_artifact(replay.blob))
    with pytest.raises(GrammarError, match="outside"):
        encode_settings_from_config(config, 4096)


def test_a_uniform_table_keeps_the_flat_keys_and_needs_no_rung(tmp_path):
    w = _weights()
    export_checkpoint({"a": w}, {"a": 4 * 256}, tmp_path, grid=E4M3_GRID, scale_refit=1,
                      body=WINDOW, window_bits=8, scale_plane=ScalePlaneKind.CHANNEL)
    config = read_checkpoint_config(tmp_path)
    assert config["body"]["kind"] == "window" and config["scale"]["plane"] == "channel"
    table = config["wire"]["recipes"]
    assert len(table) == 1 and (table[0]["q256_lo"], table[0]["q256_hi"]) == (1, 2048)
    assert table[0]["channel_sigma"] == config["scale"]["sigma"]
    # the caller's overrides are what the table records, not the module default
    assert RecipeRange.from_config(table[0]).recipe.body is WINDOW
    assert encode_settings_from_config(config) == encode_settings_from_config(config, 4 * 256)


def test_a_per_rung_config_without_its_table_is_refused():
    with pytest.raises(GrammarError, match="wire.recipes"):
        encode_settings_from_config({"scale": {"plane": PER_RUNG}})
    with pytest.raises(GrammarError, match="empty"):
        encode_settings_from_config({"wire": {"recipes": []}})


def test_window_bits_without_a_window_body_is_refused():
    with pytest.raises(GrammarError, match="no window_bits"):
        encode_linear(_weights(), grid=K2, q256=640, window_bits=8, body=BodyKind.TCQ)
