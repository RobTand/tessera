"""Projection geometry and the extra storage of direct weight consumers."""
from types import SimpleNamespace

import pytest

from tessera.serving.sharding import layer_replicas, plan_shard


@pytest.mark.parametrize("rank", [0, 1])
def test_kda_replication_keeps_the_complete_two_gate_roles(rank):
    roles = [("q_proj", 256), ("k_proj", 256), ("v_proj", 256),
             ("b_proj", 32), ("f_a_proj", 128), ("g_a_proj", 128)]
    layer = SimpleNamespace(tp_rank=rank, tp_size=2, replicated_shard_ids={4, 5})
    replicas = layer_replicas("model.layers.0.self_attn.in_proj_qkvbfg_a", layer,
                              [name for name, _ in roles])
    plan = plan_shard("model.layers.0.self_attn.in_proj_qkvbfg_a", roles=roles,
                      columns=256, out_partitions=[128, 128, 128, 16, 128, 128],
                      in_size=256, tp_rank=rank, tp_size=2, input_size=256,
                      output_size=1312, replicas=replicas)
    assert [(plan.role(name).lo, plan.role(name).hi) for name in ("f_a_proj", "g_a_proj")] == [
        (0, 128), (0, 128)]
    assert (plan.role("q_proj").lo, plan.role("q_proj").hi) == (rank * 128, (rank + 1) * 128)


@pytest.mark.parametrize("bad", [{6}, {-1}, {True}, {4.0}])
def test_replication_must_name_a_real_partition(bad):
    layer = SimpleNamespace(tp_rank=0, tp_size=2, replicated_shard_ids=bad)
    with pytest.raises(ValueError, match="replicated_shard_ids"):
        layer_replicas("kda", layer, ["q", "k", "v", "b", "f", "g"])


@pytest.mark.parametrize("family", ["TESSERA_FP8", "TESSERA_BF16", "TESSERA_NVFP4"])
def test_indexer_storage_prices_only_the_fp32_head_cache(family):
    from tessera.serving.projection_routes import direct_consumer_resident_bytes

    result = direct_consumer_resident_bytes(
        "model.layers.3.self_attn.indexer.wk_weights_proj", family, 160, 4096,
        [("wk", 128), ("weights_proj", 32)])
    assert result == {"resident_bytes_resident_mode": 32 * 4096 * 4,
                      "resident_bytes_stock": 32 * 4096 * 4}


def test_mla_storage_prices_the_actual_bf16_matrix():
    from tessera.serving.projection_routes import direct_consumer_resident_bytes

    result = direct_consumer_resident_bytes(
        "model.layers.3.self_attn.kv_b_proj", "TESSERA_FP8", 1024, 256,
        [("kv_b_proj", 1024)])
    assert result == {"resident_bytes_resident_mode": 1024 * 256 * 2,
                      "resident_bytes_stock": 1024 * 256 * 2}


def test_ordinary_projection_has_no_decoded_resident_copy():
    from tessera.serving.projection_routes import direct_consumer_resident_bytes

    assert direct_consumer_resident_bytes(
        "model.layers.3.self_attn.o_proj", "TESSERA_BF16", 256, 256,
        [("o_proj", 256)]) == {"resident_bytes_resident_mode": 0, "resident_bytes_stock": 0}


def test_indexer_cost_refuses_a_missing_head_role():
    from tessera.serving.projection_routes import direct_consumer_resident_bytes

    with pytest.raises(ValueError, match="weights_proj"):
        direct_consumer_resident_bytes(
            "model.layers.3.self_attn.indexer.wk_weights_proj", "TESSERA_FP8", 160, 4096,
            [("wk", 160)])
