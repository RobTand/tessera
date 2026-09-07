"""Synthetic CPU controls for the read-only stock worker KV inspection RPC."""
import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from experiments.full_engine_kv import inspect_worker_kv, kv_storage_observation, check_kv_capacity


class SyntheticCudaView:
    device = SimpleNamespace(type="cuda", index=0)
    dtype = torch.bfloat16
    shape = (16,)

    def __init__(self, address=8192, size=8192, offset=0):
        self.address, self.size, self.offset = address, size, offset

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.address, nbytes=lambda: self.size)

    def storage_offset(self):
        return self.offset

    def element_size(self):
        return 2

    def stride(self):
        return (1,)


@dataclass
class MambaSpec:
    block_size: int = 64
    mamba_cache_mode: str = "align"
    num_speculative_blocks: int = 0
    num_prefill_checkpoint_blocks: int = 0
    page_size_bytes: int = 1024

    def max_memory_usage_bytes(self, config):
        return 2048

    def max_num_blocks_per_req(self, config, max_len):
        return max_len // self.block_size


@dataclass
class Group:
    layer_names: list
    kv_cache_spec: MambaSpec


@dataclass
class Placement:
    size: int = 8192
    layers: tuple = ("recurrent",)
    layer_stride: int = 8192
    block_stride: int = 1024
    offset: int = 0


@dataclass
class Config:
    num_blocks: int
    kv_cache_groups: list
    kv_cache_tensors: list


@pytest.fixture
def worker():
    cache = SimpleNamespace(kv_cache_memory_bytes=8192, mamba_cache_mode="align")
    config = Config(8, [Group(["recurrent"], MambaSpec())], [Placement()])
    return SimpleNamespace(model_runner=SimpleNamespace(kv_cache_config=config,
                kv_caches=[SyntheticCudaView(), SyntheticCudaView(offset=8)], kernel_block_sizes=[64]),
            vllm_config=SimpleNamespace(cache_config=cache,
                model_config=SimpleNamespace(max_model_len=4096),
                scheduler_config=SimpleNamespace(max_num_seqs=3, max_num_batched_tokens=512),
                parallel_config=SimpleNamespace(tensor_parallel_size=1)))


@pytest.fixture
def expectations():
    return {"schema": "tessera.first_model_kv_capacity_expectations.v1",
        "group_kinds": ["recurrent"], "group_layer_counts": [1],
        "resident_pages_per_request_by_group": [2], "num_blocks": 8,
        "unique_physical_storage_bytes": 8192, "physical_pool_block_bytes": 1024,
        "max_model_len": 4096, "max_num_seqs": 3, "max_num_batched_tokens": 512,
        "tensor_parallel_size": 1, "page_size_bytes": 1024, "recurrent_cache_mode": "align",
        "num_speculative_blocks": 0, "num_prefill_checkpoint_blocks": 0, "null_blocks": 1}


def test_shared_kv_backing_is_counted_once_and_recurrent_residency_differs_from_indexing(worker, expectations):
    result = inspect_worker_kv(worker, expectations)
    assert result["storage"]["unique_physical_storage_bytes"] == 8192
    assert len(result["storage"]["views"]) == 2
    assert len(result["storage"]["storages"]) == 1
    assert result["group_details"][0]["resident_pages_per_request"] == 2
    assert result["group_details"][0]["block_table_entries_per_request"] == 64
    assert result["capacity_assertions"]["passed"]
    assert result["capacity_assertions"]["served_concurrency_verified"] is False
    assert result["received_argument_scope"] == "post-initialization runner observation"


@pytest.mark.parametrize("defect", ["double_storage", "wrong_spec", "too_many_requests", "auto_capacity", "wrong_pages", "wrong_groups"])
def test_capacity_disagreement_never_passes(worker, expectations, defect):
    result = inspect_worker_kv(worker)
    if defect == "double_storage":
        result["storage"]["unique_physical_storage_bytes"] *= 2
    elif defect == "wrong_spec":
        result["group_details"][0]["num_speculative_blocks"] = 1
    elif defect == "too_many_requests":
        result["resolved_limits"]["max_num_seqs"] = 4
    elif defect == "auto_capacity":
        result["capacity_policy"]["values"]["kv_cache_memory_bytes"] = None
    elif defect == "wrong_pages":
        result["group_details"][0]["resident_pages_per_request"] = 3
    elif defect == "wrong_groups":
        result["group_details"][0]["kind"] = "unsupported"
    assert check_kv_capacity(result, expectations)["passed"] is False


@pytest.mark.parametrize("second", [SyntheticCudaView(address=8192, size=4096),
                                    SyntheticCudaView(address=12288),
                                    SyntheticCudaView(offset=8192)])
def test_alias_size_overlap_and_invalid_view_are_refused(second):
    with pytest.raises(ValueError):
        kv_storage_observation([SyntheticCudaView(), second])


def test_inspection_does_not_mutate_worker_kv_state(worker, expectations):
    original = copy.deepcopy(worker.model_runner.kv_cache_config)
    inspect_worker_kv(worker, expectations)
    assert worker.model_runner.kv_cache_config == original
