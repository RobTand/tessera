"""tessera#777: the GLM MTP draft reads only the shards its own layers live in.

The fake loader keeps the stock control flow that matters here: ``_prepare_weights``
globs every shard in the folder and keeps the index's files, and
``get_all_weights`` streams every tensor of every returned shard. A shard is a
JSON list of tensor names, so a load records exactly which names it read.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import types
from types import SimpleNamespace as NS

import pytest

from tessera.serving import mtp_draft_shards as shards

INDEX = "model.safetensors.index.json"
RENAME = ("model.language_model.", "model.")


def spec_layer(config, name):
    """vLLM's get_spec_layer_idx_from_weight_name (glm5next/common/model.py)."""
    first = config.num_hidden_layers
    for i in range(config.num_nextn_predict_layers):
        if name.startswith(f"model.layers.{first + i}.") or name.startswith(f"layers.{first + i}."):
            return first + i
    return None


class Draft:
    def __init__(self):
        self.config = NS(num_hidden_layers=45, num_nextn_predict_layers=1)


class Target:
    config = NS(num_hidden_layers=45, num_nextn_predict_layers=1)


def make_loader():
    class Loader:
        def __init__(self):
            self.local_expert_ids = None
            self._encoder_only_lm_prefixes = None
            self.prepared: list[list[str]] = []

        def _prepare_weights(self, model_name_or_path, subfolder, revision, fall_back_to_pt,
                             allow_patterns_overrides):
            files = sorted(glob.glob(os.path.join(model_name_or_path, "*.safetensors")))
            index = json.load(open(os.path.join(model_name_or_path, INDEX)))
            files = [f for f in files if os.path.basename(f) in set(index["weight_map"].values())]
            return model_name_or_path, files, True, INDEX

        def get_all_weights(self, model_config, model):
            folder, files, _, _ = self._prepare_weights(model_config.model, None, None, True, None)
            self.prepared.append([os.path.basename(f) for f in files])
            for f in files:
                for name in json.load(open(f)):
                    yield name, f
    return Loader


# Shard -> tensor names. Shards b and c hold the spec layer (45) under both
# checkpoint spellings; a and d hold none of it.
CHECKPOINT = {
    "model-00001-of-00004.safetensors": ["model.embed_tokens.weight", "model.layers.0.mlp.w"],
    "model-00002-of-00004.safetensors": ["model.language_model.layers.45.mlp.w",
                                         "model.layers.44.mlp.w"],
    "model-00003-of-00004.safetensors": ["layers.45.shared_head.norm.weight",
                                         "layers.45.eh_proj.weight"],
    "model-00004-of-00004.safetensors": ["model.layers.44.attn.w", "lm_head.weight"],
}


@pytest.fixture
def ckpt(tmp_path):
    for shard, names in CHECKPOINT.items():
        (tmp_path / shard).write_text(json.dumps(names))
    weight_map = {name: shard for shard, names in CHECKPOINT.items() for name in names}
    (tmp_path / INDEX).write_text(json.dumps({"weight_map": weight_map}))
    return tmp_path


@pytest.fixture
def loader_cls():
    cls = make_loader()
    shards._install(cls, Draft, spec_layer, INDEX, RENAME)
    return cls


def spec_names(names):
    return {n for n in names if spec_layer(Draft().config, shards.renamed(n, RENAME)) is not None}


def test_draft_reads_only_the_spec_shards_and_every_spec_tensor(ckpt, loader_cls, caplog):
    stock = make_loader()()
    stock_names = [n for n, _ in stock.get_all_weights(NS(model=str(ckpt)), Draft())]
    loader = loader_cls()
    with caplog.at_level(logging.INFO, logger=shards.__name__):
        names = [n for n, _ in loader.get_all_weights(NS(model=str(ckpt)), Draft())]
    assert loader.prepared == [["model-00002-of-00004.safetensors",
                                "model-00003-of-00004.safetensors"]]
    # The renamed (model.language_model.) and bare (layers.) spellings both land.
    assert spec_names(names) == spec_names(stock_names) == {
        "model.language_model.layers.45.mlp.w", "layers.45.shared_head.norm.weight",
        "layers.45.eh_proj.weight"}
    assert "reads 2 of 4 checkpoint shards" in caplog.text
    assert "yielded 3 spec-layer tensors (index: 3)" in caplog.text
    assert getattr(loader, shards._PLAN) is None


def test_target_load_reads_every_shard(ckpt, loader_cls):
    loader = loader_cls()
    list(loader.get_all_weights(NS(model=str(ckpt)), Target()))
    assert loader.prepared == [sorted(CHECKPOINT)]


@pytest.mark.parametrize("why", ["ep", "encoder_only", "patterns", "secondary", "no_index",
                                 "no_spec", "not_local"])
def test_uninspected_loads_decline_to_every_shard(ckpt, loader_cls, caplog, why):
    loader, draft, model = loader_cls(), Draft(), str(ckpt)
    if why == "ep":
        loader.local_expert_ids = {0}
    elif why == "encoder_only":
        loader._encoder_only_lm_prefixes = ["model.language_model."]
    elif why == "patterns":
        draft.allow_patterns_overrides = ["*.safetensors"]
    elif why == "secondary":
        draft.secondary_weights = [object()]
    elif why == "no_index":
        os.rename(ckpt / INDEX, ckpt / "index.moved")
    elif why == "no_spec":
        draft.config = NS(num_hidden_layers=46, num_nextn_predict_layers=1)
    elif why == "not_local":
        model = "org/remote-model"
    if why in ("no_index", "not_local"):
        # The wrapper declines, then hands the load to the stock loader, which
        # (in this fake) needs the index and the folder itself and so raises.
        with caplog.at_level(logging.WARNING, logger=shards.__name__):
            with pytest.raises((OSError, ValueError)):
                list(loader.get_all_weights(NS(model=model), draft))
        assert "reads the whole checkpoint" in caplog.text
        assert getattr(loader, shards._PLAN, None) is None
        return
    with caplog.at_level(logging.WARNING, logger=shards.__name__):
        list(loader.get_all_weights(NS(model=model), draft))
    assert loader.prepared == [sorted(CHECKPOINT)]
    assert "reads the whole checkpoint" in caplog.text


def test_short_narrowed_load_fails_closed_after_exhaustion(ckpt, loader_cls):
    # The index claims a spec tensor that no shard holds.
    index = json.loads((ckpt / INDEX).read_text())
    index["weight_map"]["layers.45.ghost.weight"] = "model-00003-of-00004.safetensors"
    (ckpt / INDEX).write_text(json.dumps(index))
    loader = loader_cls()
    stream = loader.get_all_weights(NS(model=str(ckpt)), Draft())
    names = []
    with pytest.raises(RuntimeError, match="yielded 3 spec-layer tensors, the index lists 4"):
        for name, _ in stream:
            names.append(name)
    assert len(spec_names(names)) == 3  # every real tensor was yielded before the check
    assert getattr(loader, shards._PLAN) is None


def test_index_shard_missing_from_folder_fails_closed(ckpt, loader_cls):
    index = json.loads((ckpt / INDEX).read_text())
    index["weight_map"]["layers.45.extra.weight"] = "model-00009-of-00004.safetensors"
    (ckpt / INDEX).write_text(json.dumps(index))
    with pytest.raises(RuntimeError, match="did not find"):
        list(loader_cls().get_all_weights(NS(model=str(ckpt)), Draft()))


def test_plan_counts_from_the_index(ckpt):
    plan = shards.plan_draft_shards(ckpt / INDEX, Draft().config, spec_layer, RENAME)
    assert plan == shards.ShardPlan(frozenset({"model-00002-of-00004.safetensors",
                                               "model-00003-of-00004.safetensors"}), 3, 4)
    # Without the class's rename the language_model spelling is not a spec name.
    plan = shards.plan_draft_shards(ckpt / INDEX, Draft().config, spec_layer, None)
    assert plan is not None and plan.shards == {"model-00003-of-00004.safetensors"}


def _fake_vllm(monkeypatch, tmp_path, loader_source):
    loader_file = tmp_path / "default_loader.py"
    loader_file.write_text(loader_source)
    model_file = tmp_path / "model.py"
    model_file.write_text("# glm5next common model\n")
    loader_mod = types.ModuleType(shards._MODULES[0])
    loader_mod.__file__ = str(loader_file)
    loader_mod.DefaultModelLoader = make_loader()
    loader_mod.SAFE_WEIGHTS_INDEX_NAME = INDEX
    model_mod = types.ModuleType(shards._MODULES[1])
    model_mod.__file__ = str(model_file)
    model_mod.get_spec_layer_idx_from_weight_name = spec_layer
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.model_loader",
                 "vllm.models", "vllm.models.glm5next", "vllm.models.glm5next.common"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, shards._MODULES[0], loader_mod)
    monkeypatch.setitem(sys.modules, shards._MODULES[1], model_mod)
    return loader_mod, NS(Glm5NextMTP=Draft)


def test_unmatched_loader_sources_decline_without_wrapping(monkeypatch, tmp_path, caplog):
    loader_mod, glm = _fake_vllm(monkeypatch, tmp_path, "# not the inspected loader\n")
    stock = loader_mod.DefaultModelLoader.get_all_weights
    with caplog.at_level(logging.WARNING, logger=shards.__name__):
        shards.install("nightly-20260929", glm, RENAME)
    assert loader_mod.DefaultModelLoader.get_all_weights is stock
    assert not getattr(loader_mod.DefaultModelLoader, shards._INSTALLED, False)
    assert "reads the whole checkpoint" in caplog.text and "source identity" in caplog.text


def test_unknown_draft_interface_declines(monkeypatch, tmp_path, caplog):
    loader_mod, glm = _fake_vllm(monkeypatch, tmp_path, "# any\n")
    stock = loader_mod.DefaultModelLoader.get_all_weights
    with caplog.at_level(logging.WARNING, logger=shards.__name__):
        shards.install("fd4a15126", glm, None)
    assert loader_mod.DefaultModelLoader.get_all_weights is stock
    assert "no inspected loader interface for draft interface fd4a15126" in caplog.text


def test_matched_sources_install_once(monkeypatch, tmp_path):
    loader_mod, glm = _fake_vllm(monkeypatch, tmp_path, "# inspected\n")
    import hashlib
    digests = tuple(hashlib.sha256(open(sys.modules[m].__file__, "rb").read()).hexdigest()
                    for m in shards._MODULES)
    monkeypatch.setattr(shards, "_INTERFACES", (shards._ShardInterface("nightly-20260929", digests),))
    shards.install("nightly-20260929", glm, RENAME)
    wrapped = loader_mod.DefaultModelLoader.get_all_weights
    assert getattr(loader_mod.DefaultModelLoader, shards._INSTALLED)
    shards.install("nightly-20260929", glm, RENAME)  # idempotent
    assert loader_mod.DefaultModelLoader.get_all_weights is wrapped


def test_matched_sources_with_another_signature_fail_closed(monkeypatch, tmp_path):
    loader_mod, glm = _fake_vllm(monkeypatch, tmp_path, "# inspected\n")
    import hashlib
    digests = tuple(hashlib.sha256(open(sys.modules[m].__file__, "rb").read()).hexdigest()
                    for m in shards._MODULES)
    monkeypatch.setattr(shards, "_INTERFACES", (shards._ShardInterface("nightly-20260929", digests),))
    loader_mod.DefaultModelLoader.get_all_weights = lambda self, model_config, model, extra: None
    with pytest.raises(RuntimeError, match="unsupported loader signature"):
        shards.install("nightly-20260929", glm, RENAME)
