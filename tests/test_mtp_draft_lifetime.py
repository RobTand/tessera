"""CPU stand-in for the inspected GLM MTP construction/load/share boundaries.

Two interfaces: fd4a15126 (``glm5next.nvidia.mtp``; the draft builds its own
embed_tokens and per-layer heads) and the nightly-20260929 stack
(``glm5next.common.mtp``; ``SharedHead(defer_lm_head=True)`` builds no head,
so only embed_tokens is intercepted). tessera#749.

Exercises installation through the production quant-config entry, not a helper.
The allocator/profiler evidence is small CPU storage, not a GB10/image receipt.
"""
from __future__ import annotations

import concurrent.futures
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import weakref
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch

from test_serving_dispatch import runtime_modules as _runtime_modules

runtime_modules = _runtime_modules  # pytest discovers the isolated runtime fixture


NIGHTLY = {"interface": "nightly-20260929"}


@pytest.fixture
def draft_runtime(request, monkeypatch, runtime_modules):
    from tessera.serving import lane
    from tessera.serving.config import TesseraConfig

    params = getattr(request, "param", {})
    interface = params.get("interface", "fd4a15126")
    draft_heads = interface == "fd4a15126"
    glm_name, absent_glm = ("vllm.models.glm5next.nvidia.mtp", "vllm.models.glm5next.common.mtp")
    if not draft_heads:
        glm_name, absent_glm = absent_glm, glm_name

    lane.reset_for_tests()
    monkeypatch.setenv("TESSERA_SERVE_MODE", "resident")
    allocations = []
    world, rank = [1], [0]

    class Vocab(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, *, prefix="", quant_config=None):
            super().__init__()
            self.num_embeddings = self.org_vocab_size = num_embeddings
            self.embedding_dim = embedding_dim
            self.tp_size = world[0]
            self.tp_rank = rank[0]
            self.padding_size = 64
            self.params_dtype = torch.float32
            self.quant_method = sys.modules[
                "vllm.model_executor.layers.vocab_parallel_embedding"].UnquantizedEmbeddingMethod()
            self.weight = torch.nn.Parameter(torch.empty(num_embeddings // self.tp_size, embedding_dim),
                                             requires_grad=False)
            self.weight.data.fill_(3 if prefix.endswith("head") else 2)
            allocations.append((prefix, weakref.ref(self.weight)))

        def forward(self, x):
            return torch.nn.functional.embedding(x, self.weight)

    class Head(Vocab):
        pass

    glm: Any = types.ModuleType(glm_name)
    deepseek: Any = types.ModuleType("vllm.model_executor.models.deepseek_mtp")
    v1: Any = types.ModuleType("vllm.v1.spec_decode.llm_base_proposer")
    v2: Any = types.ModuleType("vllm.v1.worker.gpu.spec_decode.mtp.speculator")
    eagle: Any = types.ModuleType("vllm.v1.worker.gpu.spec_decode.eagle.utils")
    base_loader = types.ModuleType("vllm.model_executor.model_loader.base_loader")
    loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    for module in (glm, deepseek, v1, v2, eagle, base_loader, loader_utils):
        parts = module.__name__.split(".")
        for i in range(2, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    # The other interface's GLM module is absent, as on each real image.
    monkeypatch.setitem(sys.modules, absent_glm, None)
    glm.VocabParallelEmbedding = Vocab
    deepseek.ParallelLMHead = Head

    class Glm5NextMTP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.shared_head = torch.nn.Module()
            layer.shared_head.norm = torch.nn.Parameter(torch.empty(8))
            layer.shared_head.norm.data.fill_(5)
            # The nightly's SharedHead(defer_lm_head=True) builds no head.
            layer.shared_head.head = deepseek.ParallelLMHead(
                64, 8, prefix="model.layers.4.head", quant_config=config) if draft_heads else None
            self.model.layers = torch.nn.ModuleDict({"4": layer})
            self.model.embed_tokens = glm.VocabParallelEmbedding(
                64, 8, prefix="model.embed_tokens")

        def load_weights(self, weights):
            self.seen = []
            params = dict(self.named_parameters())
            for name, weight in weights:
                self.seen.append(name)
                # Stand-in for stock source-name rewrite then parameter copy.
                mapped = name.replace("model.language_model.", "model.")
                mapped = mapped.replace("model.layers.4.embed_tokens", "model.embed_tokens")
                params[mapped].data.copy_(weight[:params[mapped].shape[0]])
            return set(self.seen)

    glm.Glm5NextMTP = Glm5NextMTP
    config = TesseraConfig({}, (), {})
    speculative = NS(method="mtp", draft_model_config=NS(
        hf_config=NS(architectures=["Glm5NextMTPModel"], vocab_size=64, hidden_size=8,
                     num_hidden_layers=4, num_nextn_predict_layers=1),
        hf_text_config=NS(model_type="glm5_next_text"), dtype=torch.float32))
    current = NS(speculative_config=speculative, quant_config=config, lora_config=None)
    monkeypatch.setattr(sys.modules["vllm.config"], "get_current_vllm_config_or_none",
                        lambda: current)
    distributed = sys.modules["vllm.distributed"]
    monkeypatch.setattr(distributed, "get_pp_group", lambda: NS(world_size=1), raising=False)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_rank", lambda: rank[0], raising=False)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: world[0])
    vocab_module = sys.modules["vllm.model_executor.layers.vocab_parallel_embedding"]
    monkeypatch.setattr(vocab_module, "VocabParallelEmbedding", Vocab)
    monkeypatch.setattr(vocab_module, "ParallelLMHead", Head)

    target = torch.nn.Module()
    target.model = torch.nn.Module()
    target.model.embed_tokens = Vocab(64, 8, prefix="target.embed_tokens")
    target.lm_head = Head(64, 8, prefix="target.lm_head")
    before = [p.clone() for p in target.parameters()]
    snapshots = []

    def load(target_model):
        draft = glm.Glm5NextMTP()
        if during_load[0] is not None:
            during_load[0]()
        # A nightly draft has no head parameter; a head weight would KeyError
        # in its stock load_weights, so the nightly checkpoint carries none.
        draft.load_weights([
            ("model.language_model.layers.4.embed_tokens.weight", torch.zeros(64, 8)),
            *([("model.language_model.layers.4.shared_head.head.weight", torch.zeros(64, 8))]
              if draft_heads else []),
            ("model.language_model.layers.4.shared_head.norm", torch.full((8,), 7.0)),
        ])
        # A later load workspace overlaps vocab lifetime in the old code.
        workspace = torch.empty(512)
        workspace.fill_(1)
        snapshots.append(sum(p.numel() * p.element_size()
                             for prefix, ref in allocations if not prefix.startswith("target")
                             and (p := ref()) is not None))
        del workspace
        if fail[0]:
            raise ValueError("fixture load failure")
        draft.model.embed_tokens = target_model.model.embed_tokens
        draft.lm_head = target_model.lm_head
        draft.model.layers["4"].shared_head.head = target_model.lm_head
        if break_share[0]:
            draft.model.embed_tokens = torch.nn.Module()
        return draft

    fail, break_share, during_load = [False], [False], [None]

    class SpecDecodeBaseProposer:
        def __init__(self):
            self.vllm_config = current
        def load_model(self, target_model):
            self.model = load(target_model)

    class MTPSpeculator:
        def __init__(self):
            self.vllm_config = current
        def load_draft_model(self, target_model, target_attn_layer_names):
            return load(target_model)

    v1.SpecDecodeBaseProposer = SpecDecodeBaseProposer
    v2.MTPSpeculator = MTPSpeculator
    # The fixture models the inspected interface, not an installed vLLM image.
    # Production's source digest refusal is tested separately below.
    try:
        mtp_draft_lifetime = importlib.import_module("tessera.serving.mtp_draft_lifetime")
    except ImportError:
        pass  # RED must fail on the behavior, not a missing new module.
    else:
        if params.get("patch_sources", True):
            monkeypatch.setattr(mtp_draft_lifetime, "_require_supported_sources",
                                lambda *args: None)
    config.get_quant_method(torch.nn.Module(), "target.attention")
    yield NS(glm=glm, deepseek=deepseek, v1=v1, v2=v2, config=config, current=current,
             target=target, before=before, allocations=allocations, snapshots=snapshots,
             fail=fail, break_share=break_share, during_load=during_load,
             Vocab=Vocab, Head=Head, world=world, rank=rank, interface=interface,
             draft_heads=draft_heads)
    lane.reset_for_tests()


def _load(runtime, runner, *, legacy=False):
    if runner == "v1":
        proposer = runtime.v1.SpecDecodeBaseProposer()
        method = type(proposer).load_model
        if legacy:
            method = getattr(method, "__wrapped__", method)
        method(proposer, runtime.target)
        return proposer.model
    proposer = runtime.v2.MTPSpeculator()
    method = type(proposer).load_draft_model
    if legacy:
        method = getattr(method, "__wrapped__", method)
    return method(proposer, runtime.target, set())


def _owned_cpu_peak(trace):
    """Peak of CPU storages born in this interval, not net process memory.

    A prior profile can leave tensors for this interval's GC to free. Match
    frees to births by process/device/address; an unmatched free owns no bytes
    here. Remove freed addresses so later allocation at the same address is a
    new lifetime. Chrome's instant memory events retain the actual chronology,
    unlike aggregate function self-memory charged at function start.
    """
    storages = {}
    live, peak, prior_freed = 0, 0, 0
    events = (event for event in trace["traceEvents"] if event.get("name") == "[memory]")
    for event in sorted(events, key=lambda event: event["ts"]):
        args = event["args"]
        if args["Device Type"] != 0:  # Kineto's CPU device type.
            continue
        size = args["Bytes"]
        if not size:
            continue
        key = (event["pid"], args["Device Type"], args["Device Id"], args["Addr"])
        if size > 0:
            assert key not in storages, "CPU address allocated twice without a matched free"
            storages[key] = size
            live += size
        elif key in storages:
            owned_size = storages.pop(key)
            assert owned_size == -size, "CPU allocation/free sizes disagree"
            live -= owned_size
        else:
            prior_freed -= size
        peak = max(peak, live)
    return peak, prior_freed


def _profile_load(runtime, runner, *, legacy):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                profile_memory=True, acc_events=True) as profile:
        draft = _load(runtime, runner, legacy=legacy)
    events = profile.events() or []
    # Kineto can export a profile only once. Retain the full trace document for
    # accounting and optional receipts rather than trying to export it again.
    with tempfile.TemporaryDirectory(prefix="mtp-cpu-profile-") as directory:
        path = Path(directory) / "trace.json"
        profile.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
    peak, prior_freed = _owned_cpu_peak(trace)
    return draft, trace, {"incremental_cpu_peak_bytes": peak,
                          "ignored_preexisting_cpu_free_bytes": prior_freed,
                          "aten_empty_allocated_bytes": sum(event.cpu_memory_usage for event in events
                                                             if event.name == "aten::empty"),
                          "duplicate_live_at_workspace_bytes": runtime.snapshots[-1]}


@pytest.mark.parametrize("legacy, expected_peak", [(False, 4160), (True, 8256)])
def test_profile_peak_excludes_preexisting_tensor_garbage(draft_runtime, tmp_path, legacy, expected_peak):
    import gc
    import hashlib

    r = draft_runtime
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        # Prior profiled work can leave cyclic tensor garbage for a later interval.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    profile_memory=True, acc_events=True):
            tensor = torch.empty(126)  # 504 bytes, born before the measured interval.
            reference = weakref.ref(tensor)
            cycle: list[Any] = [tensor]
            cycle.append(cycle)
            del cycle, tensor
        assert reference() is not None
        r.during_load[0] = gc.collect
        draft, trace, evidence = _profile_load(r, "v1", legacy=legacy)
        assert reference() is None, "fixture did not free pre-existing tensor garbage"
        root = Path(os.environ.get("TESSERA_MTP_CPU_EVIDENCE", str(tmp_path)))
        root.mkdir(parents=True, exist_ok=True)
        trace_path = root / f"preexisting-garbage-{legacy}.trace.json"
        trace_path.write_text(json.dumps(trace))
        memory = [event for event in trace["traceEvents"] if event.get("name") == "[memory]"]
        print(json.dumps({"legacy": legacy, "measurement": evidence, "memory_events": memory,
                          "trace_path": str(trace_path),
                          "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()}))
        assert evidence["incremental_cpu_peak_bytes"] == expected_peak
        assert evidence["ignored_preexisting_cpu_free_bytes"] >= 504
        assert evidence["duplicate_live_at_workspace_bytes"] == (4096 if legacy else 0)
        assert draft.model.embed_tokens is r.target.model.embed_tokens
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("draft_runtime", [{}, NIGHTLY], indirect=True, ids=["fd4a15126", "nightly"])
@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize("world, rank", [(1, 0), (2, 0), (2, 1)])
def test_production_hook_avoids_duplicate_peak_and_preserves_forward(draft_runtime, runner, world, rank):
    r = draft_runtime
    r.world[0], r.rank[0] = world, rank
    r.target.model.embed_tokens = r.Vocab(64, 8, prefix="target.embed_tokens")
    r.target.lm_head = r.Head(64, 8, prefix="target.lm_head")
    r.before = [p.clone() for p in r.target.parameters()]
    # One 64x8 fp32 vocabulary is 2048 bytes. fd4a15126 builds a draft
    # embed_tokens and a per-layer head; the nightly defers the head upstream.
    duplicate_bytes = (4096 if r.draft_heads else 2048) // world
    legacy, before_trace, before = _profile_load(r, runner, legacy=True)
    assert before["duplicate_live_at_workspace_bytes"] == duplicate_bytes
    del legacy
    allocation_start = len(r.allocations)
    draft, after_trace, after = _profile_load(r, runner, legacy=False)
    evidence = {"interface": r.interface, "runner": runner, "before": before, "after": after,
                "device": "cpu", "vocab_shape": [64 // world, 8], "tp_size": world, "tp_rank": rank}
    print(json.dumps(evidence))
    if output := os.environ.get("TESSERA_MTP_CPU_EVIDENCE"):
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        label = f"{r.interface}-{runner}-tp{world}-rank{rank}"
        (root / f"{label}.json").write_text(json.dumps(evidence) + "\n")
        (root / f"{label}-before.trace.json").write_text(json.dumps(before_trace))
        (root / f"{label}-after.trace.json").write_text(json.dumps(after_trace))
    assert r.snapshots[-1] == 0, "draft vocabulary duplicates survive through load peak"
    assert before["incremental_cpu_peak_bytes"] - after["incremental_cpu_peak_bytes"] == duplicate_bytes
    assert not [prefix for prefix, ref in r.allocations[allocation_start:]
                if prefix.startswith("model")], "draft vocabulary storage was constructed before sharing"
    assert draft.model.embed_tokens is r.target.model.embed_tokens
    assert draft.lm_head is draft.model.layers["4"].shared_head.head is r.target.lm_head
    for before, after in zip(r.before, r.target.parameters(), strict=True):
        assert torch.equal(before, after), "draft load reinitialized target weights"
    assert draft.model.layers["4"].shared_head.norm.tolist() == [7] * 8
    assert draft.seen == ["model.language_model.layers.4.shared_head.norm"]
    ids = torch.tensor([1, 3])
    embeds = draft.model.embed_tokens(ids)
    assert torch.equal(embeds, r.target.model.embed_tokens(ids))
    assert torch.equal(embeds @ draft.lm_head.weight.T,
                       embeds @ r.target.lm_head.weight.T)
    assert draft.model.embed_tokens.weight.data_ptr() == r.target.model.embed_tokens.weight.data_ptr()


@pytest.mark.parametrize("draft_runtime", [{}, NIGHTLY], indirect=True, ids=["fd4a15126", "nightly"])
@pytest.mark.parametrize("runner", ["v1", "v2"])
def test_exception_restores_context_and_unrelated_thread(draft_runtime, runner):
    r = draft_runtime
    assert getattr(r.glm.VocabParallelEmbedding, "_tessera_mtp_constructor", False), \
        "production hook not installed"
    r.fail[0] = True
    with pytest.raises(ValueError, match="fixture load failure"):
        _load(r, runner)
    r.fail[0] = False
    # Normal constructor behavior, class predicates and another thread stay real.
    vocab = r.glm.VocabParallelEmbedding(64, 8, prefix="model.embed_tokens")
    assert type(vocab) is r.Vocab
    assert isinstance(vocab, r.glm.VocabParallelEmbedding)
    assert issubclass(r.Vocab, r.glm.VocabParallelEmbedding)
    others = []
    def concurrent_constructor():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            others.append(pool.submit(r.glm.VocabParallelEmbedding, 64, 8,
                                      prefix="model.embed_tokens").result())
    r.during_load[0] = concurrent_constructor
    _load(r, runner)
    assert others[0].weight.device.type == "cpu" and others[0].weight.numel() == 512
    old_binding = r.glm.VocabParallelEmbedding
    r.config.get_quant_method(torch.nn.Module(), "target.attention")
    assert r.glm.VocabParallelEmbedding is old_binding


@pytest.mark.parametrize("draft_runtime", [{}, NIGHTLY], indirect=True, ids=["fd4a15126", "nightly"])
@pytest.mark.parametrize("unsupported", ["shape", "dtype", "layout", "quantized", "lora", "pp"])
def test_unsupported_target_refuses_before_draft_allocation(draft_runtime, monkeypatch, unsupported):
    r = draft_runtime
    # The placeholder contract covers only the vocabularies an interface
    # intercepts: the head on fd4a15126, embed_tokens on the nightly.
    vocab = r.target.lm_head if r.draft_heads else r.target.model.embed_tokens
    if unsupported == "shape":
        vocab.weight = torch.nn.Parameter(torch.empty(32, 8), requires_grad=False)
    elif unsupported == "dtype":
        vocab.weight = torch.nn.Parameter(torch.empty(64, 8, dtype=torch.float16), requires_grad=False)
    elif unsupported == "layout":
        vocab.weight = torch.nn.Parameter(torch.empty(8, 64).T, requires_grad=False)
    elif unsupported == "quantized":
        vocab.quant_method = object()
    elif unsupported == "lora":
        r.current.lora_config = object()
    else:
        monkeypatch.setattr(sys.modules["vllm.distributed"], "get_pp_group", lambda: NS(world_size=2))
    count = len(r.allocations)
    with pytest.raises(RuntimeError, match="Tessera MTP"):
        _load(r, "v2")
    assert len(r.allocations) == count


@pytest.mark.parametrize("draft_runtime", [{}, NIGHTLY], indirect=True, ids=["fd4a15126", "nightly"])
def test_sharing_verification_refuses_and_restores_context(draft_runtime):
    r = draft_runtime
    r.break_share[0] = True
    with pytest.raises(RuntimeError, match="Tessera MTP.*sharing"):
        _load(r, "v2")
    assert r.glm.VocabParallelEmbedding(64, 8, prefix="model.embed_tokens").weight.numel() == 512


def test_non_mtp_load_uses_stock_constructors(draft_runtime):
    r = draft_runtime
    assert getattr(r.glm.VocabParallelEmbedding, "_tessera_mtp_constructor", False), \
        "production hook not installed"
    r.current.speculative_config.method = "eagle"
    draft = _load(r, "v2")
    assert r.snapshots[-1] == 4096
    assert len(draft.seen) == 3


def test_source_identity_is_fail_closed(draft_runtime, tmp_path):
    r = draft_runtime
    assert getattr(r.glm.VocabParallelEmbedding, "_tessera_mtp_constructor", False), \
        "production hook not installed"
    lifetime = importlib.import_module("tessera.serving.mtp_draft_lifetime")
    source = tmp_path / "mtp.py"
    source.write_text("# inspected runtime source\n")
    r.glm.__file__ = str(source)
    with pytest.raises(RuntimeError, match="Tessera MTP.*source identity"):
        lifetime.require_source_digest(r.glm, "0" * 64)


def test_nested_unrelated_load_restores_outer_interception(draft_runtime):
    r = draft_runtime
    assert getattr(r.glm.VocabParallelEmbedding, "_tessera_mtp_constructor", False), \
        "production hook not installed"
    nested = []
    def unrelated_load():
        r.during_load[0] = None
        r.current.speculative_config.method = "eagle"
        try:
            nested.append(_load(r, "v2"))
        finally:
            r.current.speculative_config.method = "mtp"
    r.during_load[0] = unrelated_load
    outer = _load(r, "v2")
    assert len(nested[0].seen) == 3
    assert r.snapshots[-1] == 0
    assert outer.model.embed_tokens is r.target.model.embed_tokens


@pytest.mark.parametrize("draft_runtime", [{}, NIGHTLY], indirect=True, ids=["fd4a15126", "nightly"])
def test_unsupported_signature_refuses_without_partial_install(draft_runtime, monkeypatch):
    r = draft_runtime
    assert getattr(r.glm.VocabParallelEmbedding, "_tessera_mtp_constructor", False), \
        "production hook not installed"
    binding = r.glm.VocabParallelEmbedding
    monkeypatch.delattr(r.glm, "_tessera_mtp_lifetime")
    monkeypatch.setattr(r.v2.MTPSpeculator, "load_draft_model", lambda self, unknown: None)
    with pytest.raises(RuntimeError, match="Tessera MTP.*signature"):
        r.config.get_quant_method(torch.nn.Module(), "target.attention")
    assert r.glm.VocabParallelEmbedding is binding


@pytest.mark.parametrize("draft_runtime", [NIGHTLY], indirect=True, ids=["nightly"])
def test_nightly_leaves_the_target_head_to_stock(draft_runtime):
    """The nightly builds no draft head, so an unusual target head is not ours to refuse."""
    r = draft_runtime
    r.target.lm_head.quant_method = object()
    draft = _load(r, "v2")
    assert r.snapshots[-1] == 0
    assert draft.model.embed_tokens is r.target.model.embed_tokens
    assert draft.model.layers["4"].shared_head.head is r.target.lm_head
    assert not getattr(r.deepseek.ParallelLMHead, "_tessera_mtp_constructor", False)


class _TesseraHeadMethod:
    """Stands in for the method ``head_route`` builds: a Tessera route's class."""


_TesseraHeadMethod.__module__ = "tessera.serving.fp8_route"


def _tessera_head(r, rows=64, cols=8):
    """The target head as ``head_route`` leaves it after preparation: a Tessera
    method, a prepared module and a row scale, and no ``weight``."""
    from tessera.serving.scheme import TESSERA_FP8

    head = r.Head(rows, cols, prefix="target.lm_head")
    del head.weight
    head.quant_method = _TesseraHeadMethod()
    head.tessera_family = TESSERA_FP8
    head.tessera_native = object()
    head.tessera_rows = rows // r.world[0]
    head.tessera_columns = cols
    head.register_buffer("scale_b", torch.ones(1, rows // r.world[0]), persistent=False)
    return head


@pytest.mark.parametrize("world, rank", [(1, 0), (2, 1)])
def test_a_declared_tessera_head_is_shared_on_fd4a15126(draft_runtime, world, rank):
    """tessera#750 WP3: a checkpoint may declare its LM head as a Tessera wire.

    On fd4a15126 the draft builds a per-layer head that stock sharing replaces
    with the target's head object, so the placeholder contract reads the
    target head's vocabulary geometry.  A Tessera head has no ``weight``; its
    prepared rows and columns are the geometry, and the draft still allocates
    no vocabulary of its own.
    """
    r = draft_runtime
    r.world[0], r.rank[0] = world, rank
    r.target.model.embed_tokens = r.Vocab(64, 8, prefix="target.embed_tokens")
    r.target.lm_head = _tessera_head(r)
    draft = _load(r, "v2")
    assert r.snapshots[-1] == 0
    assert draft.lm_head is draft.model.layers["4"].shared_head.head is r.target.lm_head
    assert draft.model.embed_tokens is r.target.model.embed_tokens


@pytest.mark.parametrize("broken", ["rows", "columns", "unprepared", "family", "bias"])
def test_a_tessera_head_that_does_not_fit_the_vocabulary_is_refused(draft_runtime, broken):
    r = draft_runtime
    head = _tessera_head(r)
    if broken == "rows":
        head.tessera_rows = 32
    elif broken == "columns":
        head.tessera_columns = 16
    elif broken == "unprepared":
        head.tessera_native = None
    elif broken == "family":
        head.tessera_family = "TESSERA_NVFP4"
    else:
        head.bias = torch.nn.Parameter(torch.zeros(64), requires_grad=False)
    r.target.lm_head = head
    count = len(r.allocations)
    with pytest.raises(RuntimeError, match="Tessera MTP"):
        _load(r, "v2")
    assert len(r.allocations) == count


@pytest.mark.parametrize("draft_runtime", [{"patch_sources": False}, {**NIGHTLY, "patch_sources": False}],
                         indirect=True, ids=["fd4a15126", "nightly"])
def test_unmatched_sources_decline_to_stock_load(draft_runtime, caplog):
    """tessera#749: no inspected interface matches, so the serve loads stock and says so."""
    import logging

    r = draft_runtime
    lifetime = importlib.import_module("tessera.serving.mtp_draft_lifetime")
    # The fixture's construction-time get_quant_method already declined.
    assert not getattr(r.glm.VocabParallelEmbedding, "_tessera_mtp_constructor", False)
    assert not getattr(r.glm, "_tessera_mtp_lifetime", False)
    assert lifetime._RESOLVED[1] is None
    with caplog.at_level(logging.WARNING, logger=lifetime.__name__):
        for _ in range(3):
            r.config.get_quant_method(torch.nn.Module(), "target.attention")
    # Resolved once per module set: no per-layer re-hash, no repeated warning.
    assert not [rec for rec in caplog.records if "saving absent" in rec.getMessage()]
    draft = _load(r, "v2")
    assert r.snapshots[-1] == (4096 if r.draft_heads else 2048)
    assert draft.model.embed_tokens is r.target.model.embed_tokens


def test_decline_is_logged_with_each_interface_reason(caplog, monkeypatch):
    import logging

    lifetime = importlib.import_module("tessera.serving.mtp_draft_lifetime")
    # Every candidate absent, whatever vLLM this interpreter carries.
    for name in (*(i.glm_module for i in lifetime._INTERFACES), *lifetime._COMMON_MODULES):
        monkeypatch.setitem(sys.modules, name, None)
    saved = list(lifetime._RESOLVED)
    lifetime._RESOLVED[:] = [None, None]
    try:
        with caplog.at_level(logging.WARNING, logger=lifetime.__name__):
            assert lifetime._supported_interface() is None
    finally:
        lifetime._RESOLVED[:] = saved
    messages = [rec.getMessage() for rec in caplog.records if "saving absent" in rec.getMessage()]
    assert len(messages) == 1 and "tessera#749" in messages[0]
    for interface in lifetime._INTERFACES:
        assert f"{interface.name}: not importable: {interface.glm_module}" in messages[0]


def test_candidate_import_crash_is_a_non_match(monkeypatch):
    """A module that raises at import (not ImportError) declines instead of failing the serve."""
    lifetime = importlib.import_module("tessera.serving.mtp_draft_lifetime")

    def boom(name, *args, **kwargs):
        raise RuntimeError(f"ops registration failed in {name}")
    monkeypatch.setattr(lifetime.importlib, "import_module", boom)
    assert lifetime._import("vllm.models.glm5next.nvidia.mtp") is None
    assert "RuntimeError" in lifetime._IMPORT_ERRORS["vllm.models.glm5next.nvidia.mtp"]


def test_interface_table_pins_every_touched_module():
    lifetime = importlib.import_module("tessera.serving.mtp_draft_lifetime")
    names = [interface.name for interface in lifetime._INTERFACES]
    assert names == ["fd4a15126", "nightly-20260929"]
    for interface in lifetime._INTERFACES:
        assert len(interface.digests) == 1 + len(lifetime._COMMON_MODULES)
        assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest in interface.digests)
    # Distinct interfaces are distinct sources; one digest set cannot match both.
    assert len({interface.digests for interface in lifetime._INTERFACES}) == len(names)


@pytest.mark.parametrize("draft_runtime", [{}, NIGHTLY, {"patch_sources": False}], indirect=True,
                         ids=["fd4a15126", "nightly", "declined"])
def test_draft_load_rename_is_the_recognized_interfaces(draft_runtime):
    """tessera#749: only the nightly's source renames the checkpoint in load_weights."""
    lifetime = importlib.import_module("tessera.serving.mtp_draft_lifetime")
    expected = {"fd4a15126": None, "nightly-20260929": ("model.language_model.", "model.")}
    if lifetime._RESOLVED[1] is None:
        assert lifetime.draft_load_rename() is None
    else:
        assert lifetime.draft_load_rename() == expected[draft_runtime.interface]


@pytest.mark.parametrize("draft_runtime", [{}, NIGHTLY, {"patch_sources": False}], indirect=True,
                         ids=["fd4a15126", "nightly", "declined"])
def test_recognized_draft_interface_names_the_class_module_and_rename(draft_runtime):
    """tessera#769: the route census accepts a stock draft outside the eugr module
    only through the interface recognized by source digest."""
    lifetime = importlib.import_module("tessera.serving.mtp_draft_lifetime")
    expected = {"fd4a15126": ("vllm.models.glm5next.nvidia.mtp", None),
                "nightly-20260929": ("vllm.models.glm5next.common.mtp",
                                     ("model.language_model.", "model."))}
    if lifetime._RESOLVED[1] is None:
        assert lifetime.recognized_draft_interface() is None
    else:
        assert lifetime.recognized_draft_interface() == expected[draft_runtime.interface]
