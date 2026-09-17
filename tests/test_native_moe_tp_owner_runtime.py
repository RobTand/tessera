"""The whole GLM routed owner's RUNTIME path, on CPU: config, route and world.

CPU only, and deliberately so: this file fixes what the harness RESOLVES --
the family/grid/rung its format names, the sidecar the loader must accept, the
execution record, the rank-local route geometry, and the world the owner binds
to -- before any device exists.  What it does not exercise is the CUDA layer
construction and the native decode; those are named where the substitution is
installed (``_stock_config_runtime``) and are the GPU qualification step, not
an assertion here.

No shape-only expectation lives here: every resolver is the harness's own, the
sidecar goes through the plugin's real ``validate_tessera_moe_scheme``, and the
binding reads a real ``torch.distributed`` group when one exists.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from experiments import bench_native_moe_operator as moe
from experiments import bench_native_operator as dense

GLM_UNIT = "model.language_model.layers.3.mlp.experts"
A4 = "TESSERA_E2M1x2_K2_R896"
A8 = "TESSERA_E4M3_K1_R1024"
A16 = "TESSERA_BF16_K1_R1024"


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _glm_shape(tp, format_name, unit=GLM_UNIT):
    return {"geometry_version": 1, "geometry_id": "glm53_next_routed_stack_v1",
            "source_id": "glm5_next", "n_routed_experts": 288, "top_k": 8,
            "hidden_size": 4096, "intermediate_size": 2048, "shared_experts": 1,
            "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc",
            "scoring_func": "sigmoid", "norm_topk_prob": True,
            "routed_scaling_factor": 2.5, "swiglu_limit": 10.0, "gated": True,
            "tensor_parallel": tp, "tensor_parallel_cut_axis": "intermediate",
            "format": format_name}


def _glm_routing():
    return {"activation": "silu", "scoring_func": "sigmoid", "renormalize": True,
            "routed_scaling_factor": 2.5, "apply_router_weight_on_input": False,
            "expert_map": None, "input_dtype": "torch.bfloat16",
            "topk_weights_dtype": "torch.float32", "topk_ids_dtype": "torch.int32",
            "device": "cuda:0",
            "weights_contract": "post_renormalization_and_routed_scaling",
            "swiglu_limit": 10.0, "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc",
            "source_protocol": {"router_class": "Glm5NextTopKRouter",
                "router_source_sha256": _sha("CPU source fixture"),
                "scoring_func": "sigmoid", "topk_method": "noaux_tc",
                "normalization_epsilon": 1e-6,
                "correction_bias": {"content_sha256": _sha("CPU bias fixture"),
                                    "dtype": "torch.float32"},
                "expert_bias_affects": "selection_only", "norm_topk_prob": True}}


def _pq_bias_block(bias):
    """PrismaQuant's GLM spelling: the FP32 bias's dtype and content digest."""
    record = dense.tensor_identity(bias)
    return {"dtype": record["dtype"], "content_sha256": record["content_sha256"]}


def _routing_with_correction_bias(bias, block=None):
    routing = _glm_routing()
    routing["source_protocol"]["correction_bias"] = block or _pq_bias_block(bias)
    return routing


def test_the_glm_correction_bias_block_is_checked_against_the_actual_payload():
    """PQ's identity pair reaches the payload check, and is what refuses.

    The cross-repository fact is that the two sides spell one captured object
    in two grammars: PQ stores ``{content_sha256, dtype}`` and this receipt
    checks the tensor it was handed.  The translation is therefore a CHECK --
    the declared dtype and digest must be the supplied bytes -- and nothing
    here rebuilds a record from the declaration.
    """
    bias = torch.zeros(288, dtype=torch.float32)
    routing = _routing_with_correction_bias(bias)
    assert set(routing["source_protocol"]["correction_bias"]) == {"content_sha256", "dtype"}
    assert moe.verify_routing_bias(routing, bias) is bias

    # Another tensor's digest is a different mixture, refused by name.
    with pytest.raises(ValueError, match="not the captured source's bytes"):
        moe.verify_routing_bias(routing, torch.ones(288, dtype=torch.float32))
    # A protocol that declares one dtype and a payload that is another.
    with pytest.raises(ValueError, match="must be the source's"):
        moe.verify_routing_bias(
            _routing_with_correction_bias(bias, {"dtype": "torch.bfloat16",
                                                 "content_sha256": _sha("elsewhere")}),
            bias)
    # The bias is the captured source's expert count, not the LFM 32.
    with pytest.raises(ValueError, match="must be the source's"):
        moe.verify_routing_bias(routing, torch.zeros(32, dtype=torch.float32))
    # And a declared bias with no payload is a refusal, not an inference.
    with pytest.raises(ValueError):
        moe.verify_routing_bias(routing, None)


def test_the_request_roster_carries_the_geometrys_own_bias_payload():
    """The roster rule reads both spellings, so a GLM request needs its bias."""
    members = [{"unit": "unit.0.w1"}, {"unit": "unit.0.w3"}, {"unit": "unit.0.w2"}]
    bias = torch.zeros(288, dtype=torch.float32)
    glm_roster = moe.request_tensor_roster(_routing_with_correction_bias(bias), members)
    assert "routing_bias" in glm_roster
    assert {f"prefill.{key}" for key in moe.TENSOR_KEYS} <= glm_roster
    assert {"source_weight/unit.0.w1", "rendered_weight/unit.0.w2"} <= glm_roster
    # The LFM spelling keeps its own answer, and a protocol that declares no
    # bias requires no payload.
    lfm = {"source_protocol": {"selection_bias": dense.tensor_identity(
        torch.zeros(32, dtype=torch.float32))}}
    assert "routing_bias" in moe.request_tensor_roster(lfm, members)
    assert "routing_bias" not in moe.request_tensor_roster(
        {"source_protocol": {"selection_bias": None}}, members)


def _routes(*families):
    from tessera.serving.scheme import ROUTES
    return {family: ROUTES[family] for family in families}


@pytest.mark.parametrize("format_name,family,grid,q256", [
    (A4, "TESSERA_NVFP4", "E2M1x2", 896),
    (A8, "TESSERA_FP8", "E4M3", 1024),
    (A16, "TESSERA_BF16", "BF16", 1024),
])
def test_the_owner_format_names_its_family_grid_and_rung(format_name, family, grid, q256):
    """The family is the ROUTE the grid resolves to, not a name carved off."""
    wire = moe.owner_wire(_glm_shape(1, format_name))
    route = _routes(family)[family]
    assert (wire["family"], wire["grid"], wire["q256"]) == (family, grid, q256)
    assert (wire["body"], wire["plane"]) == (route["body"], route["plane"])
    assert wire["activation_contract"] == route["activation_contract"]
    assert wire["policy"] == family + ":resident"
    # An A4 owner is not an A8 owner: family, rung and tile all differ.
    assert moe.owner_wire(_glm_shape(1, A4)) != moe.owner_wire(_glm_shape(1, A8))


def test_an_unknown_or_unparseable_format_is_refused():
    with pytest.raises(ValueError):
        moe.owner_wire(_glm_shape(1, "TESSERA_NOTAGRID_K1_R1024"))
    with pytest.raises(ValueError, match="Tessera format name"):
        moe.owner_wire(_glm_shape(1, "E4M3_K1_R1024"))


@pytest.mark.parametrize("format_name,family", [(A4, "TESSERA_NVFP4"), (A8, "TESSERA_FP8"),
                                                (A16, "TESSERA_BF16")])
@pytest.mark.parametrize("tp", [1, 2])
def test_the_sidecar_the_loader_gets_is_the_owners_own_recipe(format_name, family, tp):
    """The real plugin validator accepts the real scheme at the owner's rung.

    ``wire_stride`` is this rank's container width, so it is built from the
    rank-local member rows; the geometry the sidecar declares is the MODULE's,
    because a Tessera checkpoint is tensor-parallel agnostic.
    """
    from tessera.serving.scheme import validate_tessera_moe_scheme
    shape = _glm_shape(tp, format_name)
    wire = moe.owner_wire(shape)
    strides = {"w13": 4215596 if family == "TESSERA_FP8" else 4231984,
               "w2": 4219628 if family == "TESSERA_FP8" else 4236016}
    scheme = moe.owner_scheme(shape, wire, strides=strides, unit=GLM_UNIT)
    assert (scheme["family"], scheme["grid"]) == (family, wire["grid"])
    assert scheme["hidden_size"] == 4096
    # The sidecar describes the whole module at every world: the rank-local cut
    # is the loader's, and ``create_weights`` refuses a partition width that is
    # not exactly ``intermediate_size // tp``.
    assert scheme["intermediate_size"] == 2048
    assert scheme["groups"]["w13"]["rows"] == 2 * 2048
    assert [rows for _name, rows in scheme["groups"]["w13"]["roles"]] == [2048, 2048]
    assert [group["q256"] for group in scheme["groups"].values()] == [wire["q256"]] * 2
    # Feeding it back through the plugin's own front door accepts it too, which
    # is what stops this test from asserting a sidecar the loader would refuse.
    assert validate_tessera_moe_scheme(
        {key: value for key, value in scheme.items() if key != "structure"},
        GLM_UNIT)["family"] == family


@pytest.mark.parametrize("tp", [1, 2])
def test_the_rank_local_member_rows_are_the_cut_and_the_container_rows_are_not(tp):
    shape = _glm_shape(tp, A8)
    for role in moe.ROLE_ORDER:
        rows = moe._member_shape(shape, role)[0]
        declared = moe._declared_member_rows(shape, role)
        if role == "w2":
            # The down projection is cut along its COLUMNS, so its row count is
            # the hidden size at every world and the container's framing is the
            # same number this rank's tensor has.
            assert (rows, declared) == (4096, 4096)
        else:
            assert declared == 2048
            assert rows == 2048 // tp
            assert (rows == declared) is (tp == 1)


@pytest.mark.parametrize("tp", [1, 2])
def test_the_route_record_geometry_is_the_ranks_own_gate_up_stack(tp):
    """N is this rank's ``2 * N_local`` and K the hidden size, which is what
    ``moe_route.create_weights`` stores as ``tessera_rows``/``tessera_columns``."""
    assert moe.owner_route_shape(_glm_shape(tp, A8)) == f"N{2 * 2048 // tp}:K4096"


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
def test_the_member_map_names_the_exact_slice_this_rank_loads(tp, rank):
    """A unit name is not enough: a consumer needs the range, not the module.

    The map is read off the same plan the loader cuts with, so it cannot
    describe a cut nobody makes, and it is what a consumer joins to the panel's
    own statement of this rank's member shapes.
    """
    shape = _glm_shape(tp, A8)
    wire = moe.owner_wire(shape)
    scheme = moe.owner_scheme(shape, wire, strides={"w13": 4215596, "w2": 4219628},
                              unit=GLM_UNIT)
    mapping = moe.owner_member_map(shape, scheme, unit=GLM_UNIT, rank=rank, world=tp)
    local = 2048 // tp
    # At a world of one there is nothing to cut: the plan's axis is None and the
    # "slice" is the module itself, which is why the TP1 receipt is unchanged.
    assert mapping["w1"]["axis"] == ("row" if tp > 1 else None)
    assert mapping["w1"]["container_shape"] == [2048, 4096]
    assert mapping["w1"]["rank_local_shape"] == [local, 4096]
    assert [mapping["w1"]["shard_lo"], mapping["w1"]["shard_hi"]] == [rank * local,
                                                                      (rank + 1) * local]
    assert mapping["w1"]["shards"] == tp
    assert mapping["w3"]["rank_local_shape"] == mapping["w1"]["rank_local_shape"]
    assert mapping["w2"]["axis"] == ("column" if tp > 1 else None)
    assert mapping["w2"]["container_shape"] == [4096, 2048]
    assert mapping["w2"]["rank_local_shape"] == [4096, local]
    assert {entry["rank"] for entry in mapping.values()} == {rank}
    assert {entry["world_size"] for entry in mapping.values()} == {tp}
    # Every mapped role lands on exactly the rank-local member this rank's PWC
    # holds, which is the join the consumer needs.
    for role, entry in mapping.items():
        assert entry["rank_local_shape"] == moe._member_shape(shape, role)


def test_a_single_rank_resource_claim_carries_its_own_identity_and_no_peers():
    """At a world of one the per-rank bound IS the whole-owner bound."""
    receipt = {"resources": {"status": "complete_operator_bound", "scope": "torch_allocator_observation"}}
    identity = moe.per_rank_resource_identity(receipt, {"world_size": 1, "rank": 0})
    assert identity["peers"] == []
    assert identity["self"] == {"rank": 0, "world_size": 1,
                                "bound_sha256": dense.identity_sha256(receipt["resources"])}


def test_the_collective_is_counted_at_the_runner_callsite_or_refused():
    """The receipt's collective claim rests on a count, not on a config flag.

    The callsite is the name the runtime's own runner module imports, and the
    site it names is verified against the pinned image's
    ``moe_runner.py:15`` (the import) and ``:477`` (the reduction).  Where that
    module is not importable the probe refuses rather than reporting "nothing
    ran", because a silent zero is exactly how a config-only claim would look.
    """
    assert moe.RUNTIME_COLLECTIVE_MODULE.endswith("fused_moe.runner.moe_runner")
    # The site is the module the probe wraps, plus the method it counts: one
    # definition, so the name a receipt prints cannot drift from the object
    # whose calls are counted.
    assert moe.RUNTIME_COLLECTIVE_SITE == (
        "vllm.model_executor.layers.fused_moe.runner.moe_runner"
        ":_maybe_reduce_final_output")
    assert moe.RUNTIME_COLLECTIVE_SITE == (
        f"{moe.RUNTIME_COLLECTIVE_MODULE}:{moe.RUNTIME_COLLECTIVE_METHOD}")
    try:
        import vllm.model_executor.layers.fused_moe.runner.moe_runner  # noqa: F401
    except ImportError:
        with pytest.raises((ValueError, ImportError, ModuleNotFoundError)):
            with moe.observe_output_collective():
                pass
        return
    pytest.skip("the runtime is importable here; the device lane counts the real call")


def _fake_owner(*, world, runner=None):
    from types import SimpleNamespace

    class _Layer:
        pass

    layer = _Layer()
    layer.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(tp_size=world))
    return moe.WholeOwner(layer=layer, runner=runner)


def test_a_routed_partial_reaches_the_runtimes_own_reduction_or_refuses():
    """The expert method returns routed output only; the runner reduces it.

    A TP owner priced through the method alone is this rank's partial sum
    wearing the module's name, so the seam is called, and a runtime that no
    longer publishes it is a refusal rather than a harness reimplementation.
    """
    reached = []
    runner = type("R", (), {"_maybe_reduce_final_output":
                            lambda self, hidden, trunc: (reached.append(trunc), hidden * 2)[1]})()
    partial = torch.ones(2, 3, dtype=torch.bfloat16)
    # At a world of one nothing is reduced and the tensor is returned as-is.
    single = _fake_owner(world=1, runner=runner)
    assert moe.reduce_routed_output(single, partial) is partial
    assert reached == []
    # Above one the runtime's own seam is invoked on the routed partial.
    split = _fake_owner(world=2, runner=runner)
    out = moe.reduce_routed_output(split, partial)
    assert reached == [None] and out is not partial and bool((out == 2).all())
    # A runtime with no such seam refuses; it does not silently return half.
    with pytest.raises(ValueError, match="late all-reduce"):
        moe.reduce_routed_output(_fake_owner(world=2), partial)


def _torch_owner(*, register_parent=False):
    """A real parent/child module pair, in the runtime's own shape."""
    import torch.nn as nn

    class _Child(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    layer = _Child()
    runner = nn.Module()
    runner.routed_experts = layer
    if register_parent:
        # The defect root caught: an nn.Module attribute is a REGISTERED child,
        # so the runner would hang inside the layer it owns.
        layer.tessera_runner = runner
    return layer, runner


def test_the_runner_is_kept_beside_the_layer_that_it_owns():
    """The runner owns the layer; the layer never registers its parent."""
    import torch
    layer, runner = _torch_owner()
    owner = moe.WholeOwner(layer=layer, runner=runner)
    moe.verify_owner_topology(owner)
    assert list(layer.modules()) == [layer]
    assert layer._modules == {}
    assert set(layer.state_dict()) == {"weight"}
    # .to() over the layer must not walk back up into the runner.
    moved = layer.to(torch.device("cpu"))
    assert list(moved.modules()) == [moved]


def test_a_runner_registered_inside_its_own_layer_is_refused():
    """The cycle root reviewed: .to()/.state_dict() would recurse through it."""
    layer, runner = _torch_owner(register_parent=True)
    assert runner in list(layer.modules())
    with pytest.raises(ValueError, match="registered inside the layer it owns"):
        moe.verify_owner_topology(moe.WholeOwner(layer=layer, runner=runner))


def test_the_owner_releases_the_runner_when_it_is_dropped():
    """No back-reference keeps the pair alive: cleanup is the holder's."""
    import gc
    import weakref

    layer, runner = _torch_owner()
    owner = moe.WholeOwner(layer=layer, runner=runner)
    registry = weakref.ref(runner)
    del owner, runner, layer
    gc.collect()
    assert registry() is None


@pytest.mark.parametrize("tp", [1, 2])
def test_the_execution_record_is_the_owners_own_cut(tp):
    shape = _glm_shape(tp, A8)
    execution = moe.owner_execution(shape)
    assert execution["tensor_parallel"] == tp
    assert moe.validate_execution(execution, list(moe.ROLE_ORDER), shape=shape) is None
    if tp == 2:
        with pytest.raises(ValueError, match="tensor-parallel"):
            moe.validate_execution(dict(moe.EXECUTION), list(moe.ROLE_ORDER), shape=shape)
    # An LFM owner's record is untouched: the same dict, field for field.
    lfm = {"experts": 32, "hidden_size": 256, "intermediate_size": 128, "top_k": 4}
    assert moe.owner_execution(lfm) == moe.EXECUTION


def _stock_config_runtime(monkeypatch, *, tensor_parallel):
    """CPU substitution at the stock config-construction boundary.

    Named, and only this boundary: the object the engine builds from
    ``engine_args`` is a ``SimpleNamespace`` here instead of a real
    ``VllmConfig``.  Everything the harness checks about the document and about
    the cut is real.
    """
    _stock_vllm_config_stubs(monkeypatch)
    source = Path(__file__).resolve().parents[1] / "experiments/configs/lfm25_first_model_clean_20260907.json"
    document = json.loads(source.read_text())
    document["engine_args"]["tensor_parallel_size"] = tensor_parallel
    path = source.parent / f".owner-tp{tensor_parallel}.json"
    path.write_text(json.dumps(document))
    return path, document


def _stock_vllm_config_stubs(monkeypatch):
    """The named CPU substitution: vLLM's config classes, nothing else."""
    import sys
    from types import ModuleType, SimpleNamespace
    module = ModuleType("vllm.config")
    for name in ("CacheConfig", "ParallelConfig", "SchedulerConfig", "KernelConfig",
                 "CompilationConfig"):
        setattr(module, name, lambda **kwargs: SimpleNamespace(**kwargs))
    module.VllmConfig = lambda **kwargs: SimpleNamespace(**kwargs, device_config={})
    compilation = ModuleType("vllm.config.compilation")
    compilation.CompilationMode = SimpleNamespace(NONE="none")
    compilation.CUDAGraphMode = SimpleNamespace(NONE="none")
    monkeypatch.setitem(sys.modules, "vllm.config", module)
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", compilation)
    monkeypatch.setattr(moe, "_plain",
                        lambda value: vars(value) if isinstance(value, SimpleNamespace) else value)
    monkeypatch.setenv("TESSERA_SERVE_MODE", "resident")


@pytest.mark.parametrize("tensor_parallel", [1, 2])
def test_the_committed_glm_documents_declare_their_own_cut(monkeypatch, tensor_parallel):
    """The window's own documents, one per world, at the geometry's own cut.

    These are the files a two-box run passes as ``serving_config_path``.  The
    document does not get to choose the world: it declares the cut the geometry
    already carries, and the other world is refused.
    """
    _stock_vllm_config_stubs(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    path = root / f"experiments/configs/glm53_routed_owner_tp{tensor_parallel}_20260917.json"
    document = json.loads(path.read_text())
    shape = _glm_shape(tensor_parallel, A8)
    assert document["engine_args"]["tensor_parallel_size"] == shape["tensor_parallel"]
    config, identity = moe.resolve_serving_config(path, document["runtime_image"],
                                                  tensor_parallel=shape["tensor_parallel"])
    assert config.parallel_config.tensor_parallel_size == shape["tensor_parallel"]
    assert identity["resolved"]["parallel_config"]["tensor_parallel_size"] == shape["tensor_parallel"]
    other = 1 if tensor_parallel == 2 else 2
    with pytest.raises(ValueError, match="scope"):
        moe.resolve_serving_config(path, document["runtime_image"], tensor_parallel=other)


@pytest.mark.parametrize("tensor_parallel", [1, 2])
def test_the_serving_config_builds_the_owners_own_cut(monkeypatch, tensor_parallel):
    path, document = _stock_config_runtime(monkeypatch, tensor_parallel=tensor_parallel)
    try:
        config, identity = moe.resolve_serving_config(path, document["runtime_image"],
                                                      tensor_parallel=tensor_parallel)
        assert config.parallel_config.tensor_parallel_size == tensor_parallel
        assert identity["resolved"]["parallel_config"]["tensor_parallel_size"] == tensor_parallel
        # The document is not a place to discover the world: a cut the owner did
        # not declare is refused, in both directions.
        with pytest.raises(ValueError, match="scope"):
            moe.resolve_serving_config(path, document["runtime_image"],
                                       tensor_parallel=1 if tensor_parallel == 2 else 2)
        with pytest.raises(ValueError):
            moe.resolve_serving_config(path, document["runtime_image"], tensor_parallel=4)
    finally:
        path.unlink()


def test_the_distributed_declaration_must_agree_with_the_geometry():
    tp2 = _glm_shape(2, A8)
    with pytest.raises(ValueError, match="no distributed block"):
        moe.owner_distributed(tp2, None)
    block = {"world_size": 2, "rank": 1, "init_method": "tcp://10.0.0.1:29501",
             "timeout_seconds": 120}
    assert moe.owner_distributed(tp2, block) == block
    # The world size is the owner's, and a rank outside it is not a choice.
    with pytest.raises(ValueError, match="world_size"):
        moe.owner_distributed(tp2, {**block, "world_size": 1})
    with pytest.raises(ValueError, match="rank"):
        moe.owner_distributed(tp2, {**block, "rank": 2})
    # A world of two cannot meet at a temp file, and a TP1 owner takes no block.
    with pytest.raises(ValueError, match="tcp://"):
        moe.owner_distributed(tp2, {**block, "init_method": "file:///tmp/rendezvous"})
    assert moe.owner_distributed(_glm_shape(1, A8), None) == {
        "world_size": 1, "rank": 0, "init_method": None, "timeout_seconds": None}


def test_binding_reads_the_live_group_and_refuses_a_mismatch():
    """No group, or the wrong one, is a refusal -- never an assumed rank 0."""
    world = {"world_size": 1, "rank": 0, "init_method": None, "timeout_seconds": None}
    if not torch.distributed.is_initialized():
        with pytest.raises(ValueError, match="no distributed world"):
            moe.bind_owner_rank(world)
        with pytest.raises(ValueError, match="no distributed world"):
            moe.bind_owner_rank({"world_size": 2, "rank": 1, "init_method": "tcp://x:1",
                                 "timeout_seconds": 1})
        return
    assert moe.bind_owner_rank(world) == (0, 1)
    with pytest.raises(ValueError, match="live world is rank"):
        moe.bind_owner_rank({**world, "world_size": 2, "rank": 1})
    with pytest.raises(ValueError, match="live world is rank"):
        moe.bind_owner_rank({**world, "rank": 1, "world_size": 2})


def test_the_owner_route_set_comes_from_the_plugins_own_launch_table():
    """The panel's admissible route is the plugin's, per family and per world.

    ``TESSERA_NVFP4`` has a routed launch table row (its production expert
    builder serves a world above one); ``TESSERA_BF16`` has none, because its
    expert wire is reachable only through the explicit selected owner.  Both
    statements are the plugin's, read here rather than restated.
    """
    materialising = ("vllm.fused_moe.modular_kernel", "torch_materialize_stock")
    a4 = moe.owner_launch_pairs(moe.owner_wire(_glm_shape(1, A4)), world=1)
    assert a4 and all(len(pair) == 2 for pair in a4)
    assert materialising in a4
    # The backend suffix a served record carries is not a second route.
    assert moe.census_symbol_base("vllm.fused_moe.modular_kernel:FLASHINFER_CUTLASS") == materialising[0]
    a16 = moe.owner_launch_pairs(moe.owner_wire(_glm_shape(1, A16)), world=1)
    assert materialising not in a16
    assert {decoder for _symbol, decoder in a16} == {
        "research_selected_torch_window_folded_bf16",
        "research_selected_triton_window_folded_bf16"}


def test_a_tp2_fp8_owner_has_no_production_launch_to_declare():
    """At TP2 the FP8 production builder is out of scope, and so is its pair."""
    wire = moe.owner_wire(_glm_shape(2, A8))
    pairs = moe.owner_launch_pairs(wire, world=2)
    assert ("vllm.fused_moe.modular_kernel", "torch_materialize_stock") not in pairs
    assert ("vllm.fused_moe.modular_kernel", "research_selected_triton_window") in pairs
    assert ("vllm.fused_moe.modular_kernel", "research_selected_torch_window") in pairs
    # A world of one keeps the production pair: that lane is what TP1 has run.
    assert ("vllm.fused_moe.modular_kernel", "torch_materialize_stock") in \
        moe.owner_launch_pairs(wire, world=1)
    # A compressed BF16 expert stack has no production builder at any world.
    bf16 = moe.owner_launch_pairs(moe.owner_wire(_glm_shape(1, A16)), world=1)
    assert ("vllm.fused_moe.modular_kernel", "research_selected_triton_window_folded_bf16") in bf16
    assert ("vllm.fused_moe.modular_kernel", "torch_materialize_stock") not in bf16


def test_the_selected_block_is_required_exactly_where_no_production_owner_exists():
    a8_tp1, a8_tp2 = _glm_shape(1, A8), _glm_shape(2, A8)
    a16, a4_tp2 = _glm_shape(1, A16), _glm_shape(2, A4)
    block = {"schema": "tessera.research_selected_moe.v1", "max_experts_per_chunk": 8,
             "decode_backend": "triton", "expected_tensor_parallel_size": 2}
    # A8 at TP1 is the production lane and takes no block.
    assert moe.owner_research_selected(a8_tp1, moe.owner_wire(a8_tp1), None) is None
    # A8 at TP2 has no production owner; the block is required...
    with pytest.raises(ValueError, match="research_selected_moe"):
        moe.owner_research_selected(a8_tp2, moe.owner_wire(a8_tp2), None)
    # ...and must declare this owner's own cut.
    selected = moe.owner_research_selected(a8_tp2, moe.owner_wire(a8_tp2), block)
    assert selected.expected_tensor_parallel_size == 2
    with pytest.raises(ValueError, match="expected_tensor_parallel_size"):
        moe.owner_research_selected(a8_tp2, moe.owner_wire(a8_tp2),
                                    {**block, "expected_tensor_parallel_size": 1})
    # A16's expert route exists only under the explicit block, at any world.
    with pytest.raises(ValueError, match="research_selected_moe"):
        moe.owner_research_selected(a16, moe.owner_wire(a16), None)
    assert moe.owner_research_selected(a16, moe.owner_wire(a16),
                                       {**block, "expected_tensor_parallel_size": 1}) is not None
    # A4 keeps its own production builder, which serves TP2: attaching the block
    # to it would name a target the block does not serve.
    with pytest.raises(ValueError, match="does not serve"):
        moe.owner_research_selected(a4_tp2, moe.owner_wire(a4_tp2), block)


def _record(shape, dtype="torch.bfloat16"):
    import math
    return {"shape": shape, "dtype": dtype,
            "logical_bytes": math.prod(shape) * moe.DTYPE_BYTES[dtype],
            "content_sha256": _sha(str(shape) + dtype)}


def _small_a8_shape(tp, hidden=64, inter=32, experts=2):
    return {"experts": experts, "hidden_size": hidden, "intermediate_size": inter,
            "top_k": 2, "format": A8}


def _a8_unit(rows, cols, name, seed, q256=1024):
    """One real E4M3 wire, encoded here rather than imported from a sibling test."""
    export = pytest.importorskip("tessera.export")
    alphabet = pytest.importorskip("tessera.alphabet")
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(rows, cols, generator=generator) * 0.02).contiguous()
    exported, _unit, _forests = export.encode_linear_planes(
        weight, grid=alphabet.E4M3_GRID, q256=q256, name=name, verify=False)
    return exported.blob


def _expected_rank_render(blob, scheme, group, role_name, rank, world, *, rows_axis):
    """The rank-local render, derived in PLAIN tensor arithmetic.

    ``slice_unit`` promises the shard decodes to exactly ``decode(parent)[r0:r1,
    c0:c1]``.  The expectation here is a plain slice of the whole decode, so the
    arm tests the trellis-aware cut against ordinary slicing rather than against
    itself.
    """
    from tessera.serving.moe_route import _packed_group_shard_plan
    from tessera.unit_artifact import read_unit_artifact
    plan = _packed_group_shard_plan(scheme, group, "unit", rank, world)
    role = plan.role(role_name)
    whole = read_unit_artifact(blob, "cpu").bfloat16()
    return whole[role.lo:role.hi, :] if rows_axis else whole[:, role.lo:role.hi]


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
def test_each_members_cut_decodes_to_this_ranks_own_render(tp, rank):
    """The TP cut the loader makes IS what this rank's PWC member holds.

    A whole container compared against a rank-local render compares two
    different widths; the seam is ``shard_parsed_roles`` on the group's own
    plan, which at TP1 is the parent object and above it a real cut.
    """
    hidden, inter, experts = 64, 32, 2
    shape = _small_a8_shape(tp, hidden, inter, experts)
    wire = moe.owner_wire(shape)
    from tessera.fused import pack_fused
    from tessera.serving.scheme import MOE_SHARD_PROJECTIONS
    blobs, members, tensors = {}, [], {}
    for expert in range(experts):
        for role, rows, cols, seed, group, rows_axis in (
                ("w1", inter, hidden, 100 + expert, "w13", True),
                ("w3", inter, hidden, 200 + expert, "w13", True),
                ("w2", hidden, inter, 300 + expert, "w2", False)):
            unit = f"unit.{expert}.{role}"
            blob = _a8_unit(rows, cols, role, seed)
            blobs[unit] = (blob, group, role, rows_axis)
            members.append({"unit": unit, "expert": expert, "role": role, "format": A8,
                            "blob": blob, "record": {}})
    strides = {"w13": 0, "w2": 0}
    for unit, (blob, group, role, _axis) in blobs.items():
        strides[group] = max(strides[group], len(pack_fused(
            [(MOE_SHARD_PROJECTIONS[role], moe._declared_member_rows(shape, role), blob)])))
    scheme = moe.owner_scheme(shape, wire, strides=strides, unit="unit")
    for unit, (blob, group, role, rows_axis) in blobs.items():
        expected = _expected_rank_render(blob, scheme, group, MOE_SHARD_PROJECTIONS[role],
                                         rank, tp, rows_axis=rows_axis)
        tensors["source_weight/" + unit] = expected.clone()
        tensors["rendered_weight/" + unit] = expected.clone()
    # The real seam accepts every member at this rank...
    moe.verify_rank_local_member_renders(members, tensors, shape, scheme, unit="unit",
                                         rank=rank, world=tp)
    # ...and refuses a render whose VALUES differ.
    first = members[0]["unit"]
    bad = dict(tensors)
    bad["rendered_weight/" + first] = tensors["rendered_weight/" + first] + 1.0
    with pytest.raises(ValueError, match="decode differs"):
        moe.verify_rank_local_member_renders(members, bad, shape, scheme, unit="unit",
                                             rank=rank, world=tp)
    # The decode runs on the TARGET the render lives on, so the source entry is
    # not what decides where the wire is parsed: a shared-source request has no
    # source tensor in this mapping at all, and the check must still work.
    without_source = {key: value for key, value in tensors.items()
                      if not key.startswith("source_weight/")}
    assert not any(key.startswith("source_weight/") for key in without_source)
    moe.verify_rank_local_member_renders(members, without_source, shape, scheme, unit="unit",
                                         rank=rank, world=tp)
    if tp == 2:
        # ...and at a world above one, refuses the WHOLE container -- the
        # comparison that would have passed while the cut was never made.
        from tessera.unit_artifact import read_unit_artifact
        bad = dict(tensors)
        bad["rendered_weight/" + first] = read_unit_artifact(blobs[first][0], "cpu").bfloat16()
        assert bad["rendered_weight/" + first].shape != \
            tensors["rendered_weight/" + first].shape
        with pytest.raises(ValueError, match="decode differs"):
            moe.verify_rank_local_member_renders(members, bad, shape, scheme, unit="unit",
                                                 rank=rank, world=tp)


def _owner_panel(tp, format_name, route_symbol, decoder):
    """A frozen GLM whole-owner panel at this cut, for the validator only.

    Tensor RECORDS only: this panel never claims those bytes were rendered, and
    the canonical fixture's own wires are what a device run consumes.
    """
    from tessera.serving.scheme import ROUTES
    shape = moe.validate_shape(_glm_shape(tp, format_name))
    wire = moe.owner_wire(shape)
    members = []
    for expert in range(288):
        for role in moe.ROLE_ORDER:
            geometry = moe._member_shape(shape, role)
            record = {"blob_sha256": _sha(f"{GLM_UNIT}.{expert}.{role}"), "blob_bytes": 100}
            members.append({"unit": f"{GLM_UNIT}.{expert}.{role}", "expert": expert, "role": role,
                            "format": format_name, "shape": geometry,
                            "source_weight": _record(geometry),
                            "rendered_weight": _record(geometry),
                            "activation": {"clip_enabled": False, "input_global_scale": None},
                            "wire": {**record, "record": dict(record)}})
    route = {"kind": "moe", "policy": wire["policy"], "decoder": decoder,
             "contract": ROUTES[wire["family"]]["activation_contract"], "symbol": route_symbol}
    routing = _glm_routing()
    phases = {}
    for phase, m in (("prefill", 8), ("decode", 1)):
        phases[phase] = {"m": m, "expected_route": route, "transport": {}}
        for key in moe.TENSOR_KEYS:
            width = shape["top_k"] if key in ("topk_ids", "topk_weights") else shape["hidden_size"]
            dtype = routing[key + "_dtype"] if key in ("input", "topk_ids", "topk_weights") \
                else "torch.bfloat16"
            phases[phase][key] = _record([m, width], dtype)
        for key in ("topk_ids", "topk_weights"):
            phases[phase]["transport"][key] = {"source": dict(phases[phase][key]),
                                               "supplied": dict(phases[phase][key]),
                                               "operation": "identity"}
    workspace = {"schema": moe.WORKSPACE_SCHEMA, "owner": "vllm.WorkspaceManager",
                 "num_ubatches": 1, "num_lanes": 1, "locked": True,
                 "slots": [{"index": 0, "allocation": None}], "resident_bytes": 0}
    execution = moe.owner_execution(shape)
    panel = {"schema": moe.PANEL_SCHEMA, "unit": GLM_UNIT, "format": format_name, "shape": shape,
        "members": members, "profile_role_order": list(moe.ROLE_ORDER), "routing": routing,
        "probe_scope": None, "execution": execution,
        "runtime": {"schema": moe.RUNTIME_SCHEMA, "execution": execution,
            "collective": {"op": moe.RUNTIME_COLLECTIVE_OP, "site": moe.RUNTIME_COLLECTIVE_SITE,
                "required_by_this_owner": tp > 1,
                "runtime_declares_skip_final_all_reduce": False, "world_size": tp}},
        "numerics": {"atol": 2**-6, "rtol": 2**-6}, "phases": phases, "workspace": workspace,
        "workspace_sha256": dense.identity_sha256(workspace),
        "runtime_binding": {"member_formats": {m["unit"]: m["format"] for m in members},
            "member_shapes": {m["unit"]: m["shape"] for m in members},
            "member_operator_identity_sha256": {m["unit"]: _sha(m["unit"] + "joint")
                                                for m in members},
            "operator_route": route["symbol"]}}
    for key in ("routing_capture_sha256", "source_sha256", "calibration_sha256", "cost_sha256",
                "probe_identity_sha256", "native_tensors_sha256", "scheme_sha256", "config_sha256",
                "serving_config_sha256"):
        panel[key] = _sha(key)
    return panel


@pytest.mark.parametrize("tp,format_name,symbol,decoder", [
    (1, A8, "vllm.fused_moe.modular_kernel:FLASHINFER_CUTLASS", "torch_materialize_stock"),
    (2, A4, "vllm.fused_moe.modular_kernel:FLASHINFER_CUTLASS", "torch_materialize_stock"),
    (2, A8, "vllm.fused_moe.modular_kernel:TRITON_REF", "research_selected_triton_window"),
    (1, A16, "vllm.fused_moe.modular_kernel:TRITON_REF",
     "research_selected_triton_window_folded_bf16"),
])
def test_a_glm_owner_panel_validates_at_its_own_family_and_cut(tp, format_name, symbol, decoder):
    panel = _owner_panel(tp, format_name, symbol, decoder)
    assert moe.validate_panel(panel) == panel


@pytest.mark.parametrize("mutation", ["policy", "rung", "execution", "decoder", "symbol"])
def test_the_panel_refuses_a_route_from_another_family_or_cut(mutation):
    """The literals this harness used to carry cannot come back silently."""
    panel = _owner_panel(2, A4, "vllm.fused_moe.modular_kernel:FLASHINFER_CUTLASS",
                         "torch_materialize_stock")
    route = panel["phases"]["prefill"]["expected_route"]
    if mutation == "policy":
        route["policy"] = "TESSERA_FP8:resident"
    elif mutation == "rung":
        panel["shape"]["format"] = A8
    elif mutation == "execution":
        panel["execution"] = dict(moe.EXECUTION)
        panel["runtime"]["execution"] = dict(moe.EXECUTION)
    elif mutation == "decoder":
        route["decoder"] = "native_window_moe_compact"
    else:
        route["symbol"] = "vllm.fused_moe.modular_kernel:"
        panel["runtime_binding"]["operator_route"] = route["symbol"]
    with pytest.raises(ValueError):
        moe.validate_panel(copy.deepcopy(panel))
