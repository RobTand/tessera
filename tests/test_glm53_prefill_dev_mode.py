"""CPU behavioral coverage of GLM prefill's D32 source-identity cutover."""
from types import ModuleType
import sys

from types import SimpleNamespace as NS
import pytest
import torch

from test_glm53_prefill import (
    gp, _config, _hc_expand, _hc_contract, stock_forward, TwoRanks, _ops, _run,
    _kda_module, _ref_conv, _conv_inputs, _TileKernels, _tile_inputs,
)


def _supported_modules(tmp_path, monkeypatch):
    modules = {}
    for index, name in enumerate(dict.fromkeys(gp.SP_MODULES + gp.OVERLAP_MODULES)):
        path = tmp_path / f"supported_{index}.py"
        path.write_text(f"# deliberately different source identity: {name}\n")
        mod = ModuleType(name)
        mod.__file__ = str(path)
        modules[name] = mod
        monkeypatch.setitem(sys.modules, name, mod)

    def accepts(*args, **kwargs):
        return args[0] if args else None

    model = modules[gp.SP_MODULES[0]]
    model.Glm5NextDecoderLayer = type("Glm5NextDecoderLayer", (), {"forward": stock_forward})
    model.hc_expand, model.hc_contract = _hc_expand, _hc_contract
    for name in ("sp_shard", "sp_all_gather", "sp_reduce_scatter"):
        setattr(modules[gp.SP_MODULES[3]], name, accepts)
    modules[gp.SP_MODULES[4]].tensor_model_parallel_all_reduce = accepts
    kernels = modules[gp.SP_MODULES[5]]
    kernels.compute_num_split = lambda block_k, k, grid_size: 1
    kernels.mhc_fused_post_pre_split_config = lambda tokens, hidden, n: None
    kernels._MHC_POST_TILELANG_KERNEL = accepts
    kernels._MHC_PRE_BIG_FUSE_TILELANG_KERNEL = accepts
    modules[gp.SP_MODULES[6]]._hc_prenorm_gemm_outputs = accepts
    modules[gp.SP_MODULES[8]].is_deep_gemm_supported = lambda: True
    modules[gp.OVERLAP_MODULES[0]].CudaCommunicator = type(
        "CudaCommunicator", (), {"all_reduce": accepts})
    modules[gp.OVERLAP_MODULES[1]].should_nccl_symm_mem_allreduce = lambda size, x: False
    modules[gp.OVERLAP_MODULES[2]].PyNcclCommunicator = type(
        "PyNcclCommunicator", (), {"all_reduce": accepts})
    modules[gp.OVERLAP_MODULES[3]].GroupCoordinator = type(
        "GroupCoordinator", (), {"all_reduce": accepts})
    modules[gp.OVERLAP_MODULES[4]].current_stream = lambda: None
    return modules


@pytest.mark.parametrize("value", [None, "1", "", "0 "])
def test_supported_overlap_identity_drift_keeps_selection(tmp_path, monkeypatch, capsys, value):
    if value is None:
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    else:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", value)
    modules = _supported_modules(tmp_path, monkeypatch)
    selected, why = gp._match(tuple(modules[n] for n in gp.OVERLAP_MODULES),
                             gp.OVERLAP_MODULES, gp._OVERLAP_INTERFACES)
    assert selected is gp._OVERLAP_INTERFACES[0], why
    assert why == ""
    assert "[DEV-MODE]" in capsys.readouterr().out


def test_requested_overlap_installs_and_runs_after_identity_drift(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    modules = _supported_modules(tmp_path, monkeypatch)
    monkeypatch.setenv("TESSERA_GLM53_SP_MHC", "off")
    monkeypatch.setenv("TESSERA_GLM53_MHC_TILE", "3")
    monkeypatch.setenv("TESSERA_GLM53_COMM_OVERLAP", "on")
    ranks, reduced = TwoRanks(), []
    ops = _ops(ranks, tile=3, reduced=reduced)
    monkeypatch.setattr(gp, "_vllm_ops", lambda *args: ops)
    monkeypatch.setattr(gp, "_INSTALLED", {})
    states, forwards, configs = [], [], []
    cls = modules[gp.SP_MODULES[0]].Glm5NextDecoderLayer
    for _ in range(2):
        cls.forward = stock_forward
        configs.append(_config())  # keep ids live, as independent rank processes do
        assert gp.install_sp_mhc(configs[-1])
        states.append(gp._INSTALLED["state"])
        assert states[-1].overlap
        forwards.append(cls.forward)
    ref, _ = _run(TwoRanks(), stock_forward, 7)
    # Only the CUDA capture query is replaced: installer, forward and state are real.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    got, _ = _run(ranks, forwards, 7, passes=2)
    assert all(torch.equal(actual, expected) for actual, expected in zip(got, ref))
    assert all(state.pass_overlap and not state.pass_sp for state in states)
    assert reduced


@pytest.mark.parametrize("mode", [None, "0"])
def test_unsupported_overlap_callable_still_declines(tmp_path, monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    else:
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    modules = _supported_modules(tmp_path, monkeypatch)
    modules[gp.OVERLAP_MODULES[2]].PyNcclCommunicator.all_reduce = lambda self, x: x
    selected, why = gp._match(tuple(modules[n] for n in gp.OVERLAP_MODULES),
                             gp.OVERLAP_MODULES, gp._OVERLAP_INTERFACES)
    assert selected is None and why.startswith("unsupported callable interface")
    assert "PyNcclCommunicator.all_reduce" in why


def test_real_overlap_import_failure_keeps_stock_reductions(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    _supported_modules(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, gp.OVERLAP_MODULES[2], None)
    monkeypatch.setattr(gp, "_vllm_ops", lambda *args: _ops(TwoRanks(), tile=3))
    monkeypatch.setattr(gp, "_INSTALLED", {})
    assert gp._install_sp_mhc(_config(), "off", tile=3, overlap=True)
    assert not gp._INSTALLED["state"].overlap


def test_certified_identity_drift_still_declines(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    modules = _supported_modules(tmp_path, monkeypatch)
    selected, why = gp._match(tuple(modules[n] for n in gp.OVERLAP_MODULES),
                             gp.OVERLAP_MODULES, gp._OVERLAP_INTERFACES)
    assert selected is None and "no inspected interface matches" in why


def test_dev_identity_selection_never_hashes_existing_modules(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    modules = _supported_modules(tmp_path, monkeypatch)
    def unexpected_hash(module):
        raise AssertionError("identity hashing is disabled in dev mode")
    monkeypatch.setattr(gp, "_sha256", unexpected_hash)
    assert gp._match_interface(tuple(modules[n] for n in gp.SP_MODULES))[0] is gp._INTERFACES[0]


def test_kda_identity_drift_installs_and_executes_stored_method(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    mod = _kda_module(tmp_path, gp.KDA_STOCK_CONV_BLOCK.rstrip("\n"))
    conv = NS(causal_conv1d_fn=_ref_conv, __file__=mod.__file__)
    monkeypatch.setattr(gp, "_import_all", lambda names: ((mod, conv), ""))
    stock = mod.Glm5NextLinearAttention._forward
    assert gp._install_kda_conv_split(_config())
    p = 8
    qkv, weight, states, has, idx, qsl = _conv_inputs(p=p)
    old_states, new_states = states.clone(), states.clone()
    args = (NS(local_projection_size=p), qkv, weight, None)
    want = stock(*args, old_states, has, idx, qsl, None)
    got = mod.Glm5NextLinearAttention._forward(*args, new_states, has, idx, qsl, None)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert torch.equal(old_states, new_states)


def test_dev_overlap_cpu_data_shape_still_takes_stock_reduction(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    modules = _supported_modules(tmp_path, monkeypatch)
    distributed = ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: None  # CPU shape must decline before touching TP
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    kernels = modules[gp.SP_MODULES[5]]
    cpu = _TileKernels()
    kernels._MHC_POST_TILELANG_KERNEL, kernels._MHC_PRE_BIG_FUSE_TILELANG_KERNEL = cpu.post, cpu.pre
    modules[gp.SP_MODULES[6]]._hc_prenorm_gemm_outputs = cpu.gemm
    reduced = []
    def stock_reduce(x):
        reduced.append(x)
        return x
    modules[gp.SP_MODULES[4]].tensor_model_parallel_all_reduce = stock_reduce
    class Op:
        def forward_cuda(self):
            return None
    op = Op()
    op._forward_method = op.forward_cuda
    inputs = _tile_inputs(7)
    layer = NS(mhc_fused_post_pre_op=op, layer_idx=0, rms_norm_eps=inputs["rms_eps"],
               hc_eps=inputs["hc_pre_eps"], mhc_post_mult_value=inputs["hc_post_mult_value"],
               mhc_sinkhorn_iterations=inputs["sinkhorn_repeat"])
    ops = gp._vllm_ops(tuple(modules[n] for n in gp.SP_MODULES), 2, 3,
                       tuple(modules[n] for n in gp.OVERLAP_MODULES))
    got = ops.reduced_post_pre(layer, inputs["x"], inputs["residual"], inputs["post_layer_mix"],
                              inputs["comb_res_mix"], inputs["fn"], inputs["hc_scale"],
                              inputs["hc_base"], inputs["norm_weight"], inputs["norm_eps"])
    want = gp.tiled_fused_post_pre(_TileKernels(), 3, **inputs)
    assert len(reduced) == 1 and reduced[0] is inputs["x"]
    assert all(torch.equal(a, b) for a, b in zip(got, want))
