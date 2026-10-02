"""Production owners on CPU tensors; these checks do not qualify GPU numerics."""
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from tessera import kernel_window_gemv as kw
from tessera import routed_fused as rf
from tessera.errors import GrammarError

LEGACY, PM = kw.WORD_LAYOUT_LEGACY, kw.WORD_LAYOUT_PIECE_MAJOR


def bomb(*args, **kwargs):
    raise AssertionError("allocation/build/launch reached before layout refusal")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in ("TESSERA_ROUTED_PIECE_MAJOR", "TESSERA_ROUTED_FUSED",
                "TESSERA_DENSE_FUSED", "TESSERA_FUSED_E4M3_MMA"):
        monkeypatch.delenv(key, raising=False)


def unit(family="e4m3", layout=LEGACY, cols=128):
    rep = kw.Repacked(
        words=torch.arange(cols * 64, dtype=torch.int32), tile_words=cols * 64,
        n_tiles=1, rows=128, cols=cols, rows_p=512,
        perm=torch.arange(cols, dtype=torch.int32),
        runs=torch.tensor([[4, 0, cols, 0]], dtype=torch.int32), rates=(4,) * cols)
    if layout != LEGACY:
        rep = rep.with_word_layout(layout)
    return kw.WindowGemvUnit(
        rep=rep, table=torch.zeros(1 << 14, dtype=torch.bfloat16),
        scale=torch.ones(128), window_bits=14,
        plan=kw.default_plan(128, cols, M=1), family=family,
        codes_of_state=torch.zeros(1 << 14, dtype=torch.uint8) if family == "e4m3" else None,
        native=torch.arange(256, dtype=torch.uint8) if family == "e4m3" else None,
        initial_state=torch.arange(cols, dtype=torch.int32) + 17, row_offset=512)


def prepared(family="e4m3", layout=LEGACY):
    from tessera.window_gemm import prepare_window_gemm
    return prepare_window_gemm(unit(family, layout), quantizer=None,
                               arithmetic="folded" if family == "value" else "epilogue")


def bundle(family="e4m3", layout=LEGACY):
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm
    b = prepare_grouped_window_gemm([unit(family, layout)] * 2, quantizer=None,
                                   arithmetic="folded" if family == "value" else "epilogue")
    # CUDA metadata only: tensor-content checks remain real CPU operations.
    return SimpleNamespace(**{**vars(b), "quantizer": "native"}, device=torch.device("cuda"))


@pytest.mark.parametrize("layout", [PM, "unknown"])
def test_gemv_argument_boundary_refuses(layout):
    u = unit()
    u.rep.word_layout = layout
    with pytest.raises(GrammarError, match="word order"):
        kw._op_args(u)


@pytest.mark.parametrize("layout", [PM, "unknown"])
def test_dense_triton_call_refuses_before_allocation(monkeypatch, layout):
    b = dataclasses.replace(prepared("value"), word_layout=layout)
    x = torch.zeros(1, b.cols, dtype=torch.bfloat16)
    monkeypatch.setattr(torch, "empty", bomb)
    with pytest.raises(GrammarError, match="word order"):
        b(x)


@pytest.mark.parametrize("layout", [PM, "unknown"])
def test_fused_dense_preparation_refuses_before_build(monkeypatch, layout):
    b = SimpleNamespace(**{**vars(prepared()), "word_layout": layout}, device=torch.device("cuda"))
    monkeypatch.setattr(rf, "_ext", bomb)
    with pytest.raises(GrammarError, match="word order"):
        rf.prepare_dense_role(b)


@pytest.mark.parametrize("lane", ["triton", "fused"])
def test_dense_module_refuses_before_customop_tag_loss(lane):
    from tessera.serving.native_window import PreparedDenseNativeModule
    b = prepared(layout=PM)
    role = SimpleNamespace(name="gate", rows=b.rows, bundle=b)
    with pytest.raises(GrammarError, match="word order"):
        PreparedDenseNativeModule([role], rows=b.rows, columns=b.cols,
                                  device=b.device, family=b.family, lane=lane,
                                  fused_roles=[SimpleNamespace(rows=b.rows)] if lane == "fused" else None)


def test_legacy_dense_module_reaches_customop(monkeypatch):
    from tessera.serving import native_window as nw
    b = prepared()
    role = SimpleNamespace(name="gate", rows=b.rows, bundle=b)
    module = nw.PreparedDenseNativeModule([role], rows=b.rows, columns=b.cols,
                                         device=b.device, family=b.family)
    calls = []
    def op(*args):
        calls.append(args)
        return torch.zeros(1, b.rows, dtype=torch.bfloat16)
    monkeypatch.setattr(nw, "_window_gemm_dense", op)
    module.apply(torch.zeros(1, b.cols, dtype=torch.bfloat16))
    assert len(calls) == 1 and calls[0][2] is b.words


@pytest.mark.parametrize("family", ["value", "e4m3"])
def test_routed_positive_controls(monkeypatch, family):
    monkeypatch.setattr(rf, "smem_reason", lambda *a: None)
    monkeypatch.setattr(rf, "_ext", bomb)
    b = bundle(family)
    assert rf.fused_routed_window_supported(b, b, b) is None
    if family == "e4m3":
        b.word_layout = PM
        assert rf.fused_routed_window_supported(b, b, b) is None


@pytest.mark.parametrize("family,layout,mma,reason", [
    ("value", PM, "e4m3", "E4M3 family"),
    ("e4m3", PM, "f16", "MMA E4M3"),
    ("e4m3", "unknown", "e4m3", "unknown word layout"),
])
def test_routed_refusal_reaches_layout_gate(monkeypatch, family, layout, mma, reason):
    b = bundle(family)
    b.word_layout = layout
    monkeypatch.setenv("TESSERA_FUSED_E4M3_MMA", mma)
    monkeypatch.setattr(rf, "_ext", bomb)
    monkeypatch.setattr(rf, "smem_reason", bomb)
    assert reason in rf.fused_routed_window_supported(b, b, b)


def test_routed_mixed_tags_refused(monkeypatch):
    a, b = bundle(), bundle(layout=PM)
    monkeypatch.setattr(rf, "_ext", bomb)
    assert "disagree" in rf.fused_routed_window_supported(a, a, b)


@pytest.mark.parametrize("layout", [PM, "unknown"])
def test_e2m1_dense_refuses_before_build(monkeypatch, layout):
    from tessera import routed_fused_e2m1 as fe
    monkeypatch.setattr(fe, "_ext", bomb)
    u = SimpleNamespace(rep=dataclasses.replace(unit(cols=256).rep, word_layout=layout),
                        window_bits=14, arity=2, cols=256, rows=256)
    with pytest.raises(GrammarError, match="word order"):
        fe.prepare_dense_role(u, 1.0)


@pytest.mark.parametrize("layout", [PM, "unknown"])
def test_e2m1_routed_refuses_layout(monkeypatch, layout):
    from tessera import routed_fused_e2m1 as fe
    monkeypatch.setattr(fe, "_ext", bomb)
    b = SimpleNamespace(family="e2m1", word_layout=layout, experts=2)
    assert "word order" in fe.fused_routed_e2m1_supported(b, b, b)


def intake(family):
    from tessera.serving import moe_route as mr
    from tessera.serving.scheme import TESSERA_BF16, TESSERA_FP8
    name = TESSERA_BF16 if family == "value" else TESSERA_FP8
    def group(roles):
        return dict(family=name, grid="BF16" if family == "value" else "E4M3",
                    body="WINDOW", plane="CHANNEL", columns=128,
                    rows=sum(n for _, n in roles), roles=roles,
                    role_q256=[1024] * len(roles), wire_stride=4)
    declared = dict(family=name, experts=2, hidden_size=128, intermediate_size=128,
                    groups=dict(w13=group([("gate", 128), ("up", 128)]),
                                w2=group([("down", 128)])))
    return mr._RankLocalPackedIntake(declared, "layer45" if family == "value" else "layer10",
                                     torch.device("cuda"), 0, 1, compact=True)


@pytest.mark.parametrize("family,optin,fused,mma,expected", [
    ("e4m3", "1", "1", "e4m3", PM), ("e4m3", "0", "1", "e4m3", LEGACY),
    ("e4m3", "1", "0", "e4m3", LEGACY), ("e4m3", "1", "1", "f16", LEGACY),
    ("value", "1", "1", "e4m3", LEGACY),
])
def test_actual_intake_finish_and_history(monkeypatch, family, optin, fused, mma, expected):
    from tessera.serving import moe_route as mr
    monkeypatch.setenv(mr.ENV_PIECE_MAJOR, optin)
    monkeypatch.setenv("TESSERA_ROUTED_FUSED", fused)
    monkeypatch.setenv("TESSERA_FUSED_E4M3_MMA", mma)
    u = unit(family)
    def repack(blob, role, plan, target, **kwargs):
        assert kwargs["family"] == family and kwargs["device"].type == "cuda"
        return role["roles"][0][0], u
    monkeypatch.setattr(mr, "_compact_expert_units", repack)
    monkeypatch.setattr(rf, "_ext", bomb)
    owner = intake(family)
    for e in range(2):
        for group, count in (("w13", 2), ("w2", 1)):
            for i in range(count):
                owner.load(group, i, e, torch.zeros(4, dtype=torch.uint8), device="cuda")
    pointers = {part: slot["words"].data_ptr()
                for axis in owner.axis.values() for part, slot in axis._slots.items()}
    before = owner.resident_bytes()
    bundles = owner.finish(torch.full((2, 2), 4), torch.full((2,), 4))
    assert bundles.word_layout == expected and owner.axis == {} and owner._scratch == {}
    assert bundles.resident_bytes() == before
    want = u.rep.words if expected == LEGACY else u.rep.with_word_layout(PM).words
    for role in ("gate", "up", "down"):
        b = getattr(bundles, role)
        assert b.word_layout == expected and b.words_all.data_ptr() == pointers[role]
        assert b.arithmetic == ("folded" if family == "value" else "epilogue")
        assert torch.equal(b.words_all, want.expand(2, -1))
        assert torch.equal(b.init_all, u.permuted_start_state().expand(2, -1))
        assert torch.equal(b.has_init, torch.ones(2, dtype=torch.int32))


def test_cpu_load_refuses_before_repacking(monkeypatch):
    from tessera.serving import moe_route as mr
    monkeypatch.setattr(mr, "_compact_expert_units", bomb)
    with pytest.raises(ValueError, match="requires a CUDA load device"):
        intake("e4m3").load("w13", 0, 0, torch.zeros(4, dtype=torch.uint8), device="cpu")
