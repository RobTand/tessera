"""The forensics reader must not report an overload relabeling as a new graph.

WHY THIS FILE EXISTS (issue #29).  Two surviving caches for one key hold
byte-identical ``cache_key_factors.json`` and ``computation_graph.py`` files
whose only difference is the ``vllm_ir`` overload suffix: 42 of 56
``fused_add_rms_norm`` calls read ``maybe_inplace`` in one build and ``default``
in the other.  The pinned pass rewrites every ``maybe_inplace`` node to
``default``, so a dump carrying any ``maybe_inplace`` records pass progress at
write time, not the compiled artifact -- and ``compile_build_forensics.py``
reported the pair as ``computation_graph identical : False`` with no
overload-normalized reading, which is exactly what a future reader will reach
for and misread as a second source of build-to-build variation.

The rule, not the roster: normalization strips the overload suffix off
``torch.ops.vllm_ir.<op>.<overload>(`` -- derived from the same ``IR_OP`` the
census counts with, so the two cannot disagree -- and nothing else.  A graph
with a different op census must still read as different.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _forensics():
    spec = importlib.util.spec_from_file_location(
        "compile_build_forensics",
        ROOT / "experiments" / "compile_build_forensics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _graph(*, inplace: int, default: int, extra_rms: int = 0) -> bytes:
    lines = ["# dumped split graph"]
    for i in range(inplace):
        lines.append(
            f"    n{i} = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(x{i}, w{i});")
    for i in range(inplace, inplace + default):
        lines.append(
            f"    n{i} = torch.ops.vllm_ir.fused_add_rms_norm.default(x{i}, w{i});")
    for i in range(3 + extra_rms):
        lines.append(f"    r{i} = torch.ops.vllm_ir.rms_norm.default(h{i});")
    # A non-vllm_ir overload: semantic to its own dispatcher, never normalized.
    lines.append("    a = torch.ops.aten.add.Tensor(x, y);")
    return ("\n".join(lines) + "\n").encode()


def _pair(tmp_path: Path, a: bytes, b: bytes) -> tuple[Path, Path]:
    pa, pb = tmp_path / "a.py", tmp_path / "b.py"
    pa.write_bytes(a)
    pb.write_bytes(b)
    return pa, pb


def test_an_overload_relabeling_is_not_a_different_graph(tmp_path):
    """The measured case at small scale: same 6 call sites, 4 relabeled."""
    mod = _forensics()
    pa, pb = _pair(tmp_path, _graph(inplace=4, default=2), _graph(inplace=0, default=6))
    verdict = mod.compare_graphs(pa, pb)
    assert verdict["identical_raw"] is False
    assert verdict["identical_modulo_overload"] is True
    assert verdict["a_ops_total"] == verdict["b_ops_total"] == {
        "fused_add_rms_norm": 6, "rms_norm": 3}


def test_a_different_op_census_is_still_a_different_graph(tmp_path):
    """Normalization strips suffixes; it must not equate different graphs."""
    mod = _forensics()
    pa, pb = _pair(tmp_path, _graph(inplace=4, default=2), _graph(inplace=0, default=6, extra_rms=1))
    verdict = mod.compare_graphs(pa, pb)
    assert verdict["identical_raw"] is False
    assert verdict["identical_modulo_overload"] is False


def test_the_normalizer_is_scoped_to_vllm_ir(tmp_path):
    """An aten overload suffix is another dispatcher's semantics, not ours."""
    mod = _forensics()
    pa, pb = _pair(tmp_path, _graph(inplace=1, default=0), _graph(inplace=1, default=0))
    before = pa.read_text()
    assert "torch.ops.aten.add.Tensor(" in mod.normalize_graph_overloads(before)
    assert "torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(" not in mod.normalize_graph_overloads(before)
    assert "torch.ops.vllm_ir.fused_add_rms_norm(" in mod.normalize_graph_overloads(before)
    assert mod.compare_graphs(pa, pb)["identical_modulo_overload"] is True
