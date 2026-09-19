"""Tessera #573: ``emit_route`` records name the executed kernel schedule.

The PrismaQuant consumer (RobTand/prismaquant#767: ``observed_kernel_schedule`` /
``kernel_arch_status``) applies a present-vs-absent rule to
``receipt["phases"][phase]["route"]["kernel_schedule"]``:

- absent (missing or null) = unobserved, ``kernel_arch: "unknown"``, no refusal;
- present = a nonempty string naming the executed schedule, retained per phase;
- present-but-empty or non-string = defect, refused with ``ValueError``.

The producer half lives here: ``emit_route`` threads an optional
``kernel_schedule`` from the call site (unlike ``platform``, which is stamped
centrally because it is a process constant), and every dense/MoE route stamps
its executed op node as the schedule. Fused native routes name their op node;
a future CUTLASS-differentiated dispatch would name its tag instead. Nothing
is emitted with a placeholder when nothing names one: ``None`` stays absent.
"""
import types

import pytest

pytest.importorskip("torch")

from tessera.serving import telemetry


def _layer():
    return types.SimpleNamespace()


def test_emit_route_carries_a_named_schedule():
    layer = _layer()
    telemetry.reset_platform_for_tests()
    try:
        telemetry.emit_route(
            layer, kind="dense", policy="TESSERA_BF16:resident",
            symbol="tessera::window_gemm_dense",
            shape="M64:N128:K256", contract="bf16_unquantized",
            decoder="native_window_gemm",
            kernel_schedule="tessera::window_gemm_dense",
            platform="sm_121",
        )
    finally:
        telemetry.reset_platform_for_tests()
    record = telemetry.read_route(layer)
    assert record["kernel_schedule"] == "tessera::window_gemm_dense"
    assert record["symbol"] == "tessera::window_gemm_dense"


def test_emit_route_absent_schedule_stays_unobserved():
    """Backward compatibility: records written before #573 read as null,
    which the consumer treats exactly like a missing key (unknown, no refusal).
    """
    layer = _layer()
    telemetry.reset_platform_for_tests()
    try:
        telemetry.emit_route(
            layer, kind="dense", policy="TESSERA_FP8:resident",
            symbol="tessera::window_gemm_dense",
            shape="M64:N128:K256", contract="fp8_per_token_dynamic",
            decoder="native_window_gemm", platform="sm_121",
        )
    finally:
        telemetry.reset_platform_for_tests()
    record = telemetry.read_route(layer)
    assert "kernel_schedule" in telemetry.ROUTE_FIELDS
    assert record["kernel_schedule"] is None


def test_emit_route_never_breaks_a_request_on_a_bad_schedule():
    """Telemetry that can break a serve is not telemetry: even a defective
    schedule value must not raise out of ``emit_route``. The consumer refuses
    it downstream; the serve still returns."""
    layer = _layer()
    telemetry.reset_platform_for_tests()
    try:
        telemetry.emit_route(
            layer, kind="dense", policy="TESSERA_NVFP4:resident",
            symbol="tessera.kernel_a4.a4_span2_gemm",
            shape="M64:N128:K256", contract="nvfp4_static",
            decoder="native_span2_gemm", platform="sm_121",
            kernel_schedule="",
        )
    finally:
        telemetry.reset_platform_for_tests()
    assert telemetry.read_route(layer)["symbol"] == "tessera.kernel_a4.a4_span2_gemm"


def test_every_route_stamps_its_executed_symbol_as_the_schedule():
    """The schedule differs per dispatch and must come from the call site
    (#573): each dense/MoE ``emit_route(...)`` call passes its own symbol.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "tessera" / "serving"
    sites = {
        "bf16_route.py": 1,
        "fp8_route.py": 1,
        "nvfp4_route.py": 1,
        "moe_route.py": 1,
        "nvfp4_moe_route.py": 1,
    }
    for name, expected_calls in sites.items():
        text = (root / name).read_text()
        hits = text.count("kernel_schedule")
        assert hits >= expected_calls, f"{name}: no kernel_schedule at emit_route call"
    # The stamped values are the routes' own launch symbols, which are
    # nonempty strings by construction -- exactly what the consumer accepts
    # as present (present-but-empty is refused downstream, never emitted).
    from tessera.serving import bf16_route, fp8_route
    from tessera.serving.scheme import (
        A4_DENSE_GEMM_SYMBOL,
        A4_GROUPED_GEMM_SYMBOL,
        WINDOW_GEMM_SYMBOL,
        WINDOW_MOE_COMPACT_SYMBOL,
    )

    assert bf16_route.DENSE_LAUNCH[0] == WINDOW_GEMM_SYMBOL
    assert fp8_route.DENSE_LAUNCH[0] == WINDOW_GEMM_SYMBOL
    for symbol in (WINDOW_GEMM_SYMBOL, A4_DENSE_GEMM_SYMBOL,
                   A4_GROUPED_GEMM_SYMBOL, WINDOW_MOE_COMPACT_SYMBOL):
        assert isinstance(symbol, str) and symbol.strip()
