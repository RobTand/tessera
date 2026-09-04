"""What a serve EXECUTED, counted by route and shape (issue #102).

``read_route`` records the latest dispatch per module, which is what a census
asserts against. It cannot say how many forwards ran, or at which M, and that
gap is why #83's two-arm served KL measured the prefill path in both arms and
returned a bit-identical null that read like a strong result. The trace closes
it: one counter per ``(policy, shape, symbol, decoder, contract, kind)``.

Pinned here:

* off by default -- no env flag, no trace object, no file, and ``emit_route``
  unchanged;
* the file exists the moment tracing starts, because a path a serve cannot
  write must fail at startup and not silently (``emit_route`` swallows
  exceptions by contract, and arm B of the #83 campaign served with a
  READ-ONLY ``/ext``, so this is the live trap, not a hypothetical one);
* the shape is part of the key. The streamed FP8 route's fallback arm reports
  ``torch._scaled_mm`` in both regimes, so a symbol-only histogram cannot tell
  a 512-row prefill launch from an M=1 decode launch -- it would be one lane
  wearing two names, which is exactly the void experiment #104 found;
* only ``state == "served"`` counts, because ``emit_route`` is two-phase on
  some routes and a pre-launch record must not be counted as a launch.
"""

import json
import os
import stat

import pytest

pytest.importorskip("torch")

from tessera.serving import flags, telemetry  # noqa: E402


class _Layer:
    """A stand-in for a vLLM Linear: the trace reads only ``prefix``."""

    def __init__(self, prefix):
        self.prefix = prefix


def _values(**over):
    values = {"kind": "dense", "policy": "TESSERA_FP8:streamed",
              "symbol": "torch._scaled_mm", "tile_m": 0,
              "shape": "M1:N1024:K2048", "contract": "fp8_per_token_dynamic",
              "state": "served", "reason": None, "decoder": "torch_window"}
    values.update(over)
    return values


def _emit(layer, **over):
    telemetry.emit_route(layer, **{k: v for k, v in _values(**over).items()})


@pytest.fixture
def tracing(tmp_path):
    path = tmp_path / "trace" / "route-trace.json"
    trace = telemetry.start_route_trace(path)
    try:
        yield trace, path
    finally:
        telemetry.stop_route_trace()


def test_tracing_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(telemetry.ROUTE_TRACE_ENV, raising=False)
    telemetry.stop_route_trace()
    layer = _Layer("model.layers.0.mlp.down_proj")
    _emit(layer)
    assert telemetry.route_trace() is None
    assert telemetry.route_trace_snapshot() is None
    assert telemetry.read_route(layer)["symbol"] == "torch._scaled_mm"
    assert list(tmp_path.iterdir()) == []


def test_the_file_exists_the_moment_tracing_starts(tracing):
    _trace, path = tracing
    assert path.exists()
    written = json.loads(path.read_text())
    assert written["schema"] == telemetry.ROUTE_TRACE_SCHEMA
    assert written["entries"] == []
    assert written["pid"] == os.getpid()


def test_an_unwritable_path_refuses_at_startup(tmp_path):
    ro = tmp_path / "readonly"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(OSError):
            telemetry.start_route_trace(ro / "trace.json")
    finally:
        telemetry.stop_route_trace()
        ro.chmod(stat.S_IRWXU)


def test_a_relative_trace_path_is_refused(monkeypatch):
    flags.reset_for_tests(telemetry.ROUTE_TRACE_ENV)
    monkeypatch.setenv(telemetry.ROUTE_TRACE_ENV, "route-trace.json")
    with pytest.raises(ValueError) as exc:
        flags.latched_path(telemetry.ROUTE_TRACE_ENV, meaning="the trace")
    assert "ABSOLUTE" in str(exc.value)
    flags.reset_for_tests(telemetry.ROUTE_TRACE_ENV)


def test_launches_are_counted_per_route_and_shape(tracing):
    trace, path = tracing
    layers = [_Layer(f"model.layers.{i}.mlp.down_proj") for i in range(3)]
    # one prefill forward over three modules...
    for layer in layers:
        _emit(layer, shape="M512:N1024:K2048")
    # ...then four decode forwards on the GEMV lane
    for _ in range(4):
        for layer in layers:
            _emit(layer, shape="M1:N1024:K2048",
                  symbol="tessera_window_gemv::gemv", decoder="window_gemv")
    trace.flush()

    entries = {(e["shape"], e["symbol"]): e
               for e in json.loads(path.read_text())["entries"]}
    assert set(entries) == {("M512:N1024:K2048", "torch._scaled_mm"),
                            ("M1:N1024:K2048", "tessera_window_gemv::gemv")}
    prefill = entries[("M512:N1024:K2048", "torch._scaled_mm")]
    decode = entries[("M1:N1024:K2048", "tessera_window_gemv::gemv")]
    assert (prefill["launches"], prefill["modules"]) == (3, 3)
    assert (decode["launches"], decode["modules"]) == (12, 3)
    assert decode["decoder"] == "window_gemv"


def test_one_symbol_two_shapes_are_two_entries(tracing):
    """The fallback arm's discriminator is the shape, not the symbol."""
    trace, _path = tracing
    layer = _Layer("model.layers.0.self_attn.qkv_proj")
    _emit(layer, shape="M512:N6144:K1024")
    _emit(layer, shape="M1:N6144:K1024")
    shapes = {e["shape"]: e["launches"] for e in trace.snapshot()["entries"]}
    assert shapes == {"M512:N6144:K1024": 1, "M1:N6144:K1024": 1}


def test_a_pre_launch_record_is_not_a_launch(tracing):
    trace, _path = tracing
    layer = _Layer("model.layers.0.mlp.gate_up_proj")
    _emit(layer, state="error", reason="about to launch")
    assert trace.snapshot()["entries"] == []
    _emit(layer, state="served")
    assert [e["launches"] for e in trace.snapshot()["entries"]] == [1]


def test_a_second_process_does_not_clobber_the_histogram(tracing):
    """vLLM loads a general plugin in BOTH the API server and the engine core.

    Only the process holding the model counts anything.  The other one still
    writes the path at startup -- that write is the writability probe -- and
    if it were allowed to write again it would replace a full histogram with
    an empty one.  A census would then read zeros off a lane that ran, which
    is the exact shape of failure the trace exists to rule out.
    """
    trace, path = tracing
    _emit(_Layer("model.layers.0.mlp.down_proj"), shape="M1:N1024:K2048")
    trace.flush()
    assert json.loads(path.read_text())["entries"], "the counting process wrote"

    other = telemetry._RouteTrace(path)          # the API-server process
    other.flush()
    entries = json.loads(path.read_text())["entries"]
    assert [e["launches"] for e in entries] == [1], "the histogram survived"


def test_a_compiled_forward_can_emit_without_killing_the_serve(tracing):
    """The trace must DECLINE under compile, not stop the serve (#113).

    vLLM 0.28 captures the forward with ``aot_compile_fullgraph``, and Dynamo
    cannot enter a ``threading.Lock`` context manager under a full-graph
    capture. The error is raised while COMPILING the traced body, so
    ``emit_route``'s ``except Exception: pass`` never sees it: the engine core
    failed to initialise and a compiled serve with ``TESSERA_ROUTE_TRACE`` set
    never came up at all.

    ``fullgraph=True`` with the eager backend is the same capture and the same
    failure, on CPU, in milliseconds:

        torch._dynamo.exc.Unsupported: Unsupported context manager
          Explanation: Dynamo does not know how to enter a `lock` context
          manager.

    What is pinned is both halves -- the capture succeeds, AND nothing is
    counted, because a trace-time count would describe compilation and this
    class is eager-only by contract.
    """
    import torch

    trace, _path = tracing
    layer = _Layer("model.layers.0.mlp.down_proj")

    def forward(x):
        _emit(layer, shape="M*:N1024:K2048")
        return x + 1

    compiled = torch.compile(forward, fullgraph=True, backend="eager")
    compiled(torch.zeros(2))

    assert trace.snapshot()["entries"] == [], \
        "a compiled forward counted a launch it did not make"
    _emit(layer, shape="M1:N1024:K2048")
    assert [e["launches"] for e in trace.snapshot()["entries"]] == [1], \
        "eager counting after a compiled capture is unchanged"
