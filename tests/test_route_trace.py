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
    # A file cannot be a directory, even for root in the runtime image.
    # chmod alone is writable under CAP_DAC_OVERRIDE and tests no failure.
    parent = tmp_path / "not-a-directory"
    parent.write_text("occupied")
    try:
        with pytest.raises(OSError):
            telemetry.start_route_trace(parent / "trace.json")
        assert telemetry.route_trace() is None
    finally:
        telemetry.stop_route_trace()


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


# --- tessera#509: per-module identity beside the counts ---------------------

def _by_contract(entries):
    return {e["contract"]: e for e in entries}


def test_each_entry_names_the_modules_it_counted(tracing):
    """The count and the names are one fact, so both must be readable."""
    trace, _path = tracing
    _emit(_Layer("model.layers.0.mlp.down_proj"))
    _emit(_Layer("model.layers.1.mlp.down_proj"))
    entries = trace.snapshot()["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["launches"] == 2
    assert entry["module_names"] == ["model.layers.0.mlp.down_proj",
                                     "model.layers.1.mlp.down_proj"]
    assert entry["modules"] == len(entry["module_names"]) == 2
    assert entry["dispatches_without_prefix"] == 0


def test_a_repeated_prefix_counts_once_but_launches_every_time(tracing):
    trace, _path = tracing
    layer = _Layer("model.layers.0.mlp.down_proj")
    _emit(layer)
    _emit(layer)
    entry = trace.snapshot()["entries"][0]
    assert entry["launches"] == 2
    assert entry["module_names"] == ["model.layers.0.mlp.down_proj"]
    assert entry["modules"] == 1


def test_swapped_contracts_leave_the_histogram_identical_and_move_the_names(tmp_path):
    """THE NEGATIVE REGRESSION for #509.

    Two modules, two contracts, one dispatch each.  Their swap leaves the
    histogram (contract -> launches) byte-identical -- which is exactly how
    this defect hid -- while the per-entry module lists swap with them.  A
    consumer that can only read counts cannot pass this test; that is the
    point of the change.
    """
    a = _Layer("model.layers.0.mlp.experts.0.gate_proj")
    b = _Layer("model.layers.0.mlp.experts.0.up_proj")

    def histogram(entries):
        return sorted((e["contract"], e["launches"]) for e in entries)

    def named(entries):
        return {e["contract"]: e["module_names"] for e in entries}

    # Two independent runs, because the swap is a property of a whole serve
    # and mutating one trace in place would be the test reaching into the
    # object it is measuring.  ``start_route_trace`` is the public install.
    try:
        first = telemetry.start_route_trace(tmp_path / "first.json")
        _emit(a, contract="e2m1_group16_ue4m3_static")
        _emit(b, contract="fp8_per_token_dynamic")
        before_histogram = histogram(first.snapshot()["entries"])
        before_named = named(first.snapshot()["entries"])

        second = telemetry.start_route_trace(tmp_path / "second.json")
        _emit(a, contract="fp8_per_token_dynamic")
        _emit(b, contract="e2m1_group16_ue4m3_static")
        after_histogram = histogram(second.snapshot()["entries"])
        after_named = named(second.snapshot()["entries"])
    finally:
        telemetry.stop_route_trace()

    assert before_histogram == after_histogram, \
        "the fixture must not change the histogram, or it tests nothing"
    assert before_named != after_named, \
        "a swapped contract must move a module name between entries"
    assert (before_named["e2m1_group16_ue4m3_static"]
            == ["model.layers.0.mlp.experts.0.gate_proj"])
    assert (after_named["e2m1_group16_ue4m3_static"]
            == ["model.layers.0.mlp.experts.0.up_proj"])


def test_a_layer_without_a_prefix_is_named_explicitly_not_by_an_id(tracing):
    trace, _path = tracing
    layer = _Layer(None)
    _emit(layer)
    _emit(layer)
    entry = trace.snapshot()["entries"][0]
    assert entry["module_names"] == [telemetry.MODULE_NO_PREFIX]
    assert entry["modules"] == 1
    assert entry["dispatches_without_prefix"] == 2
    assert not any(name.startswith("0x") for name in entry["module_names"]), \
        "an object id is not a module identity a consumer can compare"


def test_module_names_are_sorted_and_independent_of_arrival_order(tracing):
    trace, _path = tracing
    for prefix in ("z.module", "a.module", "m.module"):
        _emit(_Layer(prefix))
    names = trace.snapshot()["entries"][0]["module_names"]
    assert names == sorted(names) == ["a.module", "m.module", "z.module"]


def test_the_header_states_its_own_rank_world_and_platform(tracing, monkeypatch):
    """Uninitialized distributed state is reported as absent, never as rank 0."""
    trace, _path = tracing
    snapshot = trace.snapshot()
    assert snapshot["identity_version"] == telemetry.IDENTITY_VERSION
    assert snapshot["rank"] is None and snapshot["world_size"] is None
    assert snapshot["rank_source"] == "unavailable"
    # The platform token is the SAME value the route records carry, including
    # "" for a process that never latched one (here: no CUDA, no token).  An
    # empty token is honest; inventing a platform would not be.
    assert isinstance(snapshot["platform"], str)

    import torch.distributed as dist
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    snapshot = trace.snapshot()
    assert (snapshot["rank"], snapshot["world_size"]) == (1, 2)
    assert snapshot["rank_source"] == "torch.distributed"


def test_a_legacy_file_stays_histogram_only_and_is_never_upgraded(tracing):
    """A pre-#509 file still reads as what it always was: a histogram.

    It carries no ``module_names`` and no header identity, and nothing here
    invents them for it -- a legacy trace must stay honestly
    histogram-only rather than be upgraded into an identity claim it never
    made.
    """
    trace, _path = tracing
    _emit(_Layer("model.layers.0.mlp.down_proj"))
    payload = trace.snapshot()
    legacy_fields = {"policy", "shape", "symbol", "decoder", "contract",
                     "kind", "launches", "modules"}
    entry = payload["entries"][0]
    assert legacy_fields <= set(entry), \
        "every field a pre-#509 reader reads must survive unchanged"
    assert entry["launches"] == 1 and entry["policy"] == "TESSERA_FP8:streamed"
    # What a legacy FILE contains, reconstructed from the fields it had: the
    # new metadata is absent, and nothing in this change supplies it on read.
    legacy_payload = {"schema": payload["schema"], "pid": 1,
                      "entries": [{k: entry[k] for k in legacy_fields}]}
    assert "identity_version" not in legacy_payload
    assert "rank" not in legacy_payload and "platform" not in legacy_payload
    assert "module_names" not in legacy_payload["entries"][0]
    assert legacy_payload["entries"][0]["modules"] == 1
