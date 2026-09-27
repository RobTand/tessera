"""TP1 transfer-qualification driver (GPU, #654) — orchestrates the legs.

One family at a time, per the decided fence (inputs/m1/HANDOVER.md §3):

1. ``fixtures`` — encode one fixture directory per sampled rate (deterministic
   seed per family+rate) so every leg consumes byte-identical inputs.
2. ``fresh`` — r=5 separate processes per sampled rate (subprocesses of this
   driver, never back-to-back in one), each preparing the operator and taking
   ``time_apply`` samples per phase; rep 0 additionally records the resource
   window via the collector, exactly as the fresh-process mode does.
3. ``resource`` — one persistent collector process across the family's sampled
   rates (``run_pass_resource``).
4. ``timing`` — one collector-free process (``run_pass_timing``).
5. ``qualify`` — assemble the r=5 noise band (medians per rate and phase,
   ``eps`` = measured minimal positive back-to-back CUDA-event elapsed time on
   this device), build the three-way cases, and run ``qualify_transfer``.

Every heavy step imports lazily; nothing here runs on CPU test doubles. The
driver refuses to continue if any leg fails rather than dropping a rate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FAMILIES = {
    "TESSERA_BF16_K1": ("BF16_GRID", 1, 256, 4096),
    "TESSERA_E4M3_K1": ("E4M3_GRID", 1, 256, 2048),
    "TESSERA_E2M1_K2": ("E2M1_GRID", 2, 128, 896),
}
SAMPLE_COUNT = 9
FRESH_REPEATS = 5


def _sampled_rates(family):
    from experiments.native_resource_transfer import stratified_rates
    _, _, low, high = FAMILIES[family]
    return stratified_rates(list(range(low, high + 1)), SAMPLE_COUNT)


def _encode_fixture(directory, family, rate, runtime_image):
    """Deterministic fixture per (family, rate): identical bytes in every leg."""
    from tessera.alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID
    from tessera.cached_unit import encoding_input_identity, make_unit_record
    from tessera.export import encode_linear
    import torch
    from tessera.unit_artifact import read_unit_artifact
    grid_name, arity, _, _ = FAMILIES[family]
    grid = {"BF16_GRID": BF16_GRID, "E4M3_GRID": E4M3_GRID, "E2M1_GRID": E2M1_GRID}[grid_name]
    unit, fmt = "fixture.dense", f"{family}_R{rate}"
    torch.manual_seed(hash((family, rate)) & 0xFFFFFFFF)
    weight = (torch.randn(32, 32, device="cuda") * 0.02).bfloat16()
    # E2M1 serves through the NVFP4 route, whose bench prepare refuses a
    # missing activation scale. Calibrate one from this fixture's own inputs
    # (standard NVFP4 headroom: the e4m3 block-scale maximum over the input
    # amax); every other route must carry None or the bench refuses instead.
    input_global_scale = None
    if grid_name == "E2M1_GRID":
        prefill_amax = float(torch.eye(32, device="cuda", dtype=torch.bfloat16)
                             .abs().max())
        input_global_scale = 448.0 / prefill_amax
    encoded = encode_linear(weight.float(), grid=grid, q256=rate, name=fmt, verify=True)
    identity = encoding_input_identity(weight, unit, grid, rate)
    record = make_unit_record(encoded.blob, identity, filename="fixture.tessera")
    rendered = read_unit_artifact(encoded.blob, device="cuda").bfloat16()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fixture.tessera").write_bytes(encoded.blob)
    (directory / "wire-record.json").write_text(json.dumps(record, indent=2) + "\n")
    tensors = {"source_weight": weight, "rendered_weight": rendered}
    for phase, rows in (("prefill", 32), ("decode", 1)):
        x = torch.eye(32, device="cuda", dtype=torch.bfloat16)[:rows].contiguous()
        tensors[f"{phase}.input"] = x
    from safetensors.torch import save_file
    save_file({name: value.clone().contiguous() for name, value in tensors.items()},
              str(directory / "tensors.safetensors"))
    request = {"schema": "tessera.native_dense_request.v1", "unit": unit, "format": fmt,
               "runtime_image": runtime_image, "input_global_scale": input_global_scale}
    (directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")


def _measure_eps_ms(iterations=1000):
    """Minimal positive back-to-back CUDA-event elapsed time on this device."""
    import torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    positives = []
    for _ in range(iterations):
        start.record()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        if elapsed > 0:
            positives.append(elapsed)
    if not positives:
        raise ValueError("no positive back-to-back CUDA event sample; cannot establish eps")
    device = torch.cuda.get_device_name(torch.cuda.current_device())
    return positives, (f"back-to-back torch.cuda.Event elapsed_time over "
                       f"{iterations} pairs on {device} (time_apply's own event timer); "
                       "eps is the minimal positive sample")


def _start_collector(library):
    """Start CUPTI before any Torch import; the collector refuses otherwise."""
    from experiments.native_operator_resources import NativeMemoryCollector
    return NativeMemoryCollector(Path(library))


def _fresh_window_leg(family, rate, fixtures_root, out_path, library):
    """Fresh allocation ground truth: one rate, mirroring pass R exactly.

    The window brackets the same per-rate sequence pass R brackets -- load,
    prepare, one prime, evict, settle -- after one unmarked warmup of this
    rate, so lazy one-time allocations land before the first mark exactly as
    pass R's warmup does. The warmup ends in evict and settle, so the window's
    own load re-requests its allocations instead of reusing cached blocks.
    The collector starts before any Torch import. This leg carries no timing:
    the noise band never comes from a perturbed collected run.
    """
    collector = _start_collector(library)
    import torch

    from experiments.native_transfer_engine import NativeTransferEngine
    rate_name = f"{family}_R{rate}"
    engine = NativeTransferEngine({rate_name: Path(fixtures_root) / f"R{rate}"},
                                  warmup_iterations=8, iterations=8)
    # same bracket order and family context as run_pass_resource: one warmup
    # of this rate (it is rates[0] here), then the marked window with the
    # identity taken after the prime, exactly as the runner does
    with engine.family(family):
        engine.warmup(rate_name)
        begin = collector.mark(f"rate:{rate_name}:begin")
        payload = engine.load(rate_name)
        prepared = engine.prepare(rate_name, payload)
        engine.prime(prepared, payload)
        binding = engine.identity(prepared, payload)
        engine.evict(prepared, payload)
        del prepared, payload
        engine.settle()
        end = collector.mark(f"rate:{rate_name}:end")
    trace = collector.finish(out_path.with_suffix(".memory.json"))
    from experiments.native_resource_transfer import ContinuousTrace
    windows = ContinuousTrace(trace, device_id=torch.cuda.current_device(),
                              context_id=collector.current_context_id())
    window = windows.window(f"rate:{rate_name}")
    if (window["begin_ns"], window["end_ns"]) != (begin, end):
        raise ValueError("fresh window markers do not match this process's collector")
    device_id = torch.cuda.current_device()
    context_id = collector.current_context_id()
    record = {"rate": rate, "binding": binding, "samples_ms": None,
              "process": engine.process(), "collector_started": True,
              "window": window, "trace": trace,
              "device_id": device_id, "context_id": context_id,
              "interval": f"rate:{rate_name}"}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    out_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def _fresh_timing_leg(family, rate, fixtures_root, out_path):
    """Fresh decision timing: no collector, ever."""
    import torch

    from experiments.native_transfer_engine import NativeTransferEngine
    rate_name = f"{family}_R{rate}"
    engine = NativeTransferEngine({rate_name: Path(fixtures_root) / f"R{rate}"},
                                  warmup_iterations=8, iterations=8)
    # same family context and per-rate order as run_pass_timing: load,
    # prepare, identity, time, evict, settle -- the timed warmup comes from
    # time_apply's own warmup iterations, as it does for pass T
    with engine.family(family):
        payload = engine.load(rate_name)
        prepared = engine.prepare(rate_name, payload)
        binding = engine.identity(prepared, payload)
        timing = engine.time(prepared, payload)
        engine.evict(prepared, payload)
        del prepared, payload
        engine.settle()
    record = {"rate": rate, "binding": binding, "samples_ms": timing,
              "process": engine.process(), "collector_started": False}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    out_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def _fresh_leg(family, rate, fixtures_root, out_path, runtime_image, library, with_collector):
    """Dispatch one fresh subprocess leg: rep 0 collects the window, the rest time."""
    if with_collector:
        _fresh_window_leg(family, rate, fixtures_root, out_path, library)
    else:
        _fresh_timing_leg(family, rate, fixtures_root, out_path)


def _read_json(path, label):
    """Fail closed, by name, when a leg input cannot be read back."""
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"{label} could not be read back: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["fixtures", "fresh", "resource", "timing", "qualify"],
                        required=True)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--fixtures-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--runtime-identity", type=Path)
    parser.add_argument("--rate", type=int)
    parser.add_argument("--rep", type=int)
    args = parser.parse_args()
    if args.step == "fixtures":
        for rate in _sampled_rates(args.family):
            _encode_fixture(args.fixtures_root / f"R{rate}", args.family, rate, args.runtime_image)
        print(json.dumps({"sampled_rates": _sampled_rates(args.family)}))
        return 0
    if args.step == "fresh":
        if args.rate is None or args.rep is None:
            parser.error("--step fresh requires --rate and --rep")
        _fresh_leg(args.family, args.rate, args.fixtures_root, args.out,
                   args.runtime_image, args.library, with_collector=(args.rep == 0))
        print(json.dumps({"rate": args.rate, "rep": args.rep, "status": "observed"}))
        return 0
    # The persistent-pass imports stay inside their branches: the resource
    # branch must start the collector before any of them imports Torch.
    identity = _read_json(args.runtime_identity, "runtime identity")
    rates = _sampled_rates(args.family)
    fixtures = {f"{args.family}_R{rate}": args.fixtures_root / f"R{rate}" for rate in rates}
    if args.step == "resource":
        # The collector starts before Torch is imported: NativeMemoryCollector
        # refuses to start once libtorch_cuda is mapped, and the engine module
        # imports Torch. Pass R needs CUPTI live before any CUDA library load,
        # so the engine import stays below the collector start on purpose.
        from experiments.native_resource_passes import run_pass_resource

        collector = _start_collector(args.library)

        from experiments.native_transfer_engine import NativeTransferEngine

        engine = NativeTransferEngine(fixtures)
        # Lazily initialize CUDA first (after the collector started), then read
        # the context id: current_context_id refuses when no context exists.
        device_id = _current_device()
        context_id = collector.current_context_id()
        records = run_pass_resource([f"{args.family}_R{rate}" for rate in rates], engine,
                                    collector, runtime_identity=identity,
                                    trace_path=args.out.with_suffix(".trace.json"),
                                    device_id=device_id, context_id=context_id)
        args.out.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        return 0
    if args.step == "timing":
        from experiments.native_resource_passes import run_pass_timing
        from experiments.native_transfer_engine import NativeTransferEngine

        engine = NativeTransferEngine(fixtures)
        resource_records = _read_json(args.fixtures_root / "pass-r.json",
                                      "pass-R records")
        records = run_pass_timing([f"{args.family}_R{rate}" for rate in rates], engine,
                                  resource_records, runtime_identity=identity)
        args.out.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        return 0
    # qualify: orchestrate the whole family (fresh subprocesses + R + T)
    from experiments.native_resource_transfer import qualify_transfer
    eps_samples, eps_source = _measure_eps_ms()
    pass_r = _read_json(args.fixtures_root / "pass-r.json", "pass-R records")
    pass_t = _read_json(args.fixtures_root / "pass-t.json", "pass-T records")
    resource_trace = _read_json(args.out.with_suffix(".trace.json")
                                if args.out.name == "qualification.json"
                                else args.fixtures_root / "pass-r.trace.json",
                                "the pass-R continuous trace")
    phases = ("prefill", "decode")
    raw_phases = {phase: {"fresh": {}, "persistent": {"process": None, "rates": []}}
                  for phase in phases}
    cases = []
    for rate in rates:
        rate_name = f"{args.family}_R{rate}"
        # Rep 0 is the fresh allocation window (collector, no timing claim);
        # the noise band comes from the FRESH_REPEATS timing legs alone.
        window_out = args.fixtures_root / f"fresh-R{rate}-w.json"
        subprocess.run([sys.executable, "-m", "experiments.native_transfer_qualify",
                        "--step", "fresh", "--family", args.family,
                        "--fixtures-root", str(args.fixtures_root), "--out", str(window_out),
                        "--runtime-image", args.runtime_image, "--library", str(args.library),
                        "--rate", str(rate), "--rep", "0"], check=True)
        fresh = _read_json(window_out, f"fresh window leg for rate {rate}")
        reps = [fresh]
        for rep in range(1, FRESH_REPEATS + 1):
            out = args.fixtures_root / f"fresh-R{rate}-{rep}.json"
            subprocess.run([sys.executable, "-m", "experiments.native_transfer_qualify",
                            "--step", "fresh", "--family", args.family,
                            "--fixtures-root", str(args.fixtures_root), "--out", str(out),
                            "--runtime-image", args.runtime_image, "--library", str(args.library),
                            "--rate", str(rate), "--rep", str(rep)], check=True)
            reps.append(_read_json(out, f"fresh rep {rep} for rate {rate}"))
        for rep in reps[1:]:
            for phase in phases:
                raw_phases[phase]["fresh"].setdefault(str(rate), []).append(
                    {"process": rep["process"], "samples_ms": rep["samples_ms"][phase]})
        resource = pass_r[rate_name]
        timing = pass_t[rate_name]
        for phase in phases:
            persistent = raw_phases[phase]["persistent"]
            if persistent["process"] is None:
                persistent["process"] = timing["process"]
            persistent["rates"].append({"q256": timing["q256"],
                                        "samples_ms": timing["samples_ms"][phase],
                                        "time_in_process": timing["time_in_process"]})
        cases.append({"rate": rate, "cut_axis": None,
                      "fresh": {"trace": fresh["trace"], "device_id": fresh["device_id"],
                                "context_id": fresh["context_id"], "interval": fresh["interval"],
                                "binding": fresh["binding"], "process": fresh["process"]},
                      "resource": {"interval": f"rate:{rate_name}",
                                   "device_id": resource["device_id"],
                                   "context_id": resource["context_id"],
                                   "binding": resource["binding"], "process": resource["process"]},
                      "timing": {"samples_ms": timing["samples_ms"],
                                 "collector_started": timing["collector_started"],
                                 "binding": timing["binding"], "process": timing["process"]}})
    domain = list(range(FAMILIES[args.family][2], FAMILIES[args.family][3] + 1))
    qualification = qualify_transfer(identity, rates=domain, cases=cases,
                                     resource_trace=resource_trace,
                                     noise_band={"source": "fresh_process_repeat_r5_pooled_log",
                                                 "gate_kind": "not_detected",
                                                 "raw": {"eps": {"samples_ms": eps_samples,
                                                                 "eps_source": eps_source},
                                                         "phases": raw_phases}})
    args.out.write_text(json.dumps(qualification, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": qualification["status"], "reasons": qualification["reasons"]}))
    return 0 if qualification["status"] == "passed" else 2


def _current_device():
    import torch
    return torch.cuda.current_device()


if __name__ == "__main__":
    raise SystemExit(main())
