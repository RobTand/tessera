"""tessera#508: summarize a vLLM worker torch-profiler trace of decode steps.

Reads every *.json.gz / *.json chrome trace under DIR and reports, for the
profiled window: GPU kernel count, merged GPU-busy time, kernel-time sum,
launch API counts (cudaLaunchKernel / cuLaunchKernel / cudaGraphLaunch...),
and the span per profiled step when step annotations are present.
  prof-summary-508.py DIR [OUT_JSON]
"""
import gzip, json, pathlib, sys, collections

root = pathlib.Path(sys.argv[1])
files = sorted(p for p in root.rglob("*") if p.is_file() and (p.name.endswith(".json.gz") or p.name.endswith(".json")))
report = {}
for path in files:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    runtime = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")]
    if not kernels and not runtime:
        continue
    iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in kernels)
    busy, cur = 0.0, None
    for s, t in iv:
        if cur is None or s > cur[1]:
            if cur:
                busy += cur[1] - cur[0]
            cur = [s, t]
        else:
            cur[1] = max(cur[1], t)
    if cur:
        busy += cur[1] - cur[0]
    span = (iv[-1][1] - iv[0][0]) if iv else 0.0
    api = collections.Counter(e["name"] for e in runtime)
    steps = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"
             and ("execute_model" in e.get("name", "") or e.get("name", "").startswith("gpu_model_runner"))]
    top = collections.Counter()
    for e in kernels:
        top[e["name"][:90]] += e.get("dur", 0)
    report[str(path.relative_to(root))] = dict(
        kernels=len(kernels), gpu_busy_us=round(busy, 1), kernel_time_sum_us=round(sum(e.get("dur", 0) for e in kernels), 1),
        kernel_span_us=round(span, 1), gpu_busy_fraction_of_span=round(busy / span, 4) if span else None,
        launch_api={k: v for k, v in api.most_common() if "aunch" in k or "Graph" in k},
        runtime_api_time_us=round(sum(e.get("dur", 0) for e in runtime), 1),
        step_annotations=[dict(name=e["name"][:80], dur_us=e.get("dur")) for e in steps][:40],
        top_kernels_us=[(k, round(v, 1)) for k, v in top.most_common(12)])
out = json.dumps(report, indent=1)
print(out)
if len(sys.argv) > 2:
    pathlib.Path(sys.argv[2]).write_text(out)
