"""Self-test of digest/usercustomize.py inside the serving image (tessera#508).

Run inside the image:  python3 selftest_digest.py OUT_DIR [buffer|full]   (exit 0 = pass)

``buffer`` (default) checks, on a toy root named like the GLM root
(``...ForConditionalGeneration``):
  1. the root is found and its module names are written to <jsonl>.names.json;
  2. every eager root forward appends one line with an input and an output slot
     per module;
  3. a breakable-capture replay re-runs the recorded buffer copies and the
     readout, so a replay after new values reports exactly the digests an eager
     forward of those values reports, for two inputs in turn;
  4. a FULL replay (no root forward runs) is read out by the wrapped model-runner
     ``execute_model`` and reports the digests an eager forward of the same
     input reports; the FULL capture itself writes no line;
  5. the MoE switch forces ``use_fused_finalize=False`` on vLLM's FlashInfer
     CUTLASS MoE binding;
  6. the profiler hook profiles the requested number of steps after the skip,
     writes its trace and tables, and removes the trigger;
  7. nothing is reported missing when every module ran eagerly first.
The runner, graph manager and MoE kernel are stand-ins with the real names
(the wraps look them up by name); the capture and replay are real CUDA graphs.

``full`` checks whole-tensor mode: identical inputs give identical checksums,
a change confined to the last row changes the whole and last-row checksums and
leaves the first-row checksum alone.
"""
import json
import os
import pathlib
import sys
import types

out = pathlib.Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "buffer"
assert mode in ("buffer", "full"), mode
out.mkdir(parents=True, exist_ok=True)
log = out / f"selftest-{mode}.dig.jsonl"
for stale in (log, pathlib.Path(str(log) + ".names.json")):
    if stale.exists():
        stale.unlink()
os.environ["T508_DIGEST"] = str(log)
os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
prof_dir = out / "selftest-prof"
trigger = out / "selftest-prof.trigger"
if mode == "full":
    os.environ["T508_DIGEST_FULL"] = "1"
else:
    os.environ["T508_MOE_DETERMINISTIC"] = "1"
    os.environ["T508_PROF_DIR"] = str(prof_dir)
    os.environ["T508_PROF_TRIGGER"] = str(trigger)
    os.environ["T508_PROF_SKIP"] = "1"
    os.environ["T508_PROF_STEPS"] = "2"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 32)
        self.b = nn.Linear(32, 16)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


class ToyForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()

    def forward(self, x):
        return self.model(x)


def lines():
    return [json.loads(line) for line in log.read_text().splitlines()]


torch.manual_seed(0)
ROWS = 8
x1 = torch.randn(ROWS, 16, device="cuda")
x2 = torch.randn(ROWS, 16, device="cuda")

if mode == "full":
    import usercustomize  # noqa: E402,F401  (installs the global hooks)

    m = ToyForConditionalGeneration().cuda()
    x3 = x1.clone()
    x3[-1] += 1.0
    with torch.inference_mode():
        m(x1)
        m(x2)
        m(x1)
        m(x3)
    got = lines()
    assert len(got) == 4, f"expected 4 readouts, got {len(got)}"
    names = json.loads(pathlib.Path(str(log) + ".names.json").read_text())
    assert names["root"] == "ToyForConditionalGeneration", names
    keys = set(got[0]["digests"])
    want = {"model|in", "model|out", "model.a|in", "model.a|out", "model.b|in", "model.b|out", "root|out"}
    assert want <= keys, sorted(keys)
    assert got[0]["digests"] == got[2]["digests"], "x1 twice must checksum identically"
    assert got[0]["digests"] != got[1]["digests"], "x1 and x2 must checksum differently"
    whole, first, last = (lambda d: d.split("|"))(got[0]["digests"]["root|out"])
    whole3, first3, last3 = got[3]["digests"]["root|out"].split("|")
    assert whole3 != whole and last3 != last and first3 == first, (got[0]["digests"]["root|out"],
                                                                     got[3]["digests"]["root|out"])
    assert all(line["full"] for line in got)
    print(f"digest self-test (full) OK: {len(got)} readouts, {len(keys)} slots; x1 == x1, x1 != x2, "
          "last-row change seen only in whole and last-row checksums")
    sys.exit(0)

# ---------------------------------------------------------------- buffer mode
# Stand-ins with the real names, installed BEFORE the hooks wrap them (the wraps
# happen at the first module forward).
from vllm.model_executor.layers.fused_moe.experts import flashinfer_cutlass_moe as fcm  # noqa: E402
from vllm.v1.worker.gpu import cudagraph_utils as cu  # noqa: E402
from vllm.v1.worker.gpu import model_runner as mr  # noqa: E402

moe_calls = []


def fake_moe(*args, **kwargs):
    moe_calls.append(kwargs.get("use_fused_finalize", "unset"))


fcm.flashinfer_cutlass_fused_moe = fake_moe


def fake_run_fullgraph(self, desc):
    self.graph.replay()
    return self.out


cu.ModelCudaGraphManager.run_fullgraph = fake_run_fullgraph


def fake_execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kwargs):
    if self.full:
        return cu.ModelCudaGraphManager.run_fullgraph(self.mgr, types.SimpleNamespace(num_tokens=ROWS))
    return self.m(self.static)


mr.GPUModelRunner.execute_model = fake_execute_model

import usercustomize  # noqa: E402,F401  (installs the global hooks)
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture  # noqa: E402

m = ToyForConditionalGeneration().cuda()
sched = types.SimpleNamespace(total_num_scheduled_tokens=ROWS)
with torch.inference_mode():
    m(x1)                      # line 1: eager x1 (allocates every slot; installs the wraps)
    m(x2)                      # line 2: eager x2
    assert mr.GPUModelRunner.execute_model is not fake_execute_model, "runner wrap not installed"
    assert fcm.flashinfer_cutlass_fused_moe is not fake_moe, "MoE wrap not installed"
    fcm.flashinfer_cutlass_fused_moe(fc1_expert_weights=x1, use_fused_finalize=True)
    assert moe_calls == [False], moe_calls
    # Breakable (PIECEWISE-style) capture on a side stream, as vLLM captures.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        static = x1.clone()
        cap = BreakableCUDAGraphCapture()
        with cap:
            m(static)          # line 3: capture-time readout (stale buffers, not compared)
    torch.cuda.current_stream().wait_stream(side)
    static.copy_(x2)
    cap.replay()               # line 4: replay with x2's values
    static.copy_(x1)
    cap.replay()               # line 5: replay with x1's values
    n_before_full = len(lines())
    # FULL-style capture: a plain CUDA graph; the readout must skip (no line).
    full_static = x1.clone()
    g = torch.cuda.CUDAGraph()
    side2 = torch.cuda.Stream()
    side2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side2):
        with torch.cuda.graph(g, stream=side2):
            full_out = m(full_static)
    torch.cuda.current_stream().wait_stream(side2)
    assert len(lines()) == n_before_full, "the FULL capture wrote a line"
    runner = types.SimpleNamespace(full=True, mgr=types.SimpleNamespace(graph=g, out=full_out),
                                   m=m, static=None)
    full_static.copy_(x2)
    mr.GPUModelRunner.execute_model(runner, sched)   # line 6: FULL replay of x2
    full_static.copy_(x1)
    mr.GPUModelRunner.execute_model(runner, sched)   # line 7: FULL replay of x1
    # Profiler: skip 1 step with work, profile 2, over eager fake steps.
    trigger.write_text("go")
    eager_runner = types.SimpleNamespace(full=False, mgr=None, m=m, static=x1)
    for _ in range(4):
        mr.GPUModelRunner.execute_model(eager_runner, sched)   # lines 8-11: eager x1
got = lines()
assert len(got) == 11, f"expected 11 readouts, got {len(got)}"
assert cap.num_eager_breaks == 1, cap
names = json.loads(pathlib.Path(str(log) + ".names.json").read_text())
assert names["root"] == "ToyForConditionalGeneration", names
assert {"model", "model.a", "model.b"} <= set(names["recorded"]), names
keys = set(got[0]["digests"])
want = {"model|in", "model|out", "model.a|in", "model.a|out", "model.b|in", "model.b|out", "root|out"}
assert want <= keys, sorted(keys)
assert got[0]["digests"] != got[1]["digests"], "x1 and x2 must digest differently"
assert got[3]["digests"] == got[1]["digests"], "breakable replay(x2) differs from eager(x2)"
assert got[4]["digests"] == got[0]["digests"], "breakable replay(x1) differs from eager(x1)"
assert got[5]["mode"] == "FULL" and got[6]["mode"] == "FULL", (got[5]["mode"], got[6]["mode"])
assert got[5]["digests"] == got[1]["digests"], "FULL replay(x2) differs from eager(x2)"
assert got[6]["digests"] == got[0]["digests"], "FULL replay(x1) differs from eager(x1)"
assert all(line["digests"] == got[0]["digests"] for line in got[7:]), "eager x1 steps differ"
assert not any(line["missing"] for line in got), [line["missing"] for line in got]
assert not trigger.exists(), "profiler did not remove its trigger"
steps = json.loads((prof_dir / "steps.json").read_text())
assert len(steps["steps"]) == 2 and steps["skip"] == 1, steps
assert (prof_dir / "trace.json").stat().st_size > 0 and (prof_dir / "by_cuda.txt").exists()
print(f"digest self-test (buffer) OK: {len(got)} readouts, {len(keys)} slots; breakable and FULL "
      f"replays equal eager for x1 and x2; MoE finalize forced off; profiler wrote "
      f"{len(steps['steps'])} steps; {cap}")
