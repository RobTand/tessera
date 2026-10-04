"""One served arm of the issue-#545 step-3 A/B: matched runtime comparison.

Runs INSIDE a serve image.  The wrapper (``served_fused_ab_arm.sh``) bind-mounts
exactly one tessera tree at ``/work`` and pip-installs it, so the runtime this
process imports IS the arm's independent variable -- nothing else about the
session moves: same artifact bytes, same tokenizer, same prompt token ids, same
sampling, same engine flags, same instruments.

SCOPE OF THE COMPARISON (parent review 2026-10-04): the arms differ by the
WHOLE runtime (contract v29 -> v45), not by the epilogue fusion alone.  Every
number here is a matched-runtime comparison result; mechanism attribution is
read from the in-engine profiles and route records, never inferred.

Timing locus, stated once and honestly: ``engine_call_ms`` is a CUDA-event
bracket around the ``llm.generate`` call as seen by THIS process.  Where the
engine executes in-process it brackets the enqueued GPU work; where the engine
core is a separate process it brackets submission only.  It is NEVER reported
as GPU kernel duration -- kernel-time claims come only from the in-engine
profiler traces (vLLM's own ``LLM.start_profile``/``stop_profile``, workers
writing into ``VLLM_TORCH_PROFILER_DIR``).  The in-engine route records read
back through ``collective_rpc`` are the proof the GPU work was the route's.

Phases, in order:

1. identity: tessera/vLLM/torch versions, packaged contract version + sha256,
   device, platform token, the launcher's runtime-image declaration
   cross-checked against the environment (#132), and the engine-core
   process facts recorded explicitly;
2. a fixed prompt set derived from the shared wikitext corpus, tokenized by
   the artifact's own tokenizer and passed as explicit ``prompt_token_ids``;
3. idle power floor, warm-up, a decode-regime window and a batch-regime
   window: every forward bracketed by CUDA events and wall clock, a 10 Hz
   board-power sampler over the session, and ONE in-engine profiler window
   per regime (a designated rep, excluded from latency/energy summaries);
4. route records after every phase: what each served module ACTUALLY stamped,
   which is the arm's dispatch proof.

The result JSON, the per-phase summaries (shared convention in
``served_fused_ab_summary.py``), the raw per-rep populations and the log land
under ``--out``; the in-engine traces land under the profiler dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from served_fused_ab_summary import phase_summary  # noqa: E402

# The census tool's rule: the model must be reachable from THIS process for
# the route records to be read in-engine.  Recorded, never assumed: the
# engine-core facts below state what actually ran where.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

WIKITEXT_SLICES = {
    "decode": (45000, 1200),
}
BATCH_OFFSETS = [0, 9000, 18000, 27000, 36000, 45000, 54000, 63000]
BATCH_LEN = 1200
PROMPT_TOKEN_BUDGET = 512


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class PowerSampler:
    """10 Hz board power over the whole session; NVML preferred, nvidia-smi fallback."""

    def __init__(self):
        self.samples: list[tuple[float, float]] = []  # (unix_ts, watts)
        self.source = None
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.source = "pynvml"
        except Exception:  # noqa: BLE001 - the fallback below answers the same question
            self._nvml = None
            try:
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"],
                    check=True, capture_output=True, timeout=10,
                )
                self.source = "nvidia-smi"
            except Exception:  # noqa: BLE001
                self.source = None

    def start(self):
        if self.source is None:
            return
        cadence = 0.1 if self._nvml is not None else 1.0

        def run():
            while not self._stop.is_set():
                w = self._read()
                if w is not None:
                    self.samples.append((time.time(), w))
                self._stop.wait(cadence)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def stats(self, t0: float, t1: float) -> dict:
        xs = [w for ts, w in self.samples if t0 <= ts <= t1]
        if not xs:
            return {"n": 0}
        xs.sort()
        n = len(xs)
        return {
            "n": n,
            "mean_w": sum(xs) / n,
            "min_w": xs[0],
            "max_w": xs[-1],
            "median_w": xs[n // 2],
        }


def build_prompts(tokenizer, corpus_text_path: Path) -> dict:
    """Fixed token-id prompts from fixed corpus byte slices."""
    data = corpus_text_path.read_bytes()
    decode_off, decode_len = WIKITEXT_SLICES["decode"]
    texts = {"decode": [data[decode_off:decode_off + decode_len].decode("utf-8", "replace")]}
    texts["batch"] = [
        data[o:o + BATCH_LEN].decode("utf-8", "replace") for o in BATCH_OFFSETS
    ]
    out = {}
    for phase, tlist in texts.items():
        ids = []
        for t in tlist:
            enc = tokenizer(t, add_special_tokens=True)
            ids.append(enc["input_ids"][:PROMPT_TOKEN_BUDGET])
        out[phase] = ids
    blob = json.dumps(out, sort_keys=True).encode()
    return {"prompts": out, "prompt_ids_sha256": hashlib.sha256(blob).hexdigest(),
            "n_prompt_tokens": {p: sum(len(x) for x in v) for p, v in out.items()}}


def collect_route_records(llm):
    """Every module's route record, read in-engine via collective_rpc."""
    return llm.collective_rpc(_rpc_read_routes)[0]


# Kept at module level so the RPC-method boundary the census documented
# (#545 TP2 census, failure 4) is respected: a callable passed as the RPC
# METHOD ships by value; one passed as an RPC argument would be pickled by
# reference and refuse.
def _rpc_read_routes(worker):
    from tessera.serving.telemetry import read_route
    out = {}
    for name, mod in worker.get_model().named_modules():
        rec = read_route(mod)
        if rec is not None:
            out[name] = rec
    return out


def engine_core_facts(llm) -> dict:
    """Where the engine actually executes, recorded not assumed."""
    engine = getattr(llm, "llm_engine", None)
    facts = {
        "vllm_enable_v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
        "llm_engine_type": type(engine).__name__ if engine is not None else None,
        "llm_engine_module": type(engine).__module__ if engine is not None else None,
    }
    core = getattr(engine, "engine_core", None)
    facts["engine_core_type"] = type(core).__name__ if core is not None else None
    if core is not None:
        # An InprocessClient means generate() ran in THIS process; anything
        # else means a separate engine-core process and the event bracket is
        # submission-only.
        facts["engine_core_in_process"] = "inprocess" in type(core).__name__.lower()
    else:
        facts["engine_core_in_process"] = None
    return facts


def identity_record() -> dict:
    import torch
    import tessera
    from tessera.serving.backend import platform_of_this_process
    rec = {
        "tessera_version": getattr(tessera, "__version__", None),
        "tessera_file": getattr(tessera, "__file__", None),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "platform_token": platform_of_this_process(torch),
        "tessera_commit": os.environ.get("TESSERA_ARM_COMMIT"),
        "tessera_arm": os.environ.get("TESSERA_ARM_NAME"),
        "runtime_image_declared": os.environ.get("TESSERA_CENSUS_RUNTIME_IMAGE"),
        "runtime_image_source": os.environ.get("TESSERA_CENSUS_RUNTIME_IMAGE_SOURCE"),
    }
    try:
        import vllm
        rec["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        from tessera.serving.contract import load_serving_contract
        contract = load_serving_contract()
        rec["contract_version"] = contract.get("contract_version")
        import tessera.serving as serving_pkg
        contract_path = Path(serving_pkg.__file__).parent / "runtime_contract.json"
        rec["contract_sha256"] = sha256_file(contract_path)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        rec["contract_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, help="label only: before|after")
    ap.add_argument("--tag", required=True, help="artifact tag for output names")
    ap.add_argument("--mode", default="resident")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profiler-dir", required=True,
                    help="VLLM_TORCH_PROFILER_DIR the wrapper exported; the "
                         "in-engine traces land here")
    ap.add_argument("--corpus-text", default="/mnt/shared/tessera-kl/wikitext_test.txt")
    ap.add_argument("--reps-decode", type=int, default=24)
    ap.add_argument("--reps-batch", type=int, default=12)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--batch-tokens", type=int, default=32)
    ap.add_argument("--profile-decode-rep", type=int, default=-1,
                    help="1-based rep the in-engine profiler runs under; "
                         "negative = last rep; excluded from summaries")
    ap.add_argument("--profile-batch-rep", type=int, default=-1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--kv-cache-memory-bytes", type=int, default=0)
    ap.add_argument("--glm", action="store_true",
                    help="apply the U1-census GLM serve flags (NoPE backend, CUSTOM "
                         "attention, fp8_ds_mla KV, triton MoE backend)")
    args = ap.parse_args()

    out = Path(args.out)
    prof_dir = Path(args.profiler_dir)
    out.mkdir(parents=True, exist_ok=True)
    prof_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"schema": "tessera.served_fused_ab_arm.v2", "arm": args.arm,
                    "artifact_tag": args.tag, "serve_mode": args.mode,
                    "comparison_scope": ("matched runtime comparison: the whole "
                                         "tessera runtime differs between arms "
                                         "(v29-era pin vs v45 pin); attribution "
                                         "comes from profiles and route records"),
                    "started_utc": utcnow(), "args": vars(args)}

    if args.glm:
        os.environ["TESSERA_RESEARCH_GLM53_NOPE"] = "1"

    sampler = PowerSampler()
    result["power_source"] = sampler.source
    sampler.start()

    fatal = None
    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        result["identity"] = identity_record()

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                            local_files_only=True)
        built = build_prompts(tok, Path(args.corpus_text))
        result["workload"] = {
            "prompt_ids_sha256": built["prompt_ids_sha256"],
            "n_prompt_tokens": built["n_prompt_tokens"],
            "sampling": {"temperature": 0.0, "greedy": True},
            "decode": {"seqs": 1, "max_tokens": args.decode_tokens,
                       "reps": args.reps_decode},
            "batch": {"seqs": len(built["prompts"]["batch"]),
                      "max_tokens": args.batch_tokens, "reps": args.reps_batch},
        }

        kwargs = {"gpu_memory_utilization": args.gpu_mem_util,
                  "max_model_len": args.max_model_len, "seed": 0}
        if args.kv_cache_memory_bytes:
            kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
        if args.glm:
            kwargs.update({"attention_backend": "CUSTOM",
                           "kv_cache_dtype": "fp8_ds_mla",
                           "moe_backend": "triton",
                           "kernel_config": {"enable_flashinfer_autotune": False},
                           "trust_remote_code": True})
        t_load0 = time.time()
        llm = LLM(model=args.model, enforce_eager=True, **kwargs)
        result["load_s"] = time.time() - t_load0
        result["engine_core_facts"] = engine_core_facts(llm)

        sp_decode = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens)
        sp_batch = SamplingParams(temperature=0.0, max_tokens=args.batch_tokens)
        prompts_decode = [{"prompt_token_ids": ids}
                          for ids in built["prompts"]["decode"]]
        prompts_batch = [{"prompt_token_ids": ids}
                         for ids in built["prompts"]["batch"]]

        # idle floor: 5 s with the engine loaded but idle
        t_floor0 = time.time()
        time.sleep(5.0)
        result["idle_floor"] = {"window_s": 5.0,
                                **sampler.stats(t_floor0, time.time())}

        # warm-up (not timed): one batch and one decode forward
        llm.generate(prompts_batch, SamplingParams(temperature=0.0, max_tokens=8),
                     use_tqdm=False)
        llm.generate(prompts_decode, SamplingParams(temperature=0.0, max_tokens=8),
                     use_tqdm=False)
        result["routes_after_warmup"] = collect_route_records(llm)

        def run_phase(name, prompts, sampling, reps, profile_rep_arg, tag):
            """Timed reps; ONE in-engine profiler window; shared summary."""
            prof_rep = (reps + profile_rep_arg if profile_rep_arg < 0
                        else profile_rep_arg)
            if not 1 <= prof_rep <= reps:
                raise ValueError(f"{name}: profiled rep {prof_rep} outside 1..{reps}")
            ph = {"started_utc": utcnow(), "profiled_rep": prof_rep}
            t0 = time.time()
            rows = []
            for r in range(1, reps + 1):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                t_r0 = time.time()
                if r == prof_rep:
                    before = {p.name for p in prof_dir.glob("*") if p.is_file()}
                    llm.start_profile()
                    start.record()
                    o = llm.generate(prompts, sampling, use_tqdm=False)
                    end.record()
                    torch.cuda.synchronize()
                    llm.stop_profile()
                    time.sleep(2.0)
                    after = {p.name for p in prof_dir.glob("*") if p.is_file()}
                    ph["in_engine_profile"] = {
                        "tool": "vllm LLM.start_profile/stop_profile (in-engine)",
                        "new_trace_files": sorted(after - before),
                    }
                else:
                    start.record()
                    o = llm.generate(prompts, sampling, use_tqdm=False)
                    end.record()
                    torch.cuda.synchronize()
                rows.append({
                    "rep": r,
                    "engine_call_ms": start.elapsed_time(end),
                    "wall_s": time.time() - t_r0,
                    "gen_tokens": sum(len(c.token_ids) for x in o for c in x.outputs),
                    "prompt_tokens": sum(len(x.prompt_token_ids) for x in o),
                    "profiled": r == prof_rep,
                })
            ph["t_start"] = t0
            ph["t_end"] = time.time()
            ph["reps"] = rows
            ph["summary"] = phase_summary(rows, prof_rep)
            # Power across the WHOLE phase window, stated as such: the
            # profiled rep's overhead is inside it, so joule figures cite the
            # summary's excluded population for tokens and this window for W.
            ph["power_window"] = {"from": ph["t_start"], "to": ph["t_end"],
                                  "covers_profiled_rep": True,
                                  **sampler.stats(t0, ph["t_end"])}
            return ph

        result["phase_decode"] = run_phase(
            "decode", prompts_decode, sp_decode,
            args.reps_decode, args.profile_decode_rep, args.tag)
        result["phase_batch"] = run_phase(
            "batch", prompts_batch, sp_batch,
            args.reps_batch, args.profile_batch_rep, args.tag)

        result["routes_final"] = collect_route_records(llm)
    except Exception:  # noqa: BLE001 - the arm's refusal is itself a record
        fatal = traceback.format_exc()
        result["fatal"] = fatal

    sampler.stop()
    result["ended_utc"] = utcnow()
    dest = out / f"arm-{args.arm}-{args.tag}.json"
    dest.write_text(json.dumps(result, indent=1))
    print(f"[arm] wrote {dest}" + (" FATAL" if fatal else " ok"))
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
