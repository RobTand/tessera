"""GPU qualification of the opt-in MoE shared-add fold; timing is a screen [S].

The stock path (image 5be13705): the fused routed window's last kernel,
``token_sum``, stores ``fused = bf16(sum_j routed_j)``; then
``MoERunner.forward`` stores ``shared_output + fused_output`` (ATen's bf16
add). With ``TESSERA_GLM53_FOLD_SHARED_ADD=1`` one kernel,
``token_sum_shared``, stores both (``tessera.serving.glm53_shared_fold``).

1. **Bitwise.** On each library the routed lane builds (value, e4m3,
   e4m3mma), at the GLM-5.3 TP2 shape (H 4096, top-k 8), for T 1 to 8
   (decode), 512, 2048 (the served chunk) and 2049:
   - served-scale normal data;
   - ``tests/shared_fold_data.adversarial``, whose targeted columns must also
     store their known answers.

   Order per case: stock, fold, stock, fold. Every output is compared as
   int16 to the first stock output.
2. **SASS.** With required ``--base-source`` (the original pre-fold
   ``routed_fused_window.cu``), ``experiments/t4_code/sass_dump.py`` compares
   every kernel of all four libraries built from it (the E2M1 one too).
   Every pre-existing kernel must be
   identical in ordered instructions; exactly ``token_sum_shared_kernel``
   is added in each library. Multiset-only equality does not qualify.
3. **Screen.** ``torch.profiler`` at T 2048: stock (token_sum + add)
   against the fold, alternating, plus CUDA-event time per iteration. It runs
   on a non-measurement row, so it is a screen [S].

Writes ``<out>/shared_fold_check.json``; exits 1 unless every case is bitwise,
the targeted answers hold, and the SASS check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.t4_code import sass_dump
from tessera import routed_fused as rf

HIDDEN, TOP_K = 4096, 8
LIBRARIES = tuple(rf.LIBRARIES)
SASS_LIBRARIES = tuple(sass_dump.LIBRARIES)
TOKENS = (1, 2, 3, 4, 5, 6, 7, 8, 512, 2048, 2049)
DATA_FORMS = ("served_scale", "adversarial")
RUN_FORMS = ("stock", "fold", "stock", "fold")
SOURCE = os.path.join(ROOT, "src", "tessera", "serving", "csrc", "routed_fused_window.cu")


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def stock(ext, routed, shared):
    fused = torch.empty(shared.shape, dtype=torch.bfloat16, device=shared.device)
    ext.token_sum(routed, fused, TOP_K)
    return shared + fused


def fold(ext, routed, shared):
    out = torch.empty(shared.shape, dtype=torch.bfloat16, device=shared.device)
    ext.token_sum_shared(routed, shared, out, TOP_K)
    return out


def check_case(lib, ext, data, tokens, seed):
    from shared_fold_data import adversarial, served_scale, targeted_failures

    if data == "adversarial":
        routed, shared = adversarial(tokens, HIDDEN, TOP_K, seed, ext.token_sum)
    elif data == "served_scale":
        routed, shared = served_scale(tokens, HIDDEN, TOP_K, seed)
    else:
        raise ValueError(f"data form outside qualification scope: {data!r}")
    outs = [(form, (stock if form == "stock" else fold)(ext, routed, shared))
            for form in RUN_FORMS]
    torch.cuda.synchronize()
    ref = outs[0][1].view(torch.int16)
    runs = [dict(order=i, form=form, sha256=digest(out),
                 differing=int((out.view(torch.int16) != ref).sum()),
                 targeted_failures=targeted_failures(out, TOP_K) if data == "adversarial" else [])
            for i, (form, out) in enumerate(outs)]
    failures = [f"run {r['order']}: {failure}" for r in runs for failure in r["targeted_failures"]]
    return dict(library=lib, data=data, tokens=tokens, routed_shape=list(routed.shape),
                shared_shape=list(shared.shape),
                nan_fraction=float(torch.isnan(outs[0][1].float()).float().mean()),
                runs=runs, targeted_failures=failures,
                bitwise=all(r["differing"] == 0 for r in runs) and not failures)


def qualify_numeric(cases):
    expected = {(lib, tokens, data) for lib in LIBRARIES for tokens in TOKENS for data in DATA_FORMS}
    seen, reasons = set(), []
    if not isinstance(cases, list):
        return dict(passed=False, failure_reasons=["numeric cases must be a list"])
    for i, case in enumerate(cases):
        label = f"case {i}"
        if not isinstance(case, dict):
            reasons.append(f"{label}: malformed case")
            continue
        key = (case.get("library"), case.get("tokens"), case.get("data"))
        if (not isinstance(key[0], str) or type(key[1]) is not int
                or not isinstance(key[2], str) or key not in expected):
            reasons.append(f"{label}: unexpected case {key!r}")
        else:
            if key in seen:
                reasons.append(f"{label}: duplicate case {key!r}")
            seen.add(key)
        if "error" in case:
            reasons.append(f"{label}: error: {case['error']}")
        if case.get("bitwise") is not True or case.get("targeted_failures") != []:
            reasons.append(f"{label}: bitwise or targeted answers failed or omitted")
        runs = case.get("runs")
        if not isinstance(runs, list) or len(runs) != len(RUN_FORMS):
            reasons.append(f"{label}: requires stock/fold/stock/fold executions")
            continue
        reference = runs[0].get("sha256") if isinstance(runs[0], dict) else None
        if not isinstance(reference, str) or re.fullmatch(r"[0-9a-f]{64}", reference) is None:
            reasons.append(f"{label}: missing or malformed first stock digest")
        for order, (run, form) in enumerate(zip(runs, RUN_FORMS)):
            run_label = f"{label} run {order}"
            if not isinstance(run, dict):
                reasons.append(f"{run_label}: malformed execution")
                continue
            if type(run.get("order")) is not int or run["order"] != order or run.get("form") != form:
                reasons.append(f"{run_label}: requires {form} at order {order}")
            if (type(run.get("differing")) is not int or run["differing"] != 0
                    or run.get("sha256") != reference):
                reasons.append(f"{run_label}: does not match first stock output bitwise")
            if "error" in run:
                reasons.append(f"{run_label}: error: {run['error']}")
            if case.get("data") == "adversarial" and run.get("targeted_failures") != []:
                reasons.append(f"{run_label}: targeted answers failed or omitted")
    missing = expected - seen
    if missing:
        reasons.append(f"missing numeric cases: {sorted(missing)!r}")
    return dict(passed=not reasons, failure_reasons=reasons)


def qualify_sass(rec):
    reasons = []
    if not isinstance(rec, dict):
        return dict(passed=False, failure_reasons=["missing or malformed SASS evidence"])
    if "error" in rec:
        reasons.append(f"SASS error: {rec['error']}")
    for command in ("dump_before_rc", "dump_after_rc", "compare_rc"):
        if type(rec.get(command)) is not int or rec[command] != 0:
            reasons.append(f"{command}: command failed or evidence omitted")
    report = rec.get("report")
    if not isinstance(report, dict):
        reasons.append("missing or malformed SASS comparison report")
        return dict(passed=False, failure_reasons=reasons)
    if set(report) != set(SASS_LIBRARIES):
        reasons.append(f"SASS libraries must be exactly {list(SASS_LIBRARIES)!r}")
    fields = {"identical", "multiset", "differs", "only_before", "only_after"}
    names = rec.get("demangled", {})
    for lib, rows in report.items():
        if (not isinstance(rows, dict) or set(rows) != fields
                or any(not isinstance(rows[k], list) for k in fields)):
            reasons.append(f"{lib}: malformed kernel classifications")
            continue
        if any(not isinstance(name, str) or not name for k in fields - {"differs"} for name in rows[k]):
            reasons.append(f"{lib}: malformed kernel names")
            continue
        if not rows["identical"]:
            reasons.append(f"{lib}: empty identical original-kernel coverage")
        for kind in ("multiset", "differs", "only_before"):
            if rows[kind]:
                reasons.append(f"{lib}: {kind} kernels do not qualify")
        kernels = [name for k in fields - {"differs"} for name in rows[k]]
        if len(set(kernels)) != len(kernels):
            reasons.append(f"{lib}: duplicate kernel classifications")
        added = rows["only_after"]
        if len(added) != 1:
            reasons.append(f"{lib}: requires exactly one added token_sum_shared_kernel")
            continue
        demangled = names.get(added[0]) if isinstance(names, dict) else None
        if (not isinstance(demangled, str)
                or re.fullmatch(r"(?:.*::)?token_sum_shared_kernel\([^()]*\)", demangled) is None):
            reasons.append(f"{lib}: added function is not exactly token_sum_shared_kernel: {demangled!r}")
    return dict(passed=not reasons, failure_reasons=reasons)


def qualify(log):
    numeric = qualify_numeric(log.get("cases"))
    sass = qualify_sass(log.get("sass"))
    reasons = ([f"numeric: {r}" for r in numeric["failure_reasons"]]
               + [f"SASS: {r}" for r in sass["failure_reasons"]])
    return dict(passed=not reasons, failure_reasons=reasons)


def sass_check(base_source, out_dir):
    tool = os.path.join(ROOT, "experiments", "t4_code", "sass_dump.py")
    rec = dict(base_source=base_source, new_source=SOURCE,
               before_dir=os.path.join(out_dir, "sass-before"),
               after_dir=os.path.join(out_dir, "sass-after"),
               compare_report=os.path.join(out_dir, "sass-compare.json"))
    try:
        with open(base_source, "rb") as source:
            rec["base_sha256"] = hashlib.sha256(source.read()).hexdigest()
        with open(SOURCE, "rb") as source:
            rec["new_sha256"] = hashlib.sha256(source.read()).hexdigest()
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([env.get("PATH", ""), "/usr/local/cuda/bin"])
        for tag, src in (("before", base_source), ("after", SOURCE)):
            destination = rec[f"{tag}_dir"]
            os.makedirs(destination, exist_ok=True)
            for lib in SASS_LIBRARIES:
                artifact = os.path.join(destination, f"{lib}.json")
                if os.path.exists(artifact):
                    os.unlink(artifact)
            dump = subprocess.run([sys.executable, tool, "dump", "--source", src,
                                   "--libraries", ",".join(SASS_LIBRARIES), "--out", destination],
                                  capture_output=True, text=True, env=env)
            rec[f"dump_{tag}_rc"] = dump.returncode
            rec[f"dump_{tag}_tail"] = (dump.stdout + dump.stderr)[-2000:]
            dump.check_returncode()
        report_path = rec["compare_report"]
        if os.path.exists(report_path):
            os.unlink(report_path)
        cmp = subprocess.run([sys.executable, tool, "compare", rec["before_dir"], rec["after_dir"],
                              "--json", report_path], capture_output=True, text=True)
        rec["compare_rc"] = cmp.returncode
        rec["compare_stdout"] = cmp.stdout[-4000:]
        with open(report_path) as report:
            rec["report"] = json.load(report)
        rec["only_after"] = sorted({name for rows in rec["report"].values() for name in rows["only_after"]})
        cmp.check_returncode()
        demangled = {}
        for lib in SASS_LIBRARIES:
            before = sass_dump._kernels(os.path.join(rec["before_dir"], f"{lib}.json"))
            after_path = os.path.join(rec["after_dir"], f"{lib}.json")
            after = sass_dump._kernels(after_path)
            if not before or any(not isinstance(seq, list) or not seq
                                 or any(not isinstance(ins, str) or not ins for ins in seq)
                                 for seq in (*before.values(), *after.values())):
                raise ValueError(f"{lib}: missing or malformed original instruction coverage")
            rows = rec["report"][lib]
            if set(rows["identical"]) != set(before) or set(rows["only_after"]) != set(after) - set(before):
                raise ValueError(f"{lib}: report does not cover the dumped kernels identically")
            with open(after_path) as artifact:
                raw_kernels = json.load(artifact)["kernels"]
            for raw in raw_kernels:
                name = sass_dump._kernel_name(raw)
                if name in rows["only_after"]:
                    result = subprocess.run(["c++filt", raw], capture_output=True, text=True)
                    result.check_returncode()
                    demangled[name] = result.stdout.strip()
        rec["demangled"] = demangled
    except (Exception, SystemExit) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec.update(qualify_sass(rec))
    return rec


def screen(ext, reps):
    from torch.profiler import ProfilerActivity, profile

    from shared_fold_data import served_scale

    routed, shared = served_scale(2048, HIDDEN, TOP_K, 77)
    for _ in range(3):
        stock(ext, routed, shared)
        fold(ext, routed, shared)
    torch.cuda.synchronize()
    result = {}
    for form in ("stock", "fold", "stock", "fold"):
        fn = stock if form == "stock" else fold
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                fn(ext, routed, shared)
            end.record()
            torch.cuda.synchronize()
        kernels = {}
        for ev in prof.key_averages():
            t = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0)
            if t and ev.count:
                kernels[ev.key[:120]] = dict(count=ev.count, us_per_call=t / ev.count)
        result.setdefault(form, []).append(dict(event_us_per_iter=1000 * start.elapsed_time(end) / reps,
                                                kernels=kernels))
    return result


def write_receipt(log, out_dir):
    log["numeric"] = qualify_numeric(log.get("cases"))
    log["all_bitwise"] = log["numeric"]["passed"]
    log["qualification"] = qualify(log)
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(out_dir, "shared_fold_check.json"), "w") as fh:
        json.dump(log, fh, indent=1)
    print("QUALIFICATION_PASS" if log["qualification"]["passed"] else "QUALIFICATION_FAIL", flush=True)
    for reason in log["qualification"]["failure_reasons"]:
        print(reason, flush=True)
    return 0 if log["qualification"]["passed"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--base-source", help="required original pre-fold routed_fused_window.cu for qualification")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    log = dict(host=socket.gethostname(), started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               torch=torch.__version__, device=None, capability=None,
               image=os.environ.get("ORACLE_IMAGE"), tessera_head=os.environ.get("TESSERA_HEAD"),
               tessera_state=os.environ.get("TESSERA_STATE"), kernel_sha=os.environ.get("KERNEL_SHA"),
               pb_action=os.environ.get("PB_ACTION_KEY"), cases=[],
               screen_label="[S] torch.profiler and CUDA events, non-measurement row",
               screen_limits="Timing screens establish no served saving or energy benefit.")
    try:
        if not args.base_source:
            raise ValueError("--base-source is required: name the original pre-fold routed_fused_window.cu")
        with open(args.base_source, "rb") as source:
            log["base_sha256"] = hashlib.sha256(source.read()).hexdigest()
        with open(SOURCE, "rb") as source:
            log["new_sha256"] = hashlib.sha256(source.read()).hexdigest()
    except (OSError, ValueError) as exc:
        log["sass"] = dict(error=f"--base-source refusal: {type(exc).__name__}: {exc}", passed=False)
        return write_receipt(log, args.out)
    try:
        log["device"] = torch.cuda.get_device_name()
        log["capability"] = list(torch.cuda.get_device_capability())
    except Exception as exc:
        log["cases"].append(dict(error=f"CUDA execution: {type(exc).__name__}: {exc}", bitwise=False))
        return write_receipt(log, args.out)
    results, exts = log["cases"], {}
    for lib in LIBRARIES:
        try:
            exts[lib] = rf._ext(lib)
        except Exception as exc:
            results.append(dict(library=lib, bitwise=False, error=f"build {type(exc).__name__}: {exc}"))
            continue
        for data in DATA_FORMS:
            for i, tokens in enumerate(TOKENS):
                try:
                    results.append(check_case(lib, exts[lib], data, tokens, 100 * i + len(results)))
                except Exception as exc:
                    results.append(dict(library=lib, data=data, tokens=tokens, bitwise=False,
                                        error=f"{type(exc).__name__}: {exc}"))
                print(json.dumps({k: v for k, v in results[-1].items() if k != "runs"}), flush=True)
    log["sass"] = sass_check(args.base_source, args.out)
    print("SASS", "PASS" if log["sass"]["passed"] else "FAIL",
          log["sass"].get("compare_stdout", log["sass"].get("error", "")), flush=True)
    if exts:
        try:
            log["screen"] = {lib: screen(ext, args.reps) for lib, ext in exts.items()}
        except Exception as exc:
            log["screen_error"] = f"{type(exc).__name__}: {exc}"
    for lib, forms in log.get("screen", {}).items():
        for form, rows in forms.items():
            print(lib, form, [round(r["event_us_per_iter"], 1) for r in rows], flush=True)
    return write_receipt(log, args.out)


if __name__ == "__main__":
    sys.exit(main())
