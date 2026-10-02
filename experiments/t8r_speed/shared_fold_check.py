"""GPU check: folding the MoE shared add into token_sum is bitwise, and what it saves.

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
2. **SASS.** With ``--base-source`` (the parent commit's
   ``routed_fused_window.cu``), ``experiments/t4_code/sass_dump.py`` compares
   every kernel of all four libraries built from it (the E2M1 one too). Every pre-existing kernel must be
   identical instruction sequences; the new kernel is the only one
   added.
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
import socket
import subprocess
import sys
import time

import torch

HIDDEN, TOP_K = 4096, 8
LIBRARIES = ("value", "e4m3", "e4m3mma")
#: Every library built from routed_fused_window.cu, the E2M1 one included.
SASS_LIBRARIES = LIBRARIES + ("e2m1",)
#: Decode 1-8 (the captured sizes are 1, 2 and 4), a short prefill, the
#: served 2048-token chunk and one token past it.
TOKENS = (1, 2, 3, 4, 5, 6, 7, 8, 512, 2048, 2049)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


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
    else:
        routed, shared = served_scale(tokens, HIDDEN, TOP_K, seed)
    outs = [(form, (stock if form == "stock" else fold)(ext, routed, shared))
            for form in ("stock", "fold", "stock", "fold")]
    torch.cuda.synchronize()
    ref = outs[0][1].view(torch.int16)
    runs = [dict(order=i, form=form, sha256=digest(out),
                 differing=int((out.view(torch.int16) != ref).sum()))
            for i, (form, out) in enumerate(outs)]
    failures = targeted_failures(outs[1][1], TOP_K) if data == "adversarial" else []
    return dict(library=lib, data=data, tokens=tokens, routed_shape=list(routed.shape),
                shared_shape=list(shared.shape),
                nan_fraction=float(torch.isnan(outs[0][1].float()).float().mean()),
                runs=runs, targeted_failures=failures,
                bitwise=all(r["differing"] == 0 for r in runs) and not failures)


def sass_check(base_source, out_dir):
    tool = os.path.join(ROOT, "experiments", "t4_code", "sass_dump.py")
    new_source = os.path.join(ROOT, "src", "tessera", "serving", "csrc", "routed_fused_window.cu")
    rec = dict(base_source=base_source,
               base_sha256=hashlib.sha256(open(base_source, "rb").read()).hexdigest(),
               new_sha256=hashlib.sha256(open(new_source, "rb").read()).hexdigest())
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), "/usr/local/cuda/bin"])
    for tag, src in (("before", base_source), ("after", new_source)):
        dump = subprocess.run([sys.executable, tool, "dump", "--source", src,
                               "--libraries", ",".join(SASS_LIBRARIES),
                               "--out", os.path.join(out_dir, f"sass-{tag}")],
                              capture_output=True, text=True, env=env)
        rec[f"dump_{tag}_rc"] = dump.returncode
        rec[f"dump_{tag}_tail"] = (dump.stdout + dump.stderr)[-2000:]
        if dump.returncode:
            rec["passed"] = False
            return rec
    report = os.path.join(out_dir, "sass-compare.json")
    cmp = subprocess.run([sys.executable, tool, "compare", os.path.join(out_dir, "sass-before"),
                          os.path.join(out_dir, "sass-after"), "--json", report],
                         capture_output=True, text=True)
    rec["compare_rc"] = cmp.returncode
    rec["compare_stdout"] = cmp.stdout[-4000:]
    rows = json.load(open(report)) if os.path.exists(report) else {}
    added = sorted({k for r in rows.values() for k in r.get("only_after", [])})
    rec["only_after"] = added
    rec["passed"] = (cmp.returncode == 0 and set(rows) == set(SASS_LIBRARIES)
                     and all(not r.get("multiset") and not r.get("differs")
                             and not r.get("only_before")
                             and len(r.get("only_after", [])) == 1
                             and "token_sum_shared_kernel" in r["only_after"][0]
                             for r in rows.values()))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--base-source", help="the parent commit's routed_fused_window.cu, for the SASS check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from tessera import routed_fused as rf

    log = dict(host=socket.gethostname(), started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               torch=torch.__version__, device=torch.cuda.get_device_name(),
               capability=list(torch.cuda.get_device_capability()),
               image=os.environ.get("ORACLE_IMAGE"), tessera_head=os.environ.get("TESSERA_HEAD"),
               tessera_state=os.environ.get("TESSERA_STATE"), kernel_sha=os.environ.get("KERNEL_SHA"),
               pb_action=os.environ.get("PB_ACTION_KEY"))
    results = []
    exts = {}
    for lib in LIBRARIES:
        try:
            exts[lib] = rf._ext(lib)
        except Exception as exc:
            results.append(dict(library=lib, bitwise=False, error=f"build {type(exc).__name__}: {exc}"))
            continue
        for data in ("served_scale", "adversarial"):
            for i, tokens in enumerate(TOKENS):
                try:
                    results.append(check_case(lib, exts[lib], data, tokens, 100 * i + len(results)))
                except Exception as exc:
                    results.append(dict(library=lib, data=data, tokens=tokens, bitwise=False,
                                        error=f"{type(exc).__name__}: {exc}"))
                print(json.dumps({k: v for k, v in results[-1].items() if k != "runs"}), flush=True)
    log["cases"] = results
    log["all_bitwise"] = bool(results) and all(r.get("bitwise") for r in results)
    if args.base_source:
        try:
            log["sass"] = sass_check(args.base_source, args.out)
        except Exception as exc:
            log["sass"] = dict(passed=False, error=f"{type(exc).__name__}: {exc}")
        print("SASS", "PASS" if log["sass"].get("passed") else "FAIL",
              log["sass"].get("compare_stdout", log["sass"].get("error", "")), flush=True)
    if "value" in exts:
        try:
            log["screen"] = {lib: screen(exts[lib], args.reps) for lib in exts}
            log["screen_label"] = "[S] torch.profiler and CUDA events, non-measurement row"
        except Exception as exc:
            log["screen_error"] = f"{type(exc).__name__}: {exc}"
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(args.out, "shared_fold_check.json"), "w") as fh:
        json.dump(log, fh, indent=1)
    ok = log["all_bitwise"] and log.get("sass", {}).get("passed", not args.base_source)
    print("ALL_BITWISE" if log["all_bitwise"] else "NOT_BITWISE", flush=True)
    for lib, forms in log.get("screen", {}).items():
        for form, rows in forms.items():
            print(lib, form, [round(r["event_us_per_iter"], 1) for r in rows], flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
