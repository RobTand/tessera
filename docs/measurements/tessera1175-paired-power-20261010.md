# tessera#1175 paired power receipt (2026-10-10)

This receipt pairs fast NVML samples with Netdata readback over one
shared UTC window. It resolves the coarse-sensor gap for work-per-joule
use. Truth is `src/tessera/serving/timing_panel.py` telemetry:
`interval_unix`, `fast_power_samples`, and raw `netdata` responses.
No collector file changed. No flag, pin, gate, or kernel code changed.

Raw series, bounds, and clock probes live in
`tessera1175-paired-power-20261010.json` beside this file. That file
holds two runs in full: the primary exclusive run and one shared
supporting run. `tests/test_tessera1175_power_bounds.py` recomputes
every derived number below from that raw block.

## Method

The capture mirrors `tools/tessera_shape_time_worker.py` lines 553-570:
`PowerSampler` starts, `t0 = time.time()`, a torch matmul loop runs,
`t1 = time.time()`, samples filter to `[t0, t1]`, and
`box_power_window._fetch` reads each Netdata host over
`[int(t0), int(t1) + 1]`. One addition covers this issue: an NTP-style
clock probe per host (HTTP `Date` header, median of five, RTT kept).

Workload: 4096x4096 bfloat16 matmuls on sparklina (NVIDIA GB10),
90 s, 10 Hz sampler, `pynvml.nvmlDeviceGetPowerUsage` source.

Actions (priority 0 throughout):

| Step | Action | Box | Result |
|------|--------|-----|--------|
| CPU preflight | `cf71898395...` | dl380g10 | imports, parse, torch CPU shapes pass in 3 s |
| Shared capture | `f6b51f65...` | sparklina | 90.0 s, fast mean 82.53 W |
| Exclusive capture (primary) | `175c22af...` | sparklina | 90.0 s, fast mean 81.27 W |

CPU preflight could not resolve `sparky`/`sparklina`. Those names
resolve only on the Sparks. Worker requests use the same names, so
this matches the serving path by design.

## Shared bounds and clock delta

Primary window: `[1791606911.71, 1791607001.72]`
(2026-10-10T04:35:11Z to 2026-10-10T04:36:41Z). Fast series: 899
samples, all inside the window. Netdata query windows bracket it:
`after=1791606911`, `before=1791607002` on both hosts.

Clock probe medians (server minus client):

| Host | Pre | Post | Max RTT |
|------|-----|------|---------|
| sparky | -0.43 s | -0.81 s | 0.022 s |
| sparklina | -0.44 s | -0.82 s | 0.002 s |

Both servers agree within 10 ms in each set. The 0.38 s set shift
is `Date` header quantization, not a clock step: each set is
internally tight, and the two truncation intervals intersect near
-0.2 s. Adopted envelope for pairing: plus or minus 1.0 s.

## Bias root cause

Netdata serves 1 s groups, but the store collects every 10 s
(`db.update_every = 10` on both boxes, tier 0). The 1 s view is a
resample of 10 s collections. Four effects follow:

1. Edge smear: the 5 W to 69 W spin-up step spreads over four
   Netdata groups (12, 20, 28, 36 W). Fast samples show the true step.
2. Ripple loss: sub-10 s structure (0.5 s NVML update blocks,
   plus or minus 2-3 W ripple) never appears in Netdata.
3. Endpoint shift: the returned view reads `[1791606909, 1791607000]`,
   not the requested `[1791606911, 1791607002]`. Pair on the
   intersection. Never interpolate a straddler (`bounded_groups`).
4. Tail lag: at immediate readback the newest Netdata group ends
   12 s before `t1` (last group 1791606990). Re-query later, or cut
   the window at the last returned group.

Peer divergence: the shared run shows sparky at 4 W, then ramping
to 33 W late in the window. Another tenant started work there.
Peer series never substitute for box-local series. Non-exclusive
placement contaminates. The primary run is exclusive.

## Work-per-joule numbers

Primary run, sparklina, intersected window `[t0, last Netdata group]`:

| Window | Fast energy | Netdata energy | Bias |
|--------|-------------|----------------|------|
| Full intersect | 6366.5 J | 6155.0 J | -3.32 % |
| Steady (`t0+10 s` on) | 5591.4 J | 5603.5 J | +0.22 % |

Shift sensitivity of the steady bias: +0.31 % at -1 s,
-1.27 % at +1 s. Shift convention: a shift of plus s pairs a
Netdata group at stamp t as true time t minus s. Plus s models
a Netdata clock ahead of true time by s. The test recomputes
each shift value from the raw series with this rule.

## Bound for work-per-joule use

Observed worst case on the qualifying capture: steady bias
+0.22 %, +0.31 % at -1 s shift, -1.27 % at +1 s shift. Maximum
absolute value is 1.27 %. The bound is plus or minus 1.5 %:
ceiling of 1.27 plus margin for 1 s `Date` header quantization
and 10 s Netdata collection.

Scope: exclusive capture of 60 s or more, thermally steady load,
tail cut at the last Netdata group, clock shift applied.
Steadiness gate: fast head mean (`t0+10 s` to `t0+25 s`) versus
tail mean (last 15 s) drift within 3 %. Primary drift is -2.39 %
and passes. The shared run drift is -5.30 % and fails: it started
hot at 90 W after prior load and cooled through the run. Its +1 s
shift bias of -2.62 % exceeds the bound and shows the cost of
unsteady use. Transient edges stay outside any joule claim.
This keeps the `timing_panel.py` HOLD: no rank reads raw
Netdata means without this window rule and this bound.

Limit: one qualifying capture supports this bound. It is an
observed maximum plus margin, not a population statistic.
A second exclusive capture should confirm it before rank use.

## Rerun

Resubmit the payload below through PrismaBuild (`--tag gb10`,
priority 0, cu130 venv). It prints one JSON block between
`TS1175-JSON-BEGIN` and `TS1175-JSON-END`. Recompute with
`tests/test_tessera1175_power_bounds.py` after appending the block
to the receipt JSON.

```python
"""Paired fast-power vs Netdata capture for tessera#1175 (no new collector file).

Mirrors tools/tessera_shape_time_worker.py lines 553-570 (PowerSampler start,
t0=time.time(), workload, t1=time.time(), samples filtered to [t0, t1],
box_power_window._fetch per Netdata host) and src/tessera/serving/timing_panel.py
telemetry truth (interval_unix shared bound, fast_power_samples, netdata raw).
Adds only an NTP-style clock-delta probe (HTTP Date header), which the worker
does not measure and which this issue must supply.
"""
import argparse
import email.utils
import json
import socket
import sys
import time
import urllib.request

sys.path.insert(0, ".")
sys.path.insert(0, "src")


def commit(units, phase):
    try:
        import prismabuild.progress as p
        p.commit(units, phase)
    except Exception:
        pass


def delta_probe(host, n=5):
    out = []
    url = "http://%s:19999/api/v1/info" % host
    for _ in range(n):
        try:
            c0 = time.time()
            with urllib.request.urlopen(url, timeout=15) as fh:
                fh.read(1 << 20)
                date = fh.headers.get("Date")
            c1 = time.time()
            if not date:
                out.append({"ok": False, "error": "no Date header"})
                continue
            srv = email.utils.parsedate_to_datetime(date).timestamp()
            mid = (c0 + c1) / 2.0
            out.append({"ok": True, "server_minus_client_s": srv - mid,
                        "rtt_s": c1 - c0})
        except Exception as exc:
            out.append({"ok": False,
                        "error": "%s: %s" % (type(exc).__name__, exc)})
    return out


def fetch_power_raw(host, after, before, points=0):
    from experiments import box_power_window as bpw
    got = bpw._fetch(host, "nvidia_smi.gpu_power_draw",
                     ("power_draw",), after, before, points)
    raw = got.get("raw_doc", got["doc"])
    return {"query": got["url"], "raw_response": raw,
            "returned_view": raw.get("view")}


def capture(args):
    import torch
    from experiments.routed_pair_oracle import PowerSampler
    if not torch.cuda.is_available():
        raise SystemExit("native capture needs a CUDA device")
    pre = {}
    for h in args.hosts:
        pre[h] = delta_probe(h, n=5)
    commit(1, "probe")
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    commit(1, "workload")
    sampler = PowerSampler(hz=10.0)
    sampler.start()
    t0 = time.time()
    end = t0 + args.seconds
    it = 0
    while time.time() < end:
        for _ in range(10):
            c = a @ b
        torch.cuda.synchronize()
        it += 10
        if it % 200 == 0:
            commit(it, "workload")
    t1 = time.time()
    sampler.stop_flag = True
    sampler.join(timeout=5)
    commit(it, "workload")
    fast = [[t, w] for t, w in sampler.samples if t0 <= t <= t1]
    post = {}
    for h in args.hosts:
        post[h] = delta_probe(h, n=5)
    commit(1, "clock")
    net = {}
    for h in args.hosts:
        try:
            net[h] = dict(fetch_power_raw(h, int(t0), int(t1) + 1, 0),
                          ok=True)
        except Exception as exc:
            net[h] = {"ok": False,
                      "error": "%s: %s" % (type(exc).__name__, exc)}
        commit(1, "fetch")
    out = {
        "mode": "capture",
        "host": socket.gethostname(),
        "device": torch.cuda.get_device_name(0),
        "interval_unix": [t0, t1],
        "sampler_source": sampler.source,
        "sampler_hz": 10.0,
        "matmuls": it,
        "matrix": [n, n, "bfloat16"],
        "fast_power_samples": fast,
        "clock_delta_pre": pre,
        "clock_delta_post": post,
        "netdata": net,
    }
    commit(1, "seal")
    print("TS1175-JSON-BEGIN")
    print(json.dumps(out))
    print("TS1175-JSON-END")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", default="sparky,sparklina")
    ap.add_argument("--seconds", type=float, default=90.0)
    args = ap.parse_args()
    args.hosts = [h for h in args.hosts.split(",") if h]
    capture(args)


main()
```
