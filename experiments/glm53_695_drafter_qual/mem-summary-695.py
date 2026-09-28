"""tessera#695: a serve arm's memory footprint, as the U4 runbook measures one.

footprint = MemAvailable at launch minus its in-run minimum, plus the swap-out
over the same span (the kernel freed that memory by swapping, so the drop alone
under-states the serve's need). Launch is the first sample, taken before the
container starts; the span ends at the last sample, after teardown.

  mem-summary-695.py SAMPLES OUT_JSON
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
page = None
rows = []
for line in open(src):
    if line.startswith("#"):
        page = int(line.split("page_bytes=")[1])
        continue
    parts = line.split()
    if len(parts) == 4:
        rows.append((float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
if len(rows) < 2 or page is None:
    raise SystemExit(f"{src}: {len(rows)} samples; no footprint")
gib = 1 << 30
launch = rows[0]
low = min(rows, key=lambda r: r[1])
swap_out = (rows[-1][2] - launch[2]) * page
swap_in = (rows[-1][3] - launch[3]) * page
drop = (launch[1] - low[1]) * 1024
record = dict(
    samples=len(rows), span_s=round(rows[-1][0] - launch[0], 1),
    memavailable_launch_gib=round(launch[1] * 1024 / gib, 2),
    memavailable_min_gib=round(low[1] * 1024 / gib, 2),
    min_at_s=round(low[0] - launch[0], 1),
    memavailable_drop_gib=round(drop / gib, 2),
    swap_out_gib=round(swap_out / gib, 3), swap_in_gib=round(swap_in / gib, 3),
    footprint_gib=round((drop + swap_out) / gib, 2),
    memavailable_end_gib=round(rows[-1][1] * 1024 / gib, 2),
)
json.dump(record, open(dst, "w"), indent=1)
print("footprint", record)
