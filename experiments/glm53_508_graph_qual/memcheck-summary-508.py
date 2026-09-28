"""tessera#508: summarize the CUDA memcheck logs of one serve arm.

srv-508.sh SANITIZE=1 writes one log per process the serve starts
($OUT/<arm>.memcheck.<pid>.log). A process that is killed at teardown never
prints its ERROR SUMMARY line, so the error records themselves are counted
too, by kind and by the kernel they were reported in, with the first reported
address offsets. Writes $OUT/<arm>.memcheck.summary.json.

  memcheck-summary-508.py RECEIPTS_DIR ARM
"""
import collections
import json
import pathlib
import re
import sys

recs, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
report = {"arm": arm, "logs": {}}
for log in sorted(recs.glob(f"{arm}.memcheck.*.log")):
    text = log.read_text(errors="replace")
    kinds = collections.Counter(re.findall(r"=========\s+(Invalid __\w+__ \w+ of size \d+ bytes)", text))
    kernels = collections.Counter(re.findall(r"=========\s+at (\S+)\+0x[0-9a-f]+ in (\S+)", text))
    addresses = re.findall(r"Address (0x[0-9a-f]+) is (out of bounds|misaligned)[^\n]*", text)
    nearest = re.findall(r"and is (\d+) bytes (after|before) the nearest allocation at (0x[0-9a-f]+) of size (\d+) bytes",
                         text)
    summary = re.findall(r"ERROR SUMMARY: (\d+) error", text)
    report["logs"][log.name] = dict(
        bytes=len(text),
        error_summary=[int(n) for n in summary],
        error_records=sum(kinds.values()),
        kinds=dict(kinds),
        sites=[dict(site=f"{fn} in {src}", count=n) for (fn, src), n in kernels.most_common(8)],
        first_addresses=[a for a, _ in addresses[:5]],
        nearest_allocation=[dict(offset=int(o), side=s, base=b, size=int(z)) for o, s, b, z in nearest[:5]],
        target_filter=re.findall(r"--kernel-name[= ]\S+", text)[:1],
    )
total = sum(v["error_records"] for v in report["logs"].values())
report["error_records_total"] = total
report["complete_summaries"] = {k: v["error_summary"] for k, v in report["logs"].items() if v["error_summary"]}
(recs / f"{arm}.memcheck.summary.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "logs"}, indent=1))
for name, v in report["logs"].items():
    print(name, v["error_records"], v["kinds"], v["sites"][:2], v["nearest_allocation"][:1])
