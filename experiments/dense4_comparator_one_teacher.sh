#!/usr/bin/env bash
# Put the NVFP4 comparator and the LDLQ bracket on ONE teacher.
#
# The comparator's published 0.5105764 was scored against qwen_teacher_bf16_v028
# (sparky); the #60 LDLQ bracket scores against qwen_rot_teacher_lina
# (sparklina).  Two BF16 teachers of the same model on the same corpus ought to
# agree, but "ought to" is not a measurement, and a matched pair whose two sides
# were scored against different teacher dumps is not matched.  So:
#   (1) re-score the comparator against the bracket's teacher, and
#   (2) score one teacher against the other, which is the size of the confound.
set -euo pipefail
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
KL=${KL:-/home/rob/dq-runs/kl_tool.py}
D=/mnt/shared/tessera-kl
OUT=${OUT:-/mnt/shared/tessera-runs/ldlq-block}
mkdir -p "$OUT"

$PY -u "$KL" compare "$D/qwen_rot_teacher_lina.json.npz" \
    "$D/qwen_stock_nvfp4-prod.json.npz" --out "$OUT/ts12_kl_nvfp4prod_vs_lina.json"

$PY -u "$KL" compare "$D/qwen_rot_teacher_lina.json.npz" \
    "$D/qwen_teacher_bf16_v028.json.npz" --out "$OUT/ts12_kl_teacher_vs_teacher.json"

$PY - <<'PY'
import json
for tag, p in (("nvfp4-prod vs lina teacher",
                "/mnt/shared/tessera-runs/ldlq-block/ts12_kl_nvfp4prod_vs_lina.json"),
               ("teacher_v028 vs teacher_lina",
                "/mnt/shared/tessera-runs/ldlq-block/ts12_kl_teacher_vs_teacher.json")):
    d = json.load(open(p))
    print(f"{tag}: all={d['all']['kl_lower_mean']!r} "
          f"confident={d['confident']['kl_lower_mean']!r} "
          f"top1={d['all']['top1_agree_pct']!r} n={d['positions']}")
PY
