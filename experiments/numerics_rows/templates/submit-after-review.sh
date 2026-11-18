#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
root=@TEMPLATE@
test "$#" -eq 1 || { echo 'require one attempt argument' >&2; exit 3; }
python3 @ROOT@/instance.py review "$root"
python3 @ROOT@/instance.py submit-template "$root" "$1" submit --urgency=8 --job-name=tessera-1717-@KEY@-@PHASE@ --dependency=singleton -n1 -t@MINUTES@m --resources-shape='@SHAPE@' --requires=host:sparklina --cwd=@TEMPLATE@ --input=/dev/null '--env=-*' --env=PATH=/usr/bin:/bin --env=HOME=/home/rob --env=XDG_RUNTIME_DIR=/run/user/1000 --env=FLUX_F58_FORCE_ASCII=1 --env=FLUX_SCRATCH=/var/tmp/flux-rob --env=OMP_NUM_THREADS=1 --env=MKL_NUM_THREADS=1 --env=OPENBLAS_NUM_THREADS=1 --output=kvs --error=kvs /bin/bash @TEMPLATE@/job.sh "$1"
