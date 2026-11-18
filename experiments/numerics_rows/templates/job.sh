#!/bin/bash
set -euo pipefail
umask 077
cd @TEMPLATE@
sha256sum -c template.sha256
refuse() { echo "$1" >&2; exit 3; }
test "$#" -eq 1 || refuse 'require one attempt argument'
attempt=$1
[[ "$attempt" =~ @PATTERN@ ]] || refuse 'attempt path differs from fixed module and unit'
id=${BASH_REMATCH[1]}
test "${#id}" -le @MAX_DIGITS@ && test "$id" -le @MAX_ATTEMPTS@ || refuse 'attempt id exceeds MAX_ATTEMPTS'
test "$(realpath -e -- "$attempt")" = "$attempt" || refuse 'attempt path is not canonical'
@PREREQUISITES@
mkdir "$attempt/@PHASE@" || refuse 'phase claim already exists or failed'
cd "$attempt/@PHASE@"
task_scratch="${FLUX_SCRATCH:?}/${FLUX_JOB_ID:?}"
name=tessera-dense-@PHASE@-"$FLUX_JOB_ID"
child=
cleanup() {
  rc=$?
  trap - EXIT
  if test -s "$task_scratch/containers.conf"; then
    podman rm -f "$name" >/dev/null 2>&1 || true
    podman unshare rm -rf -- "$task_scratch/podman-storage" "$task_scratch/podman-run" >/dev/null 2>&1 || true
  fi
  rm -rf -- "$task_scratch" || rc=3
  test ! -e "$task_scratch" || rc=3
  printf '%s\n' "$rc" > tool-exit.txt
  exit "$rc"
}
stop() {
  trap '' TERM INT
  if test -n "$child"; then kill -TERM "$child" 2>/dev/null || true; fi
  if test -s "$task_scratch/containers.conf"; then podman stop --time 1 "$name" >/dev/null 2>&1 || true; fi
  exit "$1"
}
trap cleanup EXIT
trap 'stop 143' TERM
trap 'stop 130' INT
export HOME="$task_scratch/home" TMPDIR="$task_scratch/tmp"
export XDG_CACHE_HOME="$task_scratch/cache" XDG_CONFIG_HOME="$task_scratch/config" XDG_DATA_HOME="$task_scratch/data"
export PYTHONPYCACHEPREFIX="$task_scratch/host-pycache"
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
python3 @ROOT@/instance.py check "$attempt" @PHASE@ > instance.json
export CONTAINERS_STORAGE_CONF="$task_scratch/storage.conf" CONTAINERS_CONF="$task_scratch/containers.conf"
printf '[storage]\ndriver = "overlay"\nrunroot = "%s/podman-run"\ngraphroot = "%s/podman-storage"\n[storage.options]\nadditionalimagestores = ["/home/rob/.local/share/containers/storage"]\n' "$task_scratch" "$task_scratch" > "$CONTAINERS_STORAGE_CONF"
printf '[engine]\ntmp_dir = "%s/podman-tmp"\n' "$task_scratch" > "$CONTAINERS_CONF"
grep Cpus_allowed_list /proc/self/status > affinity.txt
git clone --shared @REPOSITORY@ "$task_scratch/code"
git -C "$task_scratch/code" checkout --detach @REVISION@
test "$(git -C "$task_scratch/code" rev-parse HEAD)" = @REVISION@
PYTHONPATH="$task_scratch/code" python3 -m experiments.numerics_rows.prepare_split --verify @ROOT@
@PREFLIGHT@
podman image inspect acc684c06ec7 > "$task_scratch/image-inspect.json"
mapfile -t population_mount < <(PYTHONPATH="$task_scratch/code" python3 -c 'import json,shlex; print("\n".join(shlex.split(json.load(open("@ROOT_RAW@/campaign.json"))["population_mount"])))')
rc=0
timeout --signal=TERM --kill-after=20s @SECONDS@s podman run --rm --name "$name" --pull=never --cgroups=disabled --network=none @GPU@ -v /mnt/shared:/mnt/shared:ro -v "$PWD:$PWD:rw" -v "$task_scratch/code:/numerics:ro" "${population_mount[@]}" @OPTIONS@ --entrypoint /bin/bash acc684c06ec7 -euo pipefail @TEMPLATE@/payload.sh "$attempt" > payload.stdout 2> payload.stderr &
child=$!
wait "$child" || rc=$?
child=
printf '%s\n' "$rc" > podman-exit.txt
printf '%s\n' "$rc" > payload-exit.txt
test "$rc" -eq 0
test -s phase.json
@TELEMETRY@
find . -type f ! -name completion.sha256 -print0 | sort -z | xargs -0 sha256sum > completion.sha256
