#!/bin/bash
set -euo pipefail
umask 077
cd @TEMPLATE@
sha256sum -c template.sha256
refuse() { echo "$1" >&2; exit 3; }
test "$#" -eq @COUNT@ || refuse '@COUNT_MESSAGE@'
attempts=("$@")
patterns=(@PATTERNS@)
for index in "${!attempts[@]}"; do
attempt=${attempts[index]}
[[ "$attempt" =~ ${patterns[index]} ]] || refuse 'attempt path differs from fixed module and unit'
test "$(realpath -e -- "$attempt")" = "$attempt" || refuse 'attempt path is not canonical'
@PREREQUISITES@
@ABSENT@
done
claimed=() phase_mounts=()
task_root="${FLUX_SCRATCH:?}/${FLUX_JOB_ID:?}"
task_scratch="$task_root@SCRATCH_SUBDIR@"
name=tessera-dense-@PHASE@-"$FLUX_JOB_ID"
child=
cleanup() {
  rc=$?
  trap - EXIT
  if test -s "$task_scratch/containers.conf"; then
    podman rm -f "$name" >/dev/null 2>&1 || true
    podman unshare rm -rf -- "$task_scratch/podman-storage" "$task_scratch/podman-run" >/dev/null 2>&1 || true
  fi
  rm -rf -- "$task_root" || rc=3
  test ! -e "$task_root" || rc=3
  for output in "${claimed[@]}"; do printf '%s\n' "$rc" > "$output/tool-exit.txt"; done
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
export CONTAINERS_STORAGE_CONF="$task_scratch/storage.conf" CONTAINERS_CONF="$task_scratch/containers.conf"
printf '[storage]\ndriver = "overlay"\nrunroot = "%s/podman-run"\ngraphroot = "%s/podman-storage"\n[storage.options]\nadditionalimagestores = ["/home/rob/.local/share/containers/storage"]\n' "$task_scratch" "$task_scratch" > "$CONTAINERS_STORAGE_CONF"
printf '[engine]\ntmp_dir = "%s/podman-tmp"\n' "$task_scratch" > "$CONTAINERS_CONF"
git clone --shared @REPOSITORY@ "$task_scratch/code"
git -C "$task_scratch/code" checkout --detach @REVISION@
test "$(git -C "$task_scratch/code" rev-parse HEAD)" = @REVISION@
PYTHONPATH="$task_scratch/code" python3 -m experiments.numerics_rows.prepare_split --verify @ROOT@
for index in "${!attempts[@]}"; do python3 @ROOT@/instance.py check "${attempts[index]}" @PHASE@ > "$task_scratch/instance-$index.json"; done
for attempt in "${attempts[@]}"; do
  if mkdir "$attempt/@PHASE@"; then claimed+=("$attempt/@PHASE@"); else
    if test "${#claimed[@]}" -gt 0; then rmdir -- "${claimed[@]}"; claimed=(); fi
    refuse 'phase claim already exists or failed'
  fi
done
for index in "${!attempts[@]}"; do
  mv "$task_scratch/instance-$index.json" "${attempts[index]}/@PHASE@/instance.json"
  phase_mounts+=(-v "${attempts[index]}/@PHASE@:${attempts[index]}/@PHASE@:rw")
done
cd "${attempts[0]}/@PHASE@"
grep Cpus_allowed_list /proc/self/status > affinity.txt
@PREFLIGHT@
podman image inspect acc684c06ec7 > "$task_scratch/image-inspect.json"
mapfile -t population_mount < <(PYTHONPATH="$task_scratch/code" python3 -c 'import json,shlex; print("\n".join(shlex.split(json.load(open("@ROOT_RAW@/campaign.json"))["population_mount"])))')
rc=0
timeout --signal=TERM --kill-after=20s @SECONDS@s @CPU_LAUNCH@podman run --rm --name "$name" --pull=never --cgroups=disabled --network=none @GPU@ -v /mnt/shared:/mnt/shared:ro "${phase_mounts[@]}" -v "$task_scratch/code:/numerics:ro" "${population_mount[@]}" @OPTIONS@ --entrypoint /bin/bash acc684c06ec7 -euo pipefail @TEMPLATE@/payload.sh "${attempts[@]}" > payload.stdout 2> payload.stderr &
child=$!
wait "$child" || rc=$?
child=
for output in "${claimed[@]}"; do
  cd "$output"
  printf '%s\n' "$rc" > podman-exit.txt
  printf '%s\n' "$rc" > payload-exit.txt
  if test "$rc" -eq 0 || { test '@PHASE@' = compare && test "$rc" -eq 3 &&
    python3 -c 'import json; r=json.load(open("numerics_row.json"))["row"]; assert r.get("finalization_finished") is True and r["outcome"] in ("FAIL", "STOPPED")'; }; then
    test -s phase.json
    @TELEMETRY@
    find . -type f ! -name completion.sha256 -print0 | sort -z | xargs -0 sha256sum > completion.sha256
  fi
done
exit "$rc"
