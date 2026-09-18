#!/usr/bin/env bash
# every 2 s: time, MemFree/MemAvailable/Cached (MiB), PSI full avg10, compaction/migration counters
OUT=$1; SECS=${2:-400}
end=$(( $(date +%s) + SECS ))
while [ "$(date +%s)" -lt "$end" ]; do
  awk -v t="$(date +%H:%M:%S)" '/MemFree|MemAvailable|^Cached/{v[$1]=int($2/1024)} END{printf "%s free=%d avail=%d cached=%d ", t, v["MemFree:"], v["MemAvailable:"], v["Cached:"]}' /proc/meminfo
  awk '/^full/{split($2,x,"="); printf "psi_full10=%s ", x[2]}' /proc/pressure/memory
  awk '/compact_stall|compact_fail|pgmigrate_fail|pgmigrate_success|thp_fault_alloc|compact_isolated/{printf "%s=%s ", $1, $2}' /proc/vmstat
  echo
  sleep 2
done >> "$OUT"
