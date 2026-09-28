#!/usr/bin/env bash
# tessera#695: sample the box's memory at 1 Hz until STOP appears, for a serve
# arm's footprint: MemAvailable (kB), the cumulative swap-out page count
# (/proc/vmstat pswpout) and the swap-in count. mem-summary-695.py reads it.
#   mem-sampler-695.sh OUT STOP
OUT=$1; STOP=$2
echo "# unix_s memavailable_kb pswpout pswpin page_bytes=$(getconf PAGESIZE)" > "$OUT"
while [ ! -e "$STOP" ]; do
  echo "$(date +%s.%N | cut -c1-14) $(awk '/^MemAvailable:/{print $2}' /proc/meminfo)" \
       "$(awk '/^pswpout /{o=$2} /^pswpin /{i=$2} END{print o, i}' /proc/vmstat)" >> "$OUT"
  sleep 1
done
