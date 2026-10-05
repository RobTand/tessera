#!/bin/bash
# profiled_run.sh RCFILE SPEEDSCOPE CMD... : run CMD under py-spy record, with
# py-spy as CMD's parent (ptrace_scope 1 allows nothing else), and exit with
# CMD's own exit code, which py-spy does not return.
#
# CMD's code goes through RCFILE, written by a shell between py-spy and CMD.
# py-spy can return as soon as the profiled python exits, before that shell has
# written the file, and in a container whose init then exits the shell is
# killed with it.  So this waits for the file (up to 60 s) before exiting.
rcf=$1; out=$2; shift 2
rm -f "$rcf"
py-spy record --subprocesses --idle --nonblocking --rate "${PYSPY_RATE:-10}" --format speedscope \
  -o "$out" -- bash -c '"$@"; echo $? > "$0"' "$rcf" "$@"
spy=$?
for _ in $(seq 120); do [ -s "$rcf" ] && break; sleep 0.5; done
echo "[profiled_run] py-spy rc=$spy, command rc=$(cat "$rcf" 2>/dev/null || echo missing)"
[ -s "$rcf" ] && exit "$(cat "$rcf")"
exit 99
