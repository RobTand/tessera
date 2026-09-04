#!/usr/bin/env bash
# The four censuses of the #91 reproduction, in one process.
#
# A file rather than a ``bash -c`` string because the pool's action contract
# refuses an argv element containing a control character, and a newline is
# one: a multi-line command is not something a ledger can record as an argv.
set -x
"$WT/experiments/ts91_cache_key_repro.sh" A "X-$SUF" compiled "$SUF-A-into-X"
"$WT/experiments/ts91_cache_key_repro.sh" B "Y-$SUF" compiled "$SUF-B-into-Y"
"$WT/experiments/ts91_cache_key_repro.sh" B "X-$SUF" compiled "$SUF-B-into-X"
"$WT/experiments/ts91_cache_key_repro.sh" A "Y-$SUF" compiled "$SUF-A-into-Y"
