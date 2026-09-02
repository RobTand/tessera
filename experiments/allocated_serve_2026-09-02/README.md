# Drivers for the allocated-vs-uniform serve (2026-09-02)

These are the scripts that produced sections 6-7 of
`docs/measurements/tessera-allocated-served-2026-09-02.md` -- the separator
pair, the two mechanism arms, the depth predictor and the KL tables.  They ran
from `/home/rob/tmp/alloc-plans/`, which is scratch; the receipt cites them, so
they live here instead, verbatim as run.

They are **drivers, not a supported interface**: paths to checkpoints, plans
and logs under `/home/rob/tessera-runs/` and `/mnt/shared/tessera-runs/` are
hard-coded, and nothing here is imported by `src/`.  Read them to reproduce or
audit the measurement, not to build on.
