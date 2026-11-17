# Per-phase Flux binding

This change covers I1, I2 and the tensor-only I3 check.
No job was submitted. No host test or numeric model import ran.
The CPU job must include tests/test_numerics_phase_instance.py and tests/test_numerics_rows_phases.py.
The tests have not run. Source inspection is not a passing test verdict.

## Tensor construction

HostDense constructs a fresh adapter for each group in dense_phases.py:132.
Its wire decode and shard plan depend on the module and TP cut at dense_phases.py:29.
row.py:431 constructs sweep input from M, column count and recipe.
The recipe batching depends on that role and M at row.py:474.
The profile input uses an empty recipe at dense_phases.py:151.
row.py:974 constructs a new random generator for each diagnostic case.
It seeds the generator with the explicit seed 0, 1 or 2.
Distribution transforms depend on M and that case's input.
scale_launch.py:148 creates its memo within the unit check.
Its input, coded scale, reference and finite certificate come from the local role, plan, read and coding.
CaseWriter.scale stores array_index from that unit's scale arrays at dense_phases.py:105.
Artifact.add assigns the global ordinal only after it receives all tensors at phase_artifact.py:62.
No inspected tensor path reads the ordinal, the case ID, common identities or a prior cut's random state.
This establishes local inputs by source inspection. The E/P check must still verify the actual bytes.

## Receipt and artifact fields

Each phase adds these fields under row.phase_jobs.PHASE:

- name, one of fv1, fv1e-sparklina or fv1e-sparky.
- config_sha256, the SHA-256 of the rank-zero config as sorted compact JSON.
- broker_starttime, read from rank zero.
- uri, the live rank-zero local-uri. This is the broker endpoint, not a saved client environment.
- job_id, normalized to f58plain.

Existing phase row measurements remain beside these fields in the final compare receipt.
An unsplit timed row uses row.phase_jobs.timed with the same fields.
No field is added to phase_artifact.identities().
Sweep, profile, chunk-pair and split-column case records gain recipe.
This is case metadata. No tensor, numeric value, rule or pass condition changes.

The submit record is PHASE/instance.json. The job writes its checked copy to PHASE/results/instance.json.
Submission uses urgency zero until the instance and job ID are recorded.
The helper checks the instance again before it restores the requested urgency.
Both observations read config.get and attr.get from broker rank zero.
The job checks its actual Flux connection before it starts the container.
The INSTANCE file supplies only the native client installation prefix during that check.
Its file hash and the fv1 resolver no longer choose or bind a phase instance.

The feeder removes state.instance_sha256 and adds state.settings_sha256.
state.jobs is keyed by name:job_id. Each entry adds instance with the five fields above.
state.lanes records the selected instance and host for each prepare slot.
state.ep_tensor_check records cases and tensor_map_sha256 after I3 passes.

## Placement and barrier

The canary keeps all three phases on fv1. Capture, compare and timed rows use fv1.
Counted prepare waves use up to nine fv1e slots per available Spark.
Each fv1e prepare books 6 GiB. The pool limit is 72 GiB.
Each Spark also has at most one fv1 prepare slot under the existing pending-work yield check.
These numbers come from feeder-settings.json.
Before the wave, the feeder reads column 6 of the canary prepare's jobmem TSV files.
Missing evidence, a zero peak or a peak above the configured 5.0 GiB limit refuses the fv1e wave.
The first fv1e prepare is the counted full module that contains the canary's TP2 rank-0 scope.
The feeder collects it and runs I3 before it submits any other counted prepare.
A failed instance query does not select that E-core instance for a new feeder state.
A saved phase cannot silently move to another instance on resume.

## Tensor-only comparison

The canary scope set selects counted cases by the complete unit_scope value.
The local key contains every case field except the exclusions below.
It therefore includes scope, configuration, kind, category, recipe, distribution, seed and all scale-case parameters present.
The comparison retains duplicate multiplicity for a local key.
For every matched case it compares the complete tensor-name map, SHA-256, dtype and shape.
It reads each referenced payload and checks its SHA-256. A matching declared hash alone is insufficient.
It requires the same local keys and multiplicities within the canary scopes.
Other counted cuts and profiler files do not enter the comparison.

The excluded case fields are id, ordinal, identities, tensors, seconds, utc_start, utc_end,
elapsed_seconds, worker_elapsed_seconds, host, hostname, job_id, flux_job_id and instance.
The tensors field supplies the compared tensor map, rather than the local key.
The embedded identities field is excluded in full, including selected_cuts and canary.
Tensor file paths do not enter the comparison. Payload hashes, dtypes and shapes do.

## Tests and overlap

New tests are test_prepare_capture_bind_instances_and_hold_until_recorded and
test_ep_tensor_subset_ignores_records_and_refuses_tensor_changes.
The latter has byte, dtype, shape and seed mutations. Different ordinals, identities and times pass.
test_observe_uses_rank_zero_config executes the RPC code with stub brokers.
The submit and job brokers have different local configs and the same rank-zero config.
Both observations must give the same identity and the rank-zero config hash.
A foreign-cut case is ignored. A changed key or tensor refuses the comparison.
The existing feeder fixtures now save instance-local job IDs and identities.

The change overlaps Sol in row.py, counted_feeder.py and tests/test_numerics_rows_phases.py.
It does not edit run_routed.py. Sol's separation and drain fixes must remain in the combined head.
The new paths are phase_instance.py and tests/test_numerics_phase_instance.py.
