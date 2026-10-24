# Shared-add fold: existing campaign recovery (tessera#1360)

## 2026-10-24 — blocked evidence handoff, not served completion

This report continues tessera#1204; it creates no independent campaign or new
fold. O1/O2 report accountability is separate from O3 served acceptance. No
resolving commit, READY pair, numerical equality, add removal or measured savings
is claimed. Defaults are unchanged; promotion is out of scope.

### Ownership and admission

The supplied RobTand comment on tessera#1360 (2026-10-23T19:33:32Z,
comment 6373135202) says the attributed decision and new READY or authorized
repricing evidence remain absent. No later decision was delivered here. The
historical 120-GPU-minute reference is not current authorization. The existing
campaign owner must supply the attributed decision, reconcile the directly
overlapping tessera#1476 served receipts owner, and consume W1's admitted
repaired-artifact/readiness handoff. Neither handoff nor reconciliation is
established. No supported command for obtaining an external owner decision or
running this exact two-rank serving campaign was supplied; none is invented.
This is a precise evidence dependency, not a request to repair pytest or deploy.

A focused read of the campaign directory and applicable repository documentation
found historical manifests and failed base attempts, not a current W1 admission.
The historical artifact path in inputs.json does not establish repaired bytes,
readiness or qualification. Its full contents were not established. Repository
issue snapshots have different titles for some issue numbers; the supplied open
issue ownership list governs this task, not those stale snapshots.
Reuse tessera#799's existing opt-in implementation, with source qualification
owned in the overlapping tessera#1475 lane. tessera#1341 graph/all-reduce and
tessera#1339 MTP outcomes remain different acceptances, even if infrastructure
can be reused. No new issue or competing worker was started.

### Parent recovery and reuse

All six retained attempts are indexed below. They are not six valid baselines.
The first lacks terminal diagnostics; the second reports a missing container
without establishing cause; the next three expose mount/directory permissions;
the latest refuses VllmConfig validation because expert_ids is required. None
is classified as OOM or application READY. Reuse diagnostic/configuration bytes
only. Original process commands and numerical exit codes are not established
by these logs: do not invent an exit status from a traceback or p.failed marker.
No valid matched gate, successful TR3 result, trace pair or telemetry was found.

### Configuration and execution population

The retained latest inputs pin runtime
`4c0fd6aca04f1c3e95b094ae006c2479043c3372`, image
`sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`,
PrismaQuant `03683363699c2f421711f37aa96cb0f71469c33e`, prompt, teacher,
second-teacher and roster identities, and **5400 seconds**. Input digests are
claims from that retained manifest, not newly verified artifact admission.
Exact scorer/peer argv arrays are retained in engine.json; they are not proof
those commands completed.

Serving rank0 records TP2 / two nodes / mp, compilation NONE, MTP speculation,
one speculative token and draft TP2, max_model_len=8448, batch tokens=2048,
max_num_seqs=1, chunked prefill on, prefix caching off, triton MoE,
fp8_ds_mla KV, KV bytes=2147483648 and GPU utilization=0.5. Profiler settings
include torch shapes and gzip, active_iterations=5. In contrast engine.json's
scorer specifies eager, max_model_len=2049, batch tokens=2049, chunked prefill
off and KV bytes=1073741824; it does not specify the serve's MTP configuration.
Do not substitute that scorer configuration for the actual serving inputs.
Both arms must bind the same admitted repaired artifact digests, runtime,
requests, topology, source and parent measurement settings; only
`TESSERA_GLM53_FOLD_SHARED_ADD` changes. Preserve speculation. Request counts
and completed work are not established merely by identities/window settings;
recover original population from the owner, never shorten it to fit a pilot.

### GPU decision

Decision: **do not run**; zero new GPU experiments and zero GPU minutes requested.
CPU can validate report accounting but cannot decide served TP2 TR3 equality,
actual eligible folding, per-rank kernel removal, latency or both-host work/J.
After admission/authorization, the smallest useful experiment is one targeted
matched off/on pair through the existing campaign, not a sweep. Before each
such run its owner must record the decision, CPU limitation, exact population,
smallest case and estimated GPU minutes. An estimate is not responsibly fixed
here without the execution handoff: two 5400-second windows already imply 180
elapsed minutes (360 GPU-minutes for two GPUs), before setup/scoring; reconcile
budget/population with the owner rather than treating the historical 120-minute
citation or a ten-minute pilot as full acceptance. No run is authorized here.

### Immutable evidence index

SHA-256 values below bind retained referenced file bytes (not just mutable path
names). They do not by themselves authenticate admission or owner authority.
Unavailable slots may reference diagnostic/context bytes without claiming the
missing receipt exists. Append later observations; do not erase these failures.

```json
{
  "campaign": "tessera#1204",
  "date": "2026-10-24",
  "runtime": "4c0fd6aca04f1c3e95b094ae006c2479043c3372",
  "parent_recovery": [
    {
      "run_id": "fold1204-base-20261010T052517Z",
      "outcome": {
        "status": "reused",
        "detail": "Insufficient terminal diagnostics; no valid baseline established. Reuse inputs only, not a successful measurement.",
        "refs": [
          {
            "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T052517Z/inputs.json",
            "sha256": "0f04036f008552a2f7bb2f153051fd048233adc352ea3fbf6945405a410fe046"
          }
        ]
      }
    },
    {
      "run_id": "fold1204-base-20261010T053713Z",
      "outcome": {
        "status": "reused",
        "detail": "Rank1 log reports No such container; underlying cause not established. Not READY; no numerical result. Reuse diagnostic only.",
        "refs": [
          {
            "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T053713Z/fold-base/logs/p-rank1.log",
            "sha256": "cdc6ff681d191c33dcf11a40b3cf59b0ce41c5b1d53426348d4303160d28ae92"
          }
        ]
      }
    },
    {
      "run_id": "fold1204-base-20261010T055400Z",
      "outcome": {
        "status": "reused",
        "detail": "Docker rank1 mount-source mkdir permission denied. Not READY; reuse diagnostic only.",
        "refs": [
          {
            "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T055400Z/docker-p1.log",
            "sha256": "1670a56e133a73d5572694889859fc60839269f1fd774a81eb6bec0ec33d2ec7"
          }
        ]
      }
    },
    {
      "run_id": "fold1204-base-20261010T061025Z",
      "outcome": {
        "status": "reused",
        "detail": "Rank1 cannot create /ext/tessera and /ext/tmp: Permission denied. Not READY; reuse diagnostic only.",
        "refs": [
          {
            "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T061025Z/fold-base/logs/p-rank1.log",
            "sha256": "33ddfa04c120d5473426cc14fff362ff24c455c54fabf6aa391afb763aaa0194"
          }
        ]
      }
    },
    {
      "run_id": "fold1204-base-20261010T062735Z",
      "outcome": {
        "status": "reused",
        "detail": "Rank0 cannot create /ext/tessera and /ext/tmp: Permission denied. Not READY; reuse diagnostic only.",
        "refs": [
          {
            "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T062735Z/fold-base/logs/p-rank0.log",
            "sha256": "33ddfa04c120d5473426cc14fff362ff24c455c54fabf6aa391afb763aaa0194"
          }
        ]
      }
    },
    {
      "run_id": "fold1204-base-20261010T071354Z",
      "outcome": {
        "status": "reused",
        "detail": "Rank0 VllmConfig validation rejects model.language_model.layers.10.mlp.experts: expert_ids is required. Refusal, not OOM or READY; reuse failed-start diagnostic and pinned inputs, not baseline measurement.",
        "refs": [
          {
            "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/fold-base/logs/p-rank0.log",
            "sha256": "2cd740e418053ce67b0fefbb4a4be59f07a100bb0ca51b36adfe10fd3e56f861"
          },
          {
            "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/inputs.json",
            "sha256": "97e0e87cff428452d46e5eeb310acb40dfa500288bfad9286bbafd5570c9a6a0"
          }
        ]
      }
    }
  ],
  "evidence": {
    "owner_decision": {
      "status": "unavailable",
      "detail": "No current attributed owner decision or authorized repricing/READY receipt delivered. Supplied RobTand comment 6373135202 at 2026-10-23T19:33:32Z explicitly records their absence. Historical budget is not current authority. Campaign #1204 owner must reconcile overlapping #1476 before any run; no supported external owner-recovery command/interface was supplied.",
      "refs": []
    },
    "w1_artifact_readiness": {
      "status": "unavailable",
      "detail": "No current W1 admitted repaired-artifact digest manifest and readiness handoff was delivered or found in the inspected campaign packet. Historical artifact entrypoint and SUMMARY accounting are not admission; full artifact bytes were not inspected.",
      "refs": []
    },
    "source_qualification": {
      "status": "unavailable",
      "detail": "Existing #799 implementation and architecture are reused as requirements only; no complete source-bound CPU/CUDA/unchanged-kernel numeric and SASS receipt for the admitted repaired artifact was established. #1475 owns overlapping local qualification.",
      "refs": [
        {
          "uri": "file:///home/rob/.local/state/trustladder-live/worktrees/tessera-183856408f7d-wp2/src/tessera/serving/glm53_shared_fold.py",
          "sha256": "204f4768818b1182d0df2b8390daf140c5d090398f646b4554f76967a06c0153"
        },
        {
          "uri": "file:///home/rob/.local/state/trustladder-live/worktrees/tessera-183856408f7d-wp2/docs/ARCHITECTURE.md",
          "sha256": "fb28d0f08d5eb471a6ed410682934063f8ce4ccfb71aed0e6652e70f992f0987"
        }
      ]
    },
    "matched_arm_config": {
      "status": "unavailable",
      "detail": "Historical base inputs and serving log plus scorer engine configuration recovered, but no admitted repaired artifact pair/configuration equality receipt. Preserve serving MTP, one speculative token, draft TP2; scorer engine.json is distinct, not a replacement serve configuration.",
      "refs": [
        {
          "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/inputs.json",
          "sha256": "97e0e87cff428452d46e5eeb310acb40dfa500288bfad9286bbafd5570c9a6a0"
        },
        {
          "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/engine.json",
          "sha256": "e20f2c4bb9d4c015237bfe631dc88036dd47ec280d567a2678dc14ffef53ddf2"
        },
        {
          "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/fold-base.env",
          "sha256": "1d9e6b7fdccf6fa39acacc3e71fe3f77032b4de3097359009bad22b673e76e73"
        },
        {
          "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/fold-base/logs/p-rank0.log",
          "sha256": "2cd740e418053ce67b0fefbb4a4be59f07a100bb0ca51b36adfe10fd3e56f861"
        }
      ]
    },
    "base_readiness": {
      "status": "observed",
      "detail": "Latest historical base refused VllmConfig validation: expert_ids is required. This observes refusal only; application READY and successful numerical execution absent.",
      "refs": [
        {
          "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/fold-base/logs/p-rank0.log",
          "sha256": "2cd740e418053ce67b0fefbb4a4be59f07a100bb0ca51b36adfe10fd3e56f861"
        }
      ]
    },
    "base_rank0_log": {
      "status": "reused",
      "detail": "Latest base rank0 retained failed startup log; no fold or TR3 success inferred.",
      "refs": [
        {
          "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/fold-base/logs/p-rank0.log",
          "sha256": "2cd740e418053ce67b0fefbb4a4be59f07a100bb0ca51b36adfe10fd3e56f861"
        }
      ]
    },
    "base_rank1_log": {
      "status": "unavailable",
      "detail": "Latest attempt has no retained fold-base/logs/p-rank1.log in inspected logs directory. Earlier rank1 permission/missing-container diagnostics do not supply a matched rank1 readiness log.",
      "refs": []
    },
    "gate_readiness": {
      "status": "not_run",
      "detail": "No matched served gate READY receipt; no gate launched in this implementation work.",
      "refs": []
    },
    "gate_rank0_log": {
      "status": "unavailable",
      "detail": "No matched gate rank0 log retained in inspected parent packet.",
      "refs": []
    },
    "gate_rank1_log": {
      "status": "unavailable",
      "detail": "No matched gate rank1 log retained in inspected parent packet.",
      "refs": []
    },
    "gate_install_and_folded_calls": {
      "status": "unavailable",
      "detail": "No gate installation plus first actual eligible folded-call logs. A declined installation or rank-up marker cannot prove folding.",
      "refs": []
    },
    "tr3_outputs_and_baseline_comparison": {
      "status": "unavailable",
      "detail": "No successful matched TR3 outputs or baseline-identical KL comparison. Configured output paths and peer_roundtrip_equal are not numerical execution.",
      "refs": []
    },
    "rank0_base_gate_traces": {
      "status": "unavailable",
      "detail": "No rank0 matched base/gate torch trace pair showing eligible aten::add [2048,4096] removal.",
      "refs": []
    },
    "rank1_base_gate_traces": {
      "status": "unavailable",
      "detail": "No rank1 matched base/gate torch trace pair showing eligible aten::add [2048,4096] removal.",
      "refs": []
    },
    "latency": {
      "status": "unavailable",
      "detail": "No original-population matched latency measurement; startup diagnostics cannot establish savings.",
      "refs": []
    },
    "both_host_power_and_work_per_joule": {
      "status": "unavailable",
      "detail": "No matched both-host Netdata power, integrated joules and completed work/J receipts. No energy saving established.",
      "refs": []
    },
    "execution_population": {
      "status": "observed",
      "detail": "Inputs specify original window_seconds=5400 and prompt/panel/teacher/roster identities. Completed served population is unestablished; no pilot, truncation or zero-work sample is substituted for full acceptance. This is observation of configured population, not execution.",
      "refs": [
        {
          "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/inputs.json",
          "sha256": "97e0e87cff428452d46e5eeb310acb40dfa500288bfad9286bbafd5570c9a6a0"
        },
        {
          "uri": "file:///mnt/shared/tessera-measurements/fold-tp2-1204/fold1204-base-20261010T071354Z/engine.json",
          "sha256": "e20f2c4bb9d4c015237bfe631dc88036dd47ec280d567a2678dc14ffef53ddf2"
        }
      ]
    }
  },
  "disposition": "blocked",
  "default_changed": false,
  "promotion_decision": "out_of_scope"
}
```

## 2026-10-24 — appended report verification (CPU only)

Historical research pre-change receipt: `tests/test_shared_fold_measurement_report.py:20`
failed with `missing dated shared-fold campaign report` in both tests (exit 1,
2 failed). The retained stdout was recovered with read_test_artifact; it confirms
assertion execution, not just collection. Research also supplied a cheap-guard
receipt: 212 passed, exit 0. Both research runs reported CPU torch, NO CUDA,
zero skips, zero uncollected modules and zero device allocations. The brief
supplied the receipts below; the exact historical invocation text was not
retained in those stdout bytes and is not reconstructed as fact.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9e2fa98a1c90d3506ea69e4fd8fb3be537b611b62c8c4a94dca1ce91030578ea","size":2972}},"operation":"f8943f649f401743577475145ce2fce9871b400d504cd88f4c83027423692742","receipt":"3a9f3c358e9b51c87c22bfecd37250e142898a798a8a7fa3471549d53378b1b6"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"41c6c435a0656e1f8650873142a3db7fcaf9a6ae411061a9aba2e178cec24942","size":1787}},"operation":"7b9fdd74a287b84f5e9c1e170e81cb23e70c2115136c6ba18ac99ad13bb2c4ca","receipt":"67c9df21428dbf49973d444b74836f7d642ecd25c7b15b8aa01db74146293861"}]

New checks used plain commands through run_tests, default x86 arm:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_shared_fold_measurement_report.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_issue_refs.py tests/test_impacted_tests.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

First command: exit 0, 2 passed in 8.92s, covering both O1/O2 named tests.
Second: exit 0, 212 passed in 113.02s. Both: torch 2.11.0+cpu,
NO CUDA, zero skipped tests, zero uncollected modules, zero device allocations.
There are no skip reasons because none skipped. These checks exercise report
accounting and cheap guards only: passing does not authenticate owner decisions,
admission, configuration matching, numerical equality or savings. No GPU test
run was justified for this documentation change; CUDA qualification and served
O3 remain unavailable. The once-only full integration run and publication remain
with TrustLadder.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3b84c94721e90839269172204b87a4ec37272c0802ee8fe4c7db0b3d38dfa863","size":713}},"operation":"760710dbe928c0c4cd46d5517f23a0ad6442c9ef9213d9561c0c18cc7a7029bf","receipt":"72ff984b03fb1433318cacf373a55b042b5684c4d6d01f51cd1fb11e76bdf5f3"}]
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6cb9cf1c9104163be2b74c190d6bafd576f6c5997d75d4c3b36bc9e5a29436a1","size":1778}},"operation":"7e1d296b7c643005b1ed8df35fd3c842a0be6473e2d78cccbd21f6554b68706c","receipt":"44c571344aae22dcff0959c86a4cfc5331ff4e5297748f2a696b075e7f43fc30"}]

The research selector's historical `no changes` result was not a changed-tree
receipt: its added test was then untracked. Do not use it as evidence that this
report has no impacted checks.

Current selector command (exit 0):
`python3 tools/impacted_tests.py --root "$PWD" --ref 17b3fdb396410698c9a142615345f046d2f94a5d --json`
was executed with the absolute checkout root in place of `$PWD`. It reported
`changed: 1`, `verdict: narrowed`, including the report test and a broad
conservative reader/import closure. This is selection, not execution of that
closure; only the two targeted check commands above were run. No full suite is
claimed. Final inspection found only this new report (354 lines before this
append), no production/configuration/test edits. `git diff --check` returned 0;
the new-file `git diff --no-index --stat /dev/null docs/measurements/shared-add-fold-1360.md`
returned 1 as expected for a nonempty added-file diff, not a test failure.
