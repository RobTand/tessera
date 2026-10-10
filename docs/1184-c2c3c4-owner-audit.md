# k2 audit: owners and fix status of C2, C3, C4 (tessera#1193, parent #1184)

Post this as a comment on tessera#1184. Master lines were read on master (`4c0fd6a`). Unlanded C2 lines are cited from tessera#1086 and the independent review, not read here.

## C2 (retained-trace evidence): no owner on master; RobTand/tessera must change

- A repo-wide search for `retained_trace`, `trace_evidence`, `recipe_verify_owner`
  and `retained_trace_files_rehashed` returns no line on master (`4c0fd6a`).
- `experiments/graph_attest_702/tp2_recipe.py:230` (`def pair_arm`) through
  `:272` (`def plan`) validates arm and plan fields only. It names no trace list.
- `experiments/graph_attest_702/eager_benchmark.py:489`
  (`def profile_and_power`) executes fresh profiles for each declared cell.
  It checks no retained evidence.
- The validator lives in unlanded tessera source, not outside tessera. The
  tessera#1086 body cites tessera paths `experiments/graph_attest_702/tp2_recipe.py:248-286`
  and `eager_benchmark.py:337-347`. It names failing source `df41625d33ec`,
  corrected head `e3292489426f`, refactor `f1d2e6d69ea9`, base `62f5093ffa`,
  and a bundle with SHA256 `44f244085caa...e136e2a598866` ("reuse, do not duplicate").
  That commit is absent from origin from this seat, so its lines are cited
  from the issue, not read here. The independent review read `df41625d33ec`
  in the celestia clone and found `recipe_verify_owner_profile_evidence` at
  `tp2_recipe.py:201`, `retained_trace_files_rehashed` at `:285`, and tests at
  `tests/test_graph_attest_mnbt_matrix.py:296-326`.
- Verdict: no C2 owner exists on master. RobTand/tessera must change: k3 lands
  the retained corrected source. #1086 closed as not reproducible on master,
  and that closure landed no fix.

## C3 (intake binding): related, but it owns serve-comparison export only

- `tools/comparison_input_intake.py:69` (`def intake`) binds the comparison
  manifest to the exact source commit (`:77`). It requires an explicit reason
  for a different export source or contract (`:86`). It refuses audit-roster
  drift before publication (`:93`).
- `tools/comparison_arm_identity.py:91` (`def export_identity`), `:139`
  (`def read_comparison_arms`) and `:173` (`def comparison_artifact`) form the
  shared owner. The gate refuses drift at `:194` and `:199`.
- The gap stays open: this intake binds a serve-comparison export only. It
  binds no finding, correction, review or receipt. It names no title, body
  or rebase. `ARM_DIGEST_RECEIPT` (`tools/comparison_arm_identity.py:36`,
  `:235`) digests comparison arms only; it is not a C3 receipt binding.
- Current behavior stays pinned by `tests/test_comparison_intake.py:73`,
  `:104`, `:123`, `:157`, `:180`, `:222` and `:235`. All 28 tests pass
  (PrismaBuild action `645222f55d8b83107db8482a1aff312f2d0340e35ed5c4ab0c85b979a2c3f63e`).
- Verdict: `tools/comparison_input_intake.py` is related to C3 only, as
  suspected. It is not the C3 owner as the parent states it.

## C4 (conflict drops): no owner exists in this repository

- A search for conflict-merge and branch-only terms in `tools`, `src` and
  `tests` returns no line on master.
- `disposition` names only the MoE weight field
  (`src/tessera/export_serving.py:3949`, `src/tessera/serving_parts.py:1179`)
  and triage prose. No merge-disposition logic exists.
- `tools/merge_suite.py` merges suite populations on the merge result. Its
  functions (`_binding_refusal` at `:697`, `_arm_results` at `:1211`,
  `_verdict` at `:1245`) join published populations and pool records. They
  do not decide a merge disposition (`_git` at `:219-267` reads HEAD and
  status for receipt provenance only).

## Fix status of #1069, #1082 and #1086

- #1069 fixed C1 only, through PR #1072 (merge `83a1f38c4965`). The tool is
  `tools/selected_population.py`, and it reuses `merge_suite.py` as receipt
  owner. Proof sits in `tests/test_selected_population.py:175` (a nonzero
  pbrun code is not hidden), `:184` (a pool-failed shard is red), `:190`
  (no pool record is not green), `:316` (interpreter binding), `:338`
  (eight-client cap) and `:372` (resume survival). It fixes none of C2, C3, C4.
- #1082 (PR #994) measured E2M1 geometry. It touches no trace, intake or
  merge code. It fixes none of C2, C3, C4.
- #1086 closed as not reproducible with no code change. Its C2 defect
  (empty retained-trace lists pass after zero hashes) remains open.

## Open clauses: repository that must change

- C2: RobTand/tessera must change. Land the retained corrected source that
  #1086 names (head `e3292489426f`, bundle above; "reuse, do not duplicate")
  into `experiments/graph_attest_702/tp2_recipe.py`,
  `experiments/graph_attest_702/eager_benchmark.py` and
  `tests/test_graph_attest_mnbt_matrix.py`. The test that must fail first is
  the #1086 boundary population: an empty trace list is refused, and
  duplicate, mismatched or out-of-root evidence is refused by name, while
  shared files pass (failed-before action `eb6e0db1469e`, 12 failed;
  corrected action `871152de4bdb`, 41 passed).
- C3: RobTand/tessera must change. Extend `tools/comparison_input_intake.py`
  with `tools/comparison_arm_identity.py`, or add a sibling intake module.
  It binds every finding, correction, review and receipt to the exact
  source. It refuses a stale title, body or receipt after a rebase.
- C4: RobTand/tessera must change. No owner exists in RobTand/tessera
  (searched `tools`, `src`, `tests` and `experiments`: no merge-disposition
  code). A read-only search of the `/home/rob/prismabuild` and
  `/home/rob/prismaquant` checkouts also finds no gate (only task logs and
  scheduler prose). This seat cannot read the private fleetgraph and
  prisma-exec repositories, so it names no file there and guesses none.
  The parent clause and its owning issue #1196 are in RobTand/tessera, so
  k5 adds the gate there, in a new module. No issue in another repository
  is needed. The test that must fail first: a dropped branch-only test
  without a disposition is refused, and a removal with a recorded
  disposition passes.

## Owning issues (all in RobTand/tessera, each carries a `[P2]` title)

- C2: tessera#1194 `[P2] Retained-trace evidence: refuse empty, duplicate,
  mismatched or out-of-root traces; shared files pass`. Test that must fail
  first: an empty retained-trace list passes after zero hashes.
- C3: tessera#1195 `[P2] Intake binding: refuse a stale finding, correction,
  review, title, body or receipt after a rebase`. Test that must fail first:
  a record bound to another source, or a stale title, body or receipt after
  a source change, still qualifies.
- C4: tessera#1196 `[P2] Conflict merge: refuse a dropped branch-only
  contract test without a recorded disposition`. Test that must fail first:
  the dropped branch-only test without a disposition is accepted.

## Decomposer scope for k3, k4 and k5, and file sharing

- k3 (C2, tessera#1194): scope is landing the retained corrected source above. Its files
  are `experiments/graph_attest_702/tp2_recipe.py`,
  `experiments/graph_attest_702/eager_benchmark.py` and
  `tests/test_graph_attest_mnbt_matrix.py`.
- k4 (C3, tessera#1195): scope is the intake binding above. Its files are
  `tools/comparison_input_intake.py`, `tools/comparison_arm_identity.py`
  and `tests/test_comparison_intake.py`, with `docs/ARCHITECTURE.md` when
  a gate moves.
- k5 (C4, tessera#1196): scope is the conflict-merge refusal above. No
  owner file exists, so k5 adds a new module and its test in RobTand/tessera.
  The k5 author chooses the name. k5 must not edit `tools/merge_suite.py`.
- Sharing: k3, k4 and k5 share no file with each other. k3 files are now
  known. Only k4 extends existing gate files; k3 lands retained source into
  existing experiment files; k5 adds a new file.
