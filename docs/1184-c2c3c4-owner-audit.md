# k2 audit: owners and fix status of C2, C3, C4 (tessera#1193, parent #1184)

Post this as a comment on tessera#1184. All lines were read on master (`4c0fd6a`).

## C2 (retained-trace evidence): no owner exists in this repository

- A repo-wide search for `retained_trace`, `trace_evidence`, `recipe_verify_owner`
  and `retained_trace_files_rehashed` returns no line on master.
- `experiments/graph_attest_702/tp2_recipe.py:230` (`def pair_arm`) through
  `:272` (`def plan`) validates arm and plan fields only. It names no trace list.
- `experiments/graph_attest_702/eager_benchmark.py:489`
  (`def profile_and_power`) executes fresh profiles for each declared cell.
  It checks no retained evidence.
- This matches the closing comment of tessera#1086.

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
  or rebase. A search for those terms in both files returns only the
  issue-#885 origin note and the exception-reason help text.
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
  never touch git branches.

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

- C2: no tessera module can change yet. The validator, when it exists, lives
  outside tessera: #1086 cites a kernels graph intake and a private parent
  bundle. This agent has no access to those fleet repositories, so it names
  no file there and guesses none. The owning issue needs a `[P]` prefix
  title in the validator repo. It states the test that must fail first:
  an empty trace list is refused, and duplicate, mismatched or out-of-root
  evidence is refused by name, while shared files pass.
- C3: RobTand/tessera must change. Extend `tools/comparison_input_intake.py`
  with `tools/comparison_arm_identity.py`, or add a sibling intake module.
  It binds every finding, correction, review and receipt to the exact
  source. It refuses a stale title, body or receipt after a rebase.
- C4: no tessera module can change yet. No conflict-merge code exists here.
  The owning issue needs a `[P]` prefix title where the merge gate lives,
  or a tessera issue for a new tool. It states the test that must fail
  first: a dropped branch-only test without a disposition is refused, and
  a removal with a recorded disposition passes.

## Decomposer scope for k3, k4 and k5, and file sharing

- k3 (C2): scope is the retained-trace refusal above. Its file is unknown:
  a new file or a change outside tessera.
- k4 (C3): scope is the intake binding above. Its files are
  `tools/comparison_input_intake.py`, `tools/comparison_arm_identity.py`
  and `tests/test_comparison_intake.py`, with `docs/ARCHITECTURE.md` when
  a gate moves.
- k5 (C4): scope is the conflict-merge refusal above. Its file is unknown:
  a new file or a change outside tessera.
- Sharing: k3, k4 and k5 share no file with each other. Only k4 touches
  existing files. k3 and k5 each need a file this audit could not name.
