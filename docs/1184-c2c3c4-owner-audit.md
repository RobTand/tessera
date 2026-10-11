# k2 audit: owners and fix status of C2, C3, C4 (tessera#1193, parent #1184)

This is the audit published on tessera#1184. Current owner lines were read on master `a891e8faeaea8a9da6f56b05b6ab381dadc18364`.
The historical #1086 MNBT profile evidence remains separate from the current serve-trace scope.
Parent comments 6099914001 and 6099914324 define the revised k3 and k5 scopes.

## C2 (retained serve-trace evidence): partial owner on master

- `src/tessera/serving/contract.py:797` owns `_validate_world_size_receipt`.
  Lines 843-847 require one route trace per rank and therefore refuse an empty list.
  Lines 849-853 refuse a repeated path and name the trace.
- `_require_data_path` at `contract.py:707-719` checks the data-root grammar.
  Its docstring states that grammar is not a check of file existence or content.
  The wheel contains no experiment files. These lines do not rehash trace bytes
  or prove that a resolved path stays inside the declared root.
- The receipt-wide set at `contract.py:819` also rejects a path shared
  between different serves. This conflicts with the parent shared-file criterion.
  Within-serve duplicate refusal must remain, while cross-serve sharing must pass.
- Parent comment 6099914001 and the revised k3 scope identify this contract owner.
  Remaining gaps are byte mismatch, resolved-root proof and shared evidence.
- The MNBT profile validator cited by #1086 is a separate experiment path.
  Its retained source was not landed by that issue closure. This audit does
  not send k3 to `experiments/graph_attest_702` or claim that no C2 owner exists.
- Verdict: RobTand/tessera must extend its current serve-trace boundary through #1194.

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
- #1086 closed as not reproducible with no code change. Its retained MNBT
  profile-validator gap is separate from the current serve-trace C2 owner.

## Open clauses: repository that must change

- C2: RobTand/tessera must change its current serve-trace boundary in
  `src/tessera/serving/contract.py`, through tessera#1194.
  Preserve the existing empty-list and within-serve duplicate refusals.
  Add the missing byte-mismatch and resolved-root checks. Permit a file
  shared by two different serves, but refuse a repeated path within one serve.
  The first regression must show that the current receipt-wide set wrongly
  refuses shared evidence. Record an actual pre-fix failure during implementation.
  The retained MNBT profile validator is a separate evidence path, not k3.
- C3: RobTand/tessera must change. Extend `tools/comparison_input_intake.py`
  with `tools/comparison_arm_identity.py`, or add a sibling intake module.
  It binds every finding, correction, review and receipt to the exact
  source. It refuses a stale title, body or receipt after a rebase.
- C4: no implementation repository is selected. No verified Tessera owner
  exists for a silent conflict drop. The location affects the merge lane,
  so Rob decides it through tessera#1196. The issue location does not
  authorize a new gate in Tessera. Record the repository, owner file,
  pre-fix test and follow-up owner before an implementation child starts.
  If the decision selects another repository, file its owning issue there
  with a priority prefix and link it from #1196.

## Owning issues (all in RobTand/tessera, each carries a `[P2]` title)

- C2: tessera#1194 `[P2] Retained-trace evidence: refuse empty, duplicate,
  mismatched or out-of-root traces; shared files pass`. A proposed first
  regression is `tests/test_serving_contract.py::test_world_size_receipt_shared_trace_between_serves`.
  It must reject the current cross-serve duplicate rule. No pre-fix run is claimed here.
- C3: tessera#1195 `[P2] Intake binding: refuse a stale finding, correction,
  review, title, body or receipt after a rebase`. The first regression must
  show that a stale source-bound record still qualifies. Its implementation
  records the failing test before the fix.
- C4: tessera#1196 `[P2] Conflict merge: refuse a dropped branch-only
  contract test without a recorded disposition`. This issue records the owner
  decision. The decision must name the regression path before implementation.
  The case is a dropped branch-only test that has no recorded disposition.

## Decomposer scope for k3, k4 and k5, and shared files

- k3 (C2, tessera#1194) extends the serve-trace owner in
  `src/tessera/serving/contract.py`, with `tests/test_serving_contract.py`
  and the world-size receipt section of `docs/ARCHITECTURE.md`.
  It does not land the separate retained MNBT experiment validator.
- k4 (C3, tessera#1195) extends the intake boundary in
  `tools/comparison_input_intake.py` and `tools/comparison_arm_identity.py`,
  with `tests/test_comparison_intake.py` and the intake section of
  `docs/ARCHITECTURE.md`. Its source-binding gap remains open.
- k5 (C4, tessera#1196) records a person decision only. Rob must name
  the repository, owner file and pre-fix test before an implementation child starts.
  k5 changes no code and adds no receipt owner.
- k3 and k4 share `docs/ARCHITECTURE.md`, but they change separate sections.
  Their code and test files are distinct. Coordinate edits to that document.
  k5 has no file scope until the owner decision.
