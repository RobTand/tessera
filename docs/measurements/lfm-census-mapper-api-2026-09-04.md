# The census replays the runtime's name-only mapper

Full LFM census attempt three, action
`6c68e2a98472ede9771b196908de83880738ad732f4794d4fffa13f25e81c856`,
loaded and prepared both model shards, then failed in the observer at
`tools/tessera_route_census.py:151`: the pinned EUGR `WeightsMapper` exposes
`get_rename_mapper`, not `get_unstacked_mapper`. It produced no census result.
Its cleanup receipt records the exact container absent, no GPU compute
processes and `safe_to_release=true`. The corrected artifact and its seal were
not changed for the next attempt.

CPU-only inspection inside the exact pinned image established both the class
and the actual loader call. Action
`b32f1edc6f15839065b8869e2d3d67c104aadd7aea672797eace91dfa7db0b82`
printed `WeightsMapper`; action
`98fcb545536a354641d34ca3dc04ac5360abfc0495890a1795b1416b3daa6e11`
printed `configure_quant_config`, which calls `get_rename_mapper`. The newer
name-only view removes stacked rewrites and weight-drop rules; replaying the
raw weight mapper when this view exists would not match the quant config.
Neither inspection mounted a GPU. Their receipt CAS values are respectively
`c47088ff572b763b01031ded43eb113e21445f9d929b22211f589487510660c7`
and `1b0f4f915fa9d734b481a6431c15d6bece160b3b22567ada95aacd03fb72de1d`.

Both censuses now use `serving.weights_mapper.module_name_mapper`: current
rename view, otherwise the earlier unstacked view, otherwise a plain mapper.
An error from an existing view method propagates rather than silently
substituting raw weight semantics. This is observer compatibility, not a model
or encoder change.

Red-first evidence, both CPU/no CUDA, zero skips and uncollected modules:

- `807b267011b6`: one failure at route-census line 151 on a directly exposed
  mapper lacking the old wrapper method.
- `27c05e872cfa`: one failure at test line 88 because raw mapping returned
  `model.dead: None` and `raw.live` instead of the runtime's name-only view.

Final qualification action
`d85232bbea0df95d626ae2c8712b0ec4b30295722fdfc0c4ca916fafa413e479`
returned zero: 101 passed in 1.96 seconds, serial CPU, torch 2.11.0+cpu,
no CUDA device, zero skipped and zero uncollected modules. It includes the
whole route-module-space, construction-contract and campaign-collector files.
Receipt CAS:
`980a3e8bbcaac85a73b4bb1c1f08ea4e7d79d9f2ddc8ff2412d1cd094daa7275`.
An intermediate incomplete patch snapshot failed before the corrected green
snapshot; it supplies no campaign evidence.

Selector action `11ce2e94b2bdf1e1ad61ef135727db1d67ee073532d19c45b8e16a4769982cb3`
narrowed the cumulative branch diff to five files. They are covered by this
run and the existing exporter/model-seal/passthrough-correction receipts.
It omitted the dynamically imported construction test, which was explicitly
run above; that known selector variant was sent to the existing #129 owner.
No full suite or positive route cell is claimed by this receipt.
