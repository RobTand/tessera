# LFM dense passthrough names the constructed module

The real full-model census action
`270b382c9fe6b23c9ef20fa7a16ae1d5dc483f4d3c77581e0184ec863569610d`
reached construction after the shard-readability fix, then failed at
`serving/config.py:382`: `tessera checkpoint declares no wire for Linear
'model.layers.0.feed_forward.w13' and does not ignore it`.
The exact pinned LFM construction receipt already names this offered module.
The producer's fusion rule was missing its source pair `w1/w3`, so the
passthrough declaration named the unmerged leaves instead. The plugin's
closed-world refusal is retained.

The new parameterized test failed for both source roles before the fix at
`tests/test_export_moe_layouts.py:407`: `assert None == (...feed_forward.w13, ...)`.
PB action `edd7d475acc4` retained two failures, zero skips and zero uncollected
modules, serial CPU, torch 2.11.0+cpu, no CUDA device. The same shared rule now
drives dense targets and passthroughs while routed expert leaves remain
excluded from dense fusion.

PB action `8ab8a1f62fa1035406606f5bb555af9eee33becd52437ded508c5893e5543341`
returned zero: 38 passed, 3 skipped, zero uncollected modules in 44.46 seconds
across export MoE layouts, ignore completeness and construction gates.
Population: serial CPU, torch 2.11.0+cpu, no CUDA device. All three skip reasons
were exactly `the encoder is a GPU job`. Receipt CAS:
`c6a596c911131aabbc2b3ad0bb5aa4ce43ad80f3d5c23031fdd458ee513322ba`.
This test count does not cover the CUDA encoder surface or promote a MoE cell.

The bounded metadata correction additionally preserves unrelated declarations
such as tied `lm_head`, even though they need not have a distinct stored tensor.
Actual correction attempt `048b1221c8d8` refused the unwanted head removal before
creating a new model. A new helper test failed in action `22ebe9be737a` at
`test_ts5_passthrough_correction.py:13`; the two existing refusal conditions
already held. Final action
`42987b1e225203d6353c916e72743278dd3d6a27d9e00f65e43aecf993305ece`
passed all three tests, CPU/no CUDA, zero skips and zero uncollected modules.
Receipt CAS: `0a400a46771df47cff1dac8ed3206c52485250f1c1b63d48673b8ce1275cade7`.
The successful actual correction and its exact config/weight identities are
recorded in `tessera-lfm-campaign-2026-09-04.md` §6.
