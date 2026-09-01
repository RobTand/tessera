# The trellis minimised the wrong error — scale-weighted branch metric (2026-09-01)

**Claim.** `viterbi_columns` runs on `work / scale`: targets normalised per
half. Unweighted, the path minimises `Σ (w/c − q)²` — each position's error
divided by its own scale squared — so a column's quiet groups are served at
the loud groups' expense. A half is sixteen consecutive columns of one row,
so every position of a trellis column carries its own scale, and the per-16
plane varies by up to ~3.5 octaves (128× in weight). Weighting each
position's branch metric by `c²` makes the path minimise the true
`Σ (w − c q)²`, the objective the plane refit already descends. No wire
change; the point choice per position is unchanged (a positive scaling);
only the path — and the fold, at span 2 — sees the weight.

**Measured** (`experiments/tessera_trellis_weighting_check.py`, six GLM
experts, held-out 1024 rows, served activation quantiser; identity checks
reproduce the wire-default numbers to six digits):

| arm (4.0 bpp) | out-space weight leg | weighted / unweighted | W4A4 vs EXL3@A4 | encode s |
|---|---:|---:|---:|---:|
| span-2 LUT refit-4 unweighted (default) | 0.08044 | 1.000× | 1.137× | 2.1 |
| **span-2 LUT refit-4 scale-weighted** | **0.07983** | **1.0077× (1.005–1.010)** | **1.133×** | 2.2 |
| span-2 LUT refit-0 unweighted | 0.08528 | | 1.169× | 0.6 |
| span-2 LUT refit-0 scale-weighted | 0.08465 | 1.0077× (1.005–1.010) | 1.164× | 0.7 |
| span-1 S6b refit-0 unweighted (the export) | 0.09819 | | 1.258× | 0.4 |
| span-1 S6b refit-0 scale-weighted | 0.09684 | 1.0142× (1.012–1.020) | 1.248× | 0.4 |

Weight-space error falls with it (0.08834 → 0.08774 at the default), as it
must: the weighted Viterbi is exact for that objective at the cap rate
(`tests/test_span_and_lut.py::test_scale_weighted_trellis_never_ends_a_column_worse`
holds every column to it).

**Why so small.** Six of the seven bits per code are the point, chosen per
position and weight-invariant; the weight only moves the one select bit per
code (half a bit per pair at span 2). The refit already shrinks the plane's
spread, which is why the gain is halved from the amax plane (1.4%) to the
refit plane (0.8%).

**A first run was mis-specified** and is void
(`tessera_trellis_weighting_check_void-per-code-weight.*`): it weighted
each *code* by row `2s`'s scale, but a code's two rows sit in different
halves. It read 0.9995×. The per-row form is the one above.

**Status.** Exporter default (`export.DEFAULT_TRELLIS_WEIGHTING = "scale"`),
recorded as `trellis.weighting` in the checkpoint config and compared by the
merge guard's SHARED set; PrismaQuant's render leg passes it too.
`encode_unit`'s own default stays `"none"` so pre-existing artifacts remain
reproducible from source. Evidence class: a per-tensor screen; promotion
rides the same served A/B as the wire.
