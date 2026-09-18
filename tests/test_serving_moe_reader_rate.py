"""The routed reader's rate range is the trellis domain, not one rung (#506 leg 2).

The NVFP4 expert route decodes E2M1x2 trellis wires -- span-2 TCQ bodies over
the LUT plane -- and the packaged contract publishes the rungs its reader has
been MEASURED to decode.  Today that is one rung, q256 896, the coset trellis's
cap: every sub-cap rung of the grid exports as a WINDOW body
(``export.wire_recipe``), which this route does not serve, so the single
published rung was not a policy choice -- it was the whole readable set at the
time the range was measured.  #506 leg 2 widens it: the span-2 select pad the
row-shard work added (#492) lets the compact reader start a whole unit at any
in-domain rung, so the measured range becomes the trellis-shaped domain per
expert instead of its top rung.

What "the trellis domain" IS, derived rather than chosen: a code position's
rate is ``q256 * arity / 256``, and the grammar admits ``1 <= rate <=
rate_cap`` (``payload_bits - 1`` = 7 on E2M1x2), so q256 spans [128, 896].
The compact reader takes ONE forest per unit (``prepare_span2_compact`` refuses
mixed ``metadata.rates`` -- "the span-2 planes take one forest per unit"), so
the rungs it can read are the integral-root ones, ``q256 % 128 == 0`` on arity
2 -- exactly the seven below.  A Bresenham rung between them (700 is root
5 15/32, rates 5 and 6 mixed in one unit) is inside the grammar but not
readable by this reader; it is refused with the reason recorded, not padded
over.

GREEN SINCE contract v32: the measured range publishes [128, 896] step 128
and all seven rungs load; the receipts behind them are the seven
single-rank GPU proofs named in the changelog (PB action
a186d7bc6f1f111d856ee734d7c3359eb957d116ec82ba92a087961d748c2092).  The
refusals below the domain, above the cap, and off the step stay refused.
"""
from __future__ import annotations

import pytest

from tessera.serving import scheme as S

#: THE E2M1x2 TRELLIS DOMAIN, in the rungs the compact reader can decode:
#: ``root = q256 * arity / 256`` must be a whole per-code rate (one forest per
#: unit) between 1 and ``rate_cap`` = 7, so on arity 2 the rungs are
#: ``q256 % 128 == 0`` with ``128 <= q256 <= 896``.  A literal, so the red run
#: names every refused rung rather than a range object.
TRELLIS_DOMAIN_RUNGS = (128, 256, 384, 512, 640, 768, 896)

#: Inside the grammar's [128, 896] but NOT one forest per unit: 700 is root
#: 5 15/32, a Bresenham schedule mixing rates 5 and 6 in one unit, which
#: ``prepare_span2_compact`` refuses by name.  The published range must
#: exclude it -- a step that swallowed it would claim a readability the
#: reader's own preparer denies.
OFF_STEP_RUNG = 700


def _nvfp4_group(rows, columns, roles, q256, stride=4096, **over):
    g = {"rows": rows, "columns": columns, "roles": roles, "q256": q256,
         "wire_stride": stride}
    g.update(over)
    return g


def _nvfp4_moe(q256, experts=4, hidden=128, inter=64, **over):
    """A routed E2M1x2 expert stack at ``q256`` on the NVFP4 route's own tile
    (span-2 TCQ bodies over the LUT plane), with clean geometry -- w13 is
    [2N, K] and w2 is [K, N] over one expert, roles stack, halves equal -- so
    the reader's rung gate is the only fact that can refuse it."""
    s = {
        "family": S.TESSERA_NVFP4, "structure": S.STRUCTURE_ROUTED_MOE,
        "grid": "E2M1x2", "body": "TCQ", "plane": "LUT",
        "experts": experts,
        "groups": {
            "w13": _nvfp4_group(2 * inter, hidden,
                                [["gate_proj", inter], ["up_proj", inter]], q256),
            "w2": _nvfp4_group(hidden, inter, [["down_proj", hidden]], q256),
        },
    }
    s.update(over)
    return s


@pytest.mark.parametrize("q256", TRELLIS_DOMAIN_RUNGS)
def test_every_rung_of_the_trellis_domain_loads(q256):
    """RED UNTIL the measured range publishes: each in-domain rung normalises
    through the scheme gate the loader runs per expert stack, and today the
    six below the cap are refused with the published [896, 896] span."""
    norm = S.validate_tessera_scheme(_nvfp4_moe(q256), "m")
    assert norm["structure"] == S.STRUCTURE_ROUTED_MOE and norm["experts"] == 4
    w13, w2 = norm["groups"]["w13"], norm["groups"]["w2"]
    assert w13["q256"] == q256 and w13["role_q256"] == [q256, q256]
    assert w2["q256"] == q256 and w2["role_q256"] == [q256]


@pytest.mark.parametrize("q256", [64, OFF_STEP_RUNG, 1024])
def test_a_rung_outside_the_readable_domain_is_refused_by_name(q256):
    """64 is below the grammar (rate 1/2), 1024 is above the trellis cap
    (rate 8, the grid's whole payload width), and 700 is inside the grammar
    but not one forest per unit.  The scheme gate refuses all three naming
    the route, before AND after the range widens: the span in the message is
    the published one, and these rungs are outside it at every width."""
    with pytest.raises(ValueError,
                       match="outside the rungs this build's decoder reads for TESSERA_E2M1_K2"):
        S.validate_tessera_scheme(_nvfp4_moe(q256), "m")
