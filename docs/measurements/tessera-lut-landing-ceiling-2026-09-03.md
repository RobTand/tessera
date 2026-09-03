# The LUT landing is exact on the served default, and on the full-H refit its prize is the assignment, not the table (2026-09-03)

**Claim, weight space (measured; a screen and not a result).** Issue `#50` read
`#35`'s `refit_diagnostics()` instrumentation and found the LUT plane's
*landing* -- `_fit_lut`'s sixteen-entry fit plus nearest-in-linear assignment --
taking back 24-91% of whatever step the refit hands it, several times more than
the Jacobi-to-Gauss-Seidel step fix was worth (3.73%). It asked for the ceiling
before any optimiser: run the refit with the landing disabled and read the
six-unit geomean. Measured, on the same six dense Qwen3-0.6B units, the same
E2M1x2 `q896` cap wire and the same LDLQ 1.0/32:

PLACEHOLDER_HEADLINE

## What this does not claim

PLACEHOLDER_LIMITS
