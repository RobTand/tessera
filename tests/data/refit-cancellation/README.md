# Corrected CHANNEL refit reproduction

This is the #360 corrected encoder output for the corresponding historical
`tests/data/legacy/` input. The historical blob and reader hash are unchanged.
Only row 12 of DIAG_SV and its dependent digests differ; alphabet/body bytes match.

ARM action `0b71d393f1800fc0a3ae6f14871a09b230be41bd2bb06aaa5492c60583400212`
and x86 action `49e06e4cbb2390685c6b81625921a466b23cd3428560b37f26699124040c074e`
independently produced SHA-256
`b8e8732b35f14715426ae5af87ec7b20af445149117181c4252b153f30f83c8d`.

As in the original legacy-layout comparison, this test artifact deliberately
uses the historical explicit fixture stamp to isolate payload/writer behavior.
It is not a shipping artifact or an attestation of the corrected encoder identity.
See `docs/measurements/channel-refit-cancellation-2026-09-06.md`.
