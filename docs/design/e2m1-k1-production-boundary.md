# E2M1 K1 production boundary

Decision recorded 2026-09-29 for tessera#477.

## Decision

Arity-one E2M1 remains a research encoding, not a production serving family.
Keep the existing production refusal. Do not publish a reader range, add a
family-to-route mapping, or relax family injectivity without a measured load
and byte-exactness receipt.

This selects the refusal alternative in #477. It does not retire the research
format or assert that its kernel cannot work. A future qualification is a new
measured change, not an implicit consequence of the encoder accepting a grid.

## Where production refuses

- `src/tessera/export_serving.py:check_recipe` calls
  `serving.scheme.refuse_unserveable_wire` before the encode loop. Every plan
  override passes through this check. With no reader range for the route/grid
  pair, the normal production path names the target and refuses before encoding.
- `--allow-unserveable` is an explicit research override. It records the refusal
  in the manifest's `serving_gate` block; it establishes neither loadability nor
  qualification. A research wire is not a production admission.
- A production cost menu consumes the published serving contract, not the
  research encoder's roster. The absent K1 family has neither a reader range
  nor an attested cell to admit. Research pricing may retain that candidate,
  but it must not treat its price as production serving qualification.

The generic `encode_linear` and `wire_recipe` APIs deliberately remain wider
than the serving boundary. Refusing a research encode there would prevent the
measurement needed to qualify a new reader, without making production safer.

## Evidence and limits

The packaged contract has no E2M1 K1 reader range. The export gate reads that
contract instead of inventing a bound. Contract-driven production selection
and the export gate already fail closed; this decision documents their scope rather than
changing bytes, rates, recipes, kernels, numerics, defaults, or the contract.

No K1 device load, byte-exactness, served quality, or performance measurement is
claimed. Those receipts remain prerequisites for any later promotion.
