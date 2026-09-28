"""A test producer's reuse authority: the protocol exercised with neutral schemas.

Tessera defines ``tessera.cached_unit.ReuseAuthority`` and names no client
record (tessera#599).  These tests need a producer on the other side of it, so
this file is one: its documents carry ``fixture.*`` schemas, and its checks
have the shape a real producer's do (an adoption's identities, a proof's pins,
a policy's executed groups).  A real producer ships its own file; PrismaQuant's
is ``prismaquant/tessera_reuse_authority.py`` and carries its schema tests.

The exporter loads this file by path (``--producer-authority``), so it imports
only the standard library.
"""
from pathlib import Path

CATALOG_EXTENSION_SCHEMA = "fixture.catalog_extension.v1"
CANDIDATE_OVERLAY_SCHEMA = "fixture.candidate_overlay.v1"
ADOPTION_SCHEMA = "fixture.source_adoption.v1"
PROOF_SCHEMA = "fixture.encoder_source_proof.v1"
SERVED_POLICY_SCHEMA = "fixture.served_activation_policy.v1"
CANONICAL_CAPTURE = ("fixture.calibration_cache.v1", "fixture_prefix_f32_v1")
PATH = Path(__file__).resolve()

_DOCUMENT_SCHEMAS = {"catalog_extension": CATALOG_EXTENSION_SCHEMA,
                     "candidate_overlay": CANDIDATE_OVERLAY_SCHEMA}


class FixtureReuseAuthority:
    canonical_hessian_capture = CANONICAL_CAPTURE

    def check_document(self, role, document):
        if not isinstance(document, dict) or document.get("schema") != _DOCUMENT_SCHEMAS[role]:
            raise ValueError("rooted cached unit authority schema differs")

    def adoption_proof(self, unit, adoption, identity, original):
        if not isinstance(adoption, dict) or adoption.get("schema") != ADOPTION_SCHEMA:
            raise ValueError("cached unit source adoption schema differs")
        candidate, reference = adoption["candidate_encoding_identity"], adoption["reference_encoding_identity"]
        if (candidate != identity or reference.get("unit") != unit
                or adoption.get("reference_pair", [None])[0] != unit
                or reference.get("encoder_source_sha256") != original):
            raise ValueError("cached unit source adoption identities differ")
        for field in ("unit", "source", "calibration", "encoder_fixture_id", "projection"):
            if reference.get(field) != candidate.get(field):
                raise ValueError("cached unit source adoption changed " + field)
        return adoption["encoder_source_proof"]

    def proof_authorizes(self, proof, adoption, original):
        candidate = adoption["candidate_encoding_identity"]
        pins = proof.get("pins", {}) if isinstance(proof, dict) else {}
        return (isinstance(proof, dict) and proof.get("schema") == PROOF_SCHEMA
                and proof.get("ok") is True and proof.get("encoder_fixture_id_equal") is True
                and pins.get("old", {}).get("encoder_source_sha256") == original
                and pins.get("new", {}).get("encoder_source_sha256") == candidate["encoder_source_sha256"]
                and set((proof.get("fixture_id", {}).get("ids") or {}).values())
                == {candidate["encoder_fixture_id"]})

    def served_activations(self, policy, adoptions, units):
        rates = policy.get("rates") if isinstance(policy, dict) else None
        if (rates is None or policy.get("schema") != SERVED_POLICY_SCHEMA
                or not isinstance(rates, list) or not rates
                or rates != sorted(set(rates))):
            raise ValueError("rooted cached unit served activation policy schema differs")
        index = {name: (key, group) for key, group in policy["executed_grouping"]["groups"].items()
                 for name in group["members"]}
        expected = {}
        for name in adoptions:
            recipe = units[name]["identity"].get("recipe", {})
            if recipe.get("grid") == "E2M1x2" and recipe.get("q256") in rates:
                if name not in index:
                    raise ValueError("selected A4 unit absent from served activation policy")
                key, group = index[name]
                expected[name] = {"group": key, "input_global_scale": group["input_global_scale"]}
        return expected


PRODUCER_AUTHORITY = AUTHORITY = FixtureReuseAuthority()

