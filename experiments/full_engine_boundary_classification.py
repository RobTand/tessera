"""Compare two full-engine captures taken under two per-Linear assignments (tessera#548).

The census calls a persistent runner root, a BLAS workspace and a native
apply boundary tensor ``shared``, and the #399 derivation leaves every one of
them ``pending_548``: whether those bytes move with the selected assignment is
a property of a **set** of assignments, and one capture observes one. Writing
``fixed`` or ``candidate`` from a single capture states a claim no observation
supports, and ``fixed`` is the direction that hands a serving gate a charge
that silently moves with the menu.

This module builds the only thing that can answer it: **the two captures'
matched site bytes, side by side**. It refuses a pair that changed more than
the assignment, it matches sites by the owner string the census emitted, and
it writes **no verdict**. The classification record carries the two capture
identities and the two byte figures per site, so the rule in
``full_engine_ownership`` and PrismaQuant's independent recomputation read the
same numbers and can disagree; a producer that carried its own answer would be
certifying itself, which is the failure the consumer exists to catch.

Agreement across two captures is evidence for **those two captures**, never a
universal invariance, which is why every record here names them and why the
rule's ``reason`` repeats both digests in the view it writes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.full_engine_ownership import _parse_boundary_owner, views_by_allocation

BOUNDARY_CLASSIFICATION_SCHEMA = "tessera.full_engine_boundary_classification.v1"

#: Only a v2 boundary ledger can be compared: v1 carries no ownership
#: observation, so nothing in it says which rows are still unclassified.
LEDGER_SCHEMA_V2 = "tessera.full_engine_raw_resource_ledger.v2"

#: What tessera#548 holds equal across the pair. ``configuration_sha256`` is
#: NOT among them: the configuration document names the artifact, so a
#: substitution necessarily changes its digest. What must not move is the
#: runtime, the canonical roster the units are named in, the served model and
#: the workload the bytes were observed under.
HELD_FIXED = ("model_sha256", "runtime_manifest_sha256", "canonical_units_sha256",
              "workload_sha256")

#: The one input the pair changes.
CHANGED_INPUT = "assignment_sha256"

#: The rule a pending row's owner string is read by; the same four kinds the
#: derivation names when it abstains.
SITE_KINDS = (("native:", "native_boundary_tensor"),
              ("runner:", "persistent_runtime_root"),
              ("torch.cublas:", "blas_workspace"))


def site_kind(owner):
    """Which kind of shared site this owner string names."""
    for prefix, kind in SITE_KINDS:
        if owner.startswith(prefix):
            return kind
    return "library_cache_buffer"


def site_unit(owner):
    """The unit a native boundary owner names, or ``None`` for every other kind."""
    parsed = _parse_boundary_owner(owner)
    return None if parsed is None else parsed[0]


def pending_sites(ledger):
    """``{owner: bytes}`` over the rows the derivation left ``pending_548``.

    A site's bytes are the sum of the distinct allocation rows its owner names
    in this capture. For a native boundary tensor that is one row, because the
    v2 row rule makes ``(unit_id, invocation, kind)`` unique; for a runner root
    that names several storages it is their sum, which is the figure that has
    to be equal across the pair for the site to be evidence of anything.
    """
    if ledger.get("schema") != LEDGER_SCHEMA_V2:
        raise ValueError("a two-capture classification reads a v2 boundary ledger on both "
                         "sides; this one is " + str(ledger.get("schema")))
    views = views_by_allocation(ledger)
    sites = {}
    for row in ledger["torch_allocations"]:
        view = views.get(row["allocation_id"])
        if view is None or view.get("rule") != "pending_548":
            continue
        for owner in row["observed_owners"]:
            sites[owner] = sites.get(owner, 0) + row["bytes"]
    return sites


def _identity(ledger):
    identity = ledger.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("a capture with no identity cannot be one half of a pair")
    return identity


def boundary_classification(first, second):
    """The two captures' matched site bytes, with no verdict.

    Refuses a pair that is not the measurement #548 asks for: both ledgers are
    v2, every held-fixed identity coordinate agrees, and the assignment digest
    differs. A pair that fails any of those cannot separate "these bytes do not
    depend on the assignment" from "nothing about the run changed".
    """
    identities = [_identity(first), _identity(second)]
    tables = [pending_sites(first), pending_sites(second)]
    digests = [first.get("capture_sha256"), second.get("capture_sha256")]
    if not all(digests) or digests[0] == digests[1]:
        raise ValueError("a pair is two distinct captures, each naming its own capture_sha256")
    differing = [name for name in HELD_FIXED if identities[0].get(name) != identities[1].get(name)]
    if differing:
        raise ValueError("the pair did not hold the runtime, roster, model and workload fixed; "
                         "these differ: " + ", ".join(differing))
    if identities[0].get(CHANGED_INPUT) == identities[1].get(CHANGED_INPUT):
        raise ValueError("both captures ran one assignment, so nothing was substituted; "
                         "#548 needs two")
    sites = []
    for owner in sorted(set(tables[0]) | set(tables[1])):
        present = [digest for digest, table in zip(digests, tables) if owner in table]
        sites.append({"owner": owner, "kind": site_kind(owner), "unit": site_unit(owner),
                      "bytes": {digest: table[owner] for digest, table in zip(digests, tables)
                                if owner in table},
                      "present_in": present})
    return {
        "schema": BOUNDARY_CLASSIFICATION_SCHEMA,
        "captures": [dict(identity, capture_sha256=digest,
                          pending_sites=len(table),
                          pending_bytes=sum(table.values()))
                     for identity, digest, table in zip(identities, digests, tables)],
        "changed_input": CHANGED_INPUT,
        "held_fixed": list(HELD_FIXED),
        "sites": sites,
        "sites_in_both": sum(1 for site in sites if len(site["present_in"]) == 2),
        "scope": ("the bytes each still-unclassified shared site carried in two captures that "
                  "differ in the selected assignment and nothing else; a comparison, not a "
                  "verdict, and evidence for these two captures rather than for every "
                  "assignment"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True,
                        help="ledger.json of the first capture (v2 boundary ledger)")
    parser.add_argument("--second", type=Path, required=True,
                        help="ledger.json of the substitution capture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = boundary_classification(json.loads(args.first.read_text()),
                                     json.loads(args.second.read_text()))
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    agreed = sum(1 for site in record["sites"]
                 if len(site["present_in"]) == 2 and len(set(site["bytes"].values())) == 1)
    print(json.dumps({"sites": len(record["sites"]), "sites_in_both": record["sites_in_both"],
                      "sites_agreeing": agreed, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
