"""A fused module's roles share a family, not a rate (#37).

THE DEFECT THIS PINS.  ``experiments/export_tessera_serving.py`` grouped a
vLLM-fused module on ``(grid, q256)`` equality and passed anything else through
at source precision, while its own header stated the weaker -- and correct --
rule that the roles must share one FAMILY.  PrismaQuant's group knapsack is on
by default and allocates exactly the group the exporter refused, so the default
allocator configuration could produce an allocation with no Tessera export at
all, and nothing crossed the two halves.

WHY THE HEADER WAS THE RIGHT ONE.  The decoders read each role from that role's
OWN manifest: ``fp8_route``/``bf16_route`` call ``prepare_window`` on each
role's ``body_bits``/``rates``/``window_bits``/``window_codes`` and concatenate,
and ``ops.prepare_tessera_module`` packs each role's own
``rate``/``arity``/``memory``/``half`` scalars and decodes into that role's row
slice.  What is genuinely a module fact is the family (vLLM builds one method
per module) and, on NVFP4, the shared global -- and the global is carried by an
exact binade shift, not by the rate.

THE LOAD-BEARING ASSERTION is the decode identity below: a three-role container
at three different rungs decodes ELEMENT FOR ELEMENT to what the three roles
decode as three one-member modules -- the shape the exporter writes for an
unfused Linear, i.e. the path the unrelaxed code took.  Same assertion on a real
Qwen3-0.6B q/k/v in ``experiments/fused_member_rung_identity.py``.

WHAT IS NOT CLAIMED HERE.  Nothing is served.  No ``lane_eligibility`` cell and
no ``attested_rungs_q256`` moved, and the contract says so in a value
(``fused_module.mixed_rung_receipt`` is ``False``).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tessera.alphabet import BF16_GRID, E4M3_GRID                       # noqa: E402
from tessera.control import grid_for_name                               # noqa: E402
from tessera.export import encode_linear_planes                         # noqa: E402
from tessera.fused import pack_fused                                    # noqa: E402
from tessera.serving.contract import load_serving_contract, validate_serving_contract  # noqa: E402
from tessera.serving.scheme import (                                    # noqa: E402
    FUSED_CONTAINER, FUSED_MODULE_FIELDS, FUSED_MODULE_SCHEMA, FUSED_Q256_SPELLING,
    TESSERA_BF16, TESSERA_FP8, parse_tessera_blob_for_scheme, validate_tessera_scheme)

ROOT = Path(__file__).resolve().parents[1]
CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")

#: Three roles, three rungs, none equal, all inside the E4M3 family's published
#: ``reader_rate_range_q256`` [256, 2048] and the BF16 family's [256, 4096].
GROUP = (("q_proj", 64, 900), ("k_proj", 32, 1024), ("v_proj", 32, 1200))
COLUMNS = 512


def _exporter():
    spec = importlib.util.spec_from_file_location(
        "export_tessera_serving", ROOT / "experiments" / "export_tessera_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheme(family, grid, q256, roles, columns=COLUMNS, wire_bytes=1):
    return {"family": family, "grid": grid, "body": "WINDOW", "plane": "CHANNEL",
            "q256": q256, "rows": sum(r for _, r in roles), "columns": columns,
            "wire_bytes": wire_bytes, "roles": [[n, r] for n, r in roles]}


# --- the contract publishes the rule -----------------------------------------

def test_the_contract_publishes_the_fused_rule_as_a_value_not_as_prose():
    """Principle 14 for the question a producer's group allocator asks.

    PrismaQuant's ``allocator_candidates`` says in a COMMENT that one rung per
    fused group "is not the serving constraint".  It is right, and until this
    block existed it had no way to know: the runtime published families, rungs,
    cells, rate ranges, world sizes and a quant method, and not one word a gate
    could read about what varies inside a fused module.
    """
    block = load_serving_contract()["fused_module"]
    assert block["schema"] == FUSED_MODULE_SCHEMA
    assert block["container"] == FUSED_CONTAINER
    assert block["fields"] == FUSED_MODULE_FIELDS
    assert block["sidecar_q256"] == FUSED_Q256_SPELLING
    assert block["fields"]["q256"] == "per_member" and block["fields"]["family"] == "shared"
    # The other claim, and it is not this one: no receipt covers a SERVED
    # mixed-rung module, exactly as ``max_world_size`` stays 1 while
    # ``loader_axes`` says both axes cut.
    assert block["mixed_rung_receipt"] is False


def test_the_published_block_cannot_drift_from_the_dict_the_loader_gates_on():
    contract = load_serving_contract()
    contract["fused_module"]["fields"]["q256"] = "shared"
    with pytest.raises(ValueError, match="FUSED_MODULE_FIELDS"):
        validate_serving_contract(contract)
    contract["fused_module"]["fields"]["q256"] = "per_member"
    contract["fused_module"]["sidecar_q256"] = "per_role_list_only"
    with pytest.raises(ValueError, match="sidecar_q256"):
        validate_serving_contract(contract)


# --- the sidecar -------------------------------------------------------------

def test_the_scheme_carries_one_rung_per_role_and_still_reads_the_old_spelling():
    roles = [(n, r) for n, r, _ in GROUP]
    rungs = [q for _, _, q in GROUP]

    mixed = validate_tessera_scheme(_scheme(TESSERA_FP8, "E4M3", rungs, roles), "t")
    assert mixed["role_q256"] == rungs and mixed["q256"] == rungs

    # Every checkpoint written before #37 carries a scalar, and normalises to
    # the scalar it always did -- one rung, repeated per role.
    uniform = validate_tessera_scheme(_scheme(TESSERA_FP8, "E4M3", 1024, roles), "t")
    assert uniform["q256"] == 1024 and uniform["role_q256"] == [1024, 1024, 1024]
    # ... and so does a list that happens to agree with itself.
    assert validate_tessera_scheme(
        _scheme(TESSERA_FP8, "E4M3", [1024] * 3, roles), "t")["q256"] == 1024


def test_a_rung_list_that_does_not_line_up_with_its_roles_is_refused():
    roles = [(n, r) for n, r, _ in GROUP]
    with pytest.raises(ValueError, match="per-role list of 2 rungs"):
        validate_tessera_scheme(_scheme(TESSERA_FP8, "E4M3", [900, 1024], roles), "t")
    with pytest.raises(ValueError, match=r"q256\[1\] must be an integer"):
        validate_tessera_scheme(_scheme(TESSERA_FP8, "E4M3", [900, "1024", 1200], roles), "t")


def test_every_role_goes_through_the_reader_range_gate_the_module_used_to():
    """The relaxation does not widen what the decoder promises to read.

    2049 is one past the E4M3 family's published ceiling, and it is refused
    naming the ROLE -- an operator cannot act on "somewhere in this module".
    """
    roles = [(n, r) for n, r, _ in GROUP]
    with pytest.raises(ValueError, match=r"role 'v_proj'.*q256=2049"):
        validate_tessera_scheme(_scheme(TESSERA_FP8, "E4M3", [900, 1024, 2049], roles), "t")


# --- the exporter's grouping key ---------------------------------------------

def test_the_grouping_key_is_the_module_scheme_and_not_the_rate():
    export = _exporter()
    e4m3, bf16 = grid_for_name("E4M3"), grid_for_name("BF16")
    k2 = grid_for_name("E2M1x2")
    # Two rungs, one key: this is the group the old ``(grid, q256)`` check
    # passed through at source precision.
    assert export.module_scheme_key(e4m3, 900) == export.module_scheme_key(e4m3, 1200)
    # Two families: still separated, because vLLM builds one method per module.
    assert export.module_scheme_key(e4m3, 1024) != export.module_scheme_key(bf16, 1024)
    # Two BODIES on one grid: separated, because they are two decoders.  The
    # E2M1x2 sub-cap window body is refused by ``check_recipe`` long before
    # this, and the key does not lean on that.
    assert export.module_scheme_key(k2, 512) != export.module_scheme_key(k2, 896)


# --- the decode --------------------------------------------------------------

def _encode(grid, device):
    torch.manual_seed(7)
    blobs = []
    for i, (name, rows, q256) in enumerate(GROUP):
        weight = (torch.randn(rows, COLUMNS, device=device) * 0.02)
        weight[: rows // 8] *= 2.0 ** (i + 1)
        exported, _unit, _forests = encode_linear_planes(
            weight.contiguous(), grid=grid, q256=q256, name=name, verify=False)
        blobs.append((name, exported.rows, exported.blob))
    return blobs


@pytest.mark.parametrize("grid,family,prepare_name", [
    (E4M3_GRID, TESSERA_FP8, "prepare_tessera_fp8_module"),
    (BF16_GRID, TESSERA_BF16, "prepare_tessera_bf16_module"),
])
@requires_cuda
def test_a_mixed_rung_group_decodes_exactly_as_its_roles_decode_alone(grid, family, prepare_name):
    """The relaxation is bit-exact against the path it relaxes.

    Arm A is what the exporter wrote before #37 for these three tensors: three
    one-member containers, each with its own scalar-``q256`` scheme, prepared
    and decoded independently.  Arm B is the one container with a per-role rung
    list.  Every element and every row scale must agree; if any of the module's
    decode were driven by one module-level rate, the three roles could not all
    come back right.
    """
    if family == TESSERA_FP8:
        from tessera.serving.fp8_route import prepare_tessera_fp8_module as prepare
    else:
        from tessera.serving.bf16_route import prepare_tessera_bf16_module as prepare
    device = torch.device("cuda")
    blobs = _encode(grid, device)
    roles = [(n, r) for n, r, _ in GROUP]
    rungs = [q for _, _, q in GROUP]

    # A: the unrelaxed path -- one member, one scalar rung, one module each.
    singles = []
    for (name, rows, blob), rung in zip(blobs, rungs):
        one = pack_fused([(name, rows, blob)])
        scheme = _scheme(family, grid.name, rung, [(name, rows)], wire_bytes=len(one))
        parsed = parse_tessera_blob_for_scheme(one, scheme, f"t.{name}", device=device)
        assert [n for n, _ in parsed] == [name]
        singles.append(prepare(parsed, device=device))

    # B: the relaxed path -- one container, one rung per role.
    container = pack_fused(blobs)
    scheme = _scheme(family, grid.name, rungs, roles, wire_bytes=len(container))
    parsed = parse_tessera_blob_for_scheme(container, scheme, "t.qkv", device=device)
    assert [n for n, _ in parsed] == [n for n, _ in roles]
    module = prepare(parsed, device=device)

    assert torch.equal(module.decode(), torch.cat([m.decode() for m in singles], 0))
    assert torch.equal(module.row_scale(), torch.cat([m.row_scale() for m in singles], 0))
    assert module.decode().shape[0] == sum(r for _, r in roles)


@requires_cuda
def test_the_parse_holds_each_member_to_its_own_declared_rung():
    """A sidecar that promised the wrong rate for a role is a refusal.

    The list is read POSITIONALLY against ``roles``, so the failure mode worth
    refusing is a permuted one: the bytes are all legal, every rung is legal,
    and only the pairing is wrong.  It is caught because each member's manifest
    is compared against the rate the sidecar promised for THAT member.
    """
    device = torch.device("cuda")
    blobs = _encode(E4M3_GRID, device)
    container = pack_fused(blobs)
    roles = [(n, r) for n, r, _ in GROUP]
    rungs = [q for _, _, q in GROUP]
    swapped = [rungs[1], rungs[0], rungs[2]]
    scheme = _scheme(TESSERA_FP8, "E4M3", swapped, roles, wire_bytes=len(container))
    with pytest.raises(ValueError, match=r"role 'q_proj'.*refusing rather than serving"):
        parse_tessera_blob_for_scheme(container, scheme, "t.qkv", device=device)
