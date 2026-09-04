"""Issue #92: the top-level ``format`` a compressed-tensors export declares.

Both exporters wrote the constant ``"mixed-precision"`` into
``quantization_config.format`` whatever they had just built.  The stock NVFP4
comparator twin is not mixed -- one group, ``nvfp4-pack-quantized``, over every
target -- and the field is the whole of vLLM's FP4-model predicate
(``ModelConfig.is_nvfp4_quantized``, read below), so the comparator arm of a
speed comparison answered False where a uniform-NVFP4 checkpoint from anyone
else answers True.  Nothing in our receipts recorded the difference.

These tests pin three things, all offline:

* a declaration whose groups agree on one format says THAT format, and one
  whose groups disagree says ``mixed-precision`` -- the honest label for a
  genuinely mixed artifact stays, and it stops being a constant;
* every export that writes the field also writes the resolved predicate, so
  the consequence of mixing is recorded and priced rather than silent;
* the recorded predicate is vLLM's, not a paraphrase of it -- the attestation
  carries the image and version it was read in, because a runtime that moves
  the check makes a stamped record stale (AGENTS principle 14).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tessera.stock import (
    FLOAT_QUANTIZED, MIXED_PRECISION, NVFP4_PACK_QUANTIZED,
    VLLM_FP4_PREDICATE_ATTESTATION, declared_format, vllm_fp4_predicate,
)

ROOT = Path(__file__).resolve().parents[1]


def _group(fmt, *targets):
    return {"format": fmt, "weights": {}, "targets": list(targets)}


def _exporter():
    """``export_stock_compressed`` by path, as the export-gate tests load theirs."""
    spec = importlib.util.spec_from_file_location(
        "export_stock_compressed", ROOT / "experiments" / "export_stock_compressed.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the label ---------------------------------------------------------------

def test_a_uniform_nvfp4_export_declares_nvfp4_not_mixed():
    groups = {"group_0": _group(NVFP4_PACK_QUANTIZED, "re:.*q_proj.*")}
    assert declared_format(groups) == NVFP4_PACK_QUANTIZED


def test_a_uniform_fp8_export_declares_float_quantized():
    assert declared_format({"group_0": _group(FLOAT_QUANTIZED, "re:.*")}) == FLOAT_QUANTIZED


def test_groups_that_agree_are_uniform_however_many_keys_they_use():
    """The label describes the weights, not the exporter's bookkeeping.

    vLLM's predicate asks what the checkpoint is; two keys that name the same
    format still name one format, and calling that artifact mixed would be as
    wrong as calling the one-key one mixed.
    """
    groups = {"group_0": _group(NVFP4_PACK_QUANTIZED, "re:.*attn.*"),
              "group_1": _group(NVFP4_PACK_QUANTIZED, "re:.*mlp.*")}
    assert declared_format(groups) == NVFP4_PACK_QUANTIZED


def test_a_genuinely_mixed_export_keeps_the_mixed_label():
    groups = {"group_0": _group(NVFP4_PACK_QUANTIZED, "re:.*mlp.*"),
              "group_1": _group(FLOAT_QUANTIZED, "re:.*attn.*")}
    assert declared_format(groups) == MIXED_PRECISION


def test_no_groups_has_no_format_to_declare():
    with pytest.raises(ValueError, match="declares no"):
        declared_format({})


# --- the recorded consequence ------------------------------------------------

def test_the_uniform_nvfp4_declaration_resolves_the_predicate_true():
    record = vllm_fp4_predicate("compressed-tensors", NVFP4_PACK_QUANTIZED)
    assert record["vllm_is_nvfp4_quantized"] is True


def test_the_mixed_declaration_records_the_fusion_it_gives_up():
    """Mixing is allowed; mixing silently is not.

    The mixed label is the honest one, and this record is what makes the loss a
    priced property of the artifact instead of a side effect nobody reads.
    """
    record = vllm_fp4_predicate("compressed-tensors", MIXED_PRECISION)
    assert record["vllm_is_nvfp4_quantized"] is False
    assert "does not contain 'nvfp4'" in record["reason"]
    assert "fuse_act_quant" in record["consequence"]


def test_the_tessera_route_fails_the_predicate_on_quant_method_not_on_format():
    """Our own plugin's checkpoints keep ``mixed-precision``, and say why.

    ``is_nvfp4_quantized`` requires ``quantization == "compressed-tensors"``,
    which a ``quant_method: "tessera"`` checkpoint is not, and
    ``tessera.serving.config`` never reads the top-level field.  Renaming it
    there would change nothing and would claim something.  The record says the
    predicate is False for a reason that no format string can alter.
    """
    record = vllm_fp4_predicate("tessera", MIXED_PRECISION)
    assert record["vllm_is_nvfp4_quantized"] is False
    assert "quant_method" in record["reason"]
    nvfp4_named = vllm_fp4_predicate("tessera", NVFP4_PACK_QUANTIZED)
    assert nvfp4_named["vllm_is_nvfp4_quantized"] is False


def test_the_record_carries_the_runtime_it_was_read_in():
    attested = VLLM_FP4_PREDICATE_ATTESTATION
    assert attested["version"] == "0.28.0"
    assert attested["image_id"].startswith("sha256:")
    assert attested["box"] == "sparky"
    assert attested["predicate"]["path"] == "vllm/config/model.py"
    assert '"nvfp4" in quant_config.get("format", "").lower()' in attested["predicate"]["source"]
    assert "enforce-eager" in attested["scope"]
    # The tag floats between the boxes (tessera#100), so a record naming only
    # the tag would not say which build answered.
    assert attested["image"] == "vllm/vllm-openai:latest"
    assert len(attested["image_id"]) == len("sha256:") + 64
    # Resolved, not deferred: a stamped "pending" would be a record that reads
    # like an answer and is not one.
    assert attested["nvfp4_pattern_built"].startswith("yes.")
    assert "PENDING" not in attested["nvfp4_pattern_built"]


def test_the_predicate_is_the_quoted_source_and_not_a_paraphrase():
    """Re-derive the recorded answer from the attested expression itself.

    If someone edits the predicate's implementation without editing the quote,
    or the other way round, the two stop agreeing here.
    """
    source = VLLM_FP4_PREDICATE_ATTESTATION["predicate"]["source"]
    for quant_method, fmt in [
        ("compressed-tensors", NVFP4_PACK_QUANTIZED),
        ("compressed-tensors", FLOAT_QUANTIZED),
        ("compressed-tensors", MIXED_PRECISION),
        ("tessera", NVFP4_PACK_QUANTIZED),
    ]:
        expected = eval(source, {}, {  # noqa: S307 -- the point is to run vLLM's own line
            "self": type("M", (), {"quantization": quant_method})(),
            "quant_config": {"format": fmt},
        })
        assert vllm_fp4_predicate(quant_method, fmt)["vllm_is_nvfp4_quantized"] is bool(expected)


# --- the exporters go through it ---------------------------------------------

def test_the_stock_exporter_declares_the_group_it_wrote():
    module = _exporter()
    block, record = module.stock_quantization_config(
        {"group_0": _group(NVFP4_PACK_QUANTIZED, "re:.*")}, ["lm_head"])
    assert block["format"] == NVFP4_PACK_QUANTIZED
    assert block["quant_method"] == "compressed-tensors"
    assert record["vllm_is_nvfp4_quantized"] is True


def test_the_stock_exporter_mixes_and_records_the_consequence():
    module = _exporter()
    block, record = module.stock_quantization_config(
        {"group_0": _group(NVFP4_PACK_QUANTIZED, "re:.*mlp.*"),
         "group_1": _group(FLOAT_QUANTIZED, "re:.*attn.*")}, ["lm_head"])
    assert block["format"] == MIXED_PRECISION
    assert record["vllm_is_nvfp4_quantized"] is False
    assert record["attested"]["version"] == "0.28.0"


def test_nothing_quantized_writes_no_quantization_config():
    """A checkpoint of plain bf16 tensors must not advertise compressed ones."""
    module = _exporter()
    block, record = module.stock_quantization_config({}, ["lm_head"])
    assert block is None and record is None


def test_neither_exporter_hardcodes_the_top_level_format_any_more():
    """Pin the rule, not today's two call sites.

    A third exporter that reintroduces the constant fails here, which is the
    point: the defect was a literal, and a literal is what must not come back.
    """
    offenders = []
    for path in sorted((ROOT / "experiments").rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if '"format": "mixed-precision"' in line or "'format': 'mixed-precision'" in line:
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert offenders == [], (
        "the top-level format is derived from the config groups "
        f"(tessera.stock.declared_format); hardcoded at {offenders}")
