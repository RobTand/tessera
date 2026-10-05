"""``tessera.dev_mode.seal_check``: the one run-identity seal helper.

D32: dev mode is ON unless ``PRISMAQUANT_DEV_MODE`` is exactly ``0``; a
run-identity mismatch stamps one ``[DEV-MODE]`` line and continues with the
stored data. Certified mode (``0``) keeps the refusal the site raised before
dev mode. Byte integrity is out of scope here and stays in both modes.
"""
import pytest

from tessera.dev_mode import (
    DEV_MODE_ENV,
    NOT_COMPUTED,
    dev_mode_enabled,
    dev_warning,
    seal_check,
)


def test_dev_mode_is_on_unless_exactly_zero() -> None:
    for value in (None, "", "1", "true", "0 ", "00", "false"):
        environ = {} if value is None else {DEV_MODE_ENV: value}
        assert dev_mode_enabled(environ), value
    assert not dev_mode_enabled({DEV_MODE_ENV: "0"})
    assert dev_mode_enabled()


def test_agreement_returns_true_and_prints_nothing(capsys) -> None:
    assert seal_check("serving source", "a" * 64, "a" * 64, where="t") is True
    assert capsys.readouterr().out == ""


def test_dev_mismatch_stamps_once_and_returns_false(capsys) -> None:
    recorded, running = "a" * 64, "b" * 64
    assert seal_check("serving source", recorded, running,
                      where="lane eligibility") is False
    out = capsys.readouterr().out
    assert out.count("[DEV-MODE]") == 1
    assert "seal serving source differs" in out
    assert recorded in out and running in out
    assert "continuing with the stored data" in out


def test_certified_zero_refuses_with_generated_message() -> None:
    with pytest.raises(RuntimeError, match="source pin differs"):
        seal_check("source pin", "a" * 64, "b" * 64, where="t",
                   environ={DEV_MODE_ENV: "0"})


def test_certified_refusal_preserves_the_site_exception() -> None:
    class PinRefusal(RuntimeError):
        pass

    instance = PinRefusal("the old refusal, verbatim")
    with pytest.raises(PinRefusal, match="^the old refusal, verbatim$"):
        seal_check("source pin", "a", "b", where="t", refusal=instance,
                   environ={DEV_MODE_ENV: "0"})
    with pytest.raises(PinRefusal, match="source pin differs"):
        seal_check("source pin", "a", "b", where="t", refusal=PinRefusal,
                   environ={DEV_MODE_ENV: "0"})
    with pytest.raises(PinRefusal, match="source pin differs"):
        seal_check("source pin", "a", "b", where="t", refusal=lambda: PinRefusal("made"),
                   environ={DEV_MODE_ENV: "0"})


def test_same_overrides_plain_equality() -> None:
    assert seal_check("canonical bytes", b"ab", b"ab", where="t",
                      same=True, environ={DEV_MODE_ENV: "0"}) is True
    assert seal_check("canonical bytes", b"ab", b"cd", where="t", same=False) is False


def test_not_computed_stamps_in_dev_mode_and_refuses_certified(capsys) -> None:
    assert seal_check("source digest", "a" * 64, NOT_COMPUTED, where="t") is False
    out = capsys.readouterr().out
    assert out.count("[DEV-MODE]") == 1
    assert "not computed" in out
    assert "a" * 64 in out
    with pytest.raises(RuntimeError, match="differs"):
        seal_check("source digest", "a" * 64, NOT_COMPUTED, where="t",
                   environ={DEV_MODE_ENV: "0"})


def test_mapping_mismatch_names_the_first_differing_field(capsys) -> None:
    recorded = {"tessera_commit": "c" * 40, "serving_source_sha256": "a" * 64}
    running = {"tessera_commit": "c" * 40, "serving_source_sha256": "b" * 64}
    assert seal_check("runtime code", recorded, running, where="t") is False
    out = capsys.readouterr().out
    assert "at serving_source_sha256" in out


def test_dev_warning_carries_the_grep_able_prefix(capsys) -> None:
    dev_warning("seal example")
    assert capsys.readouterr().out == "[DEV-MODE] seal example\n"
