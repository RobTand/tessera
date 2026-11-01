"""CPU acceptance checks for the three prefill selectors in issue #1250.

Exercise the real readers without installing vLLM hooks or relying on their
per-config installation caches. Each scenario checks existing parsing first,
then the missing process-stability contract. Shared resets isolate scenarios.
"""
import pytest

from tessera.serving import flags
from tessera.serving import glm53_prefill as gp


ONORM = "TESSERA_GLM53_ONORM_CUDA"
SP = "TESSERA_GLM53_SP_MHC"
KDA = "TESSERA_GLM53_KDA_CONV_SPLIT"
REQUIRED = "TESSERA_TEST_REQUIRED_PREFILL_MODE"
READERS = {
    ONORM: gp.onorm_cuda_enabled,
    SP: gp.sp_mode,
    KDA: gp.kda_conv_split_mode,
}


@pytest.fixture(autouse=True)
def fresh_flags(monkeypatch):
    for flag in (*READERS, REQUIRED):
        flags.reset_for_tests(flag)
        monkeypatch.delenv(flag, raising=False)
    try:
        yield
    finally:
        for flag in (*READERS, REQUIRED):
            flags.reset_for_tests(flag)


def _setting(monkeypatch, flag, raw):
    if raw is None:
        monkeypatch.delenv(flag, raising=False)
    else:
        monkeypatch.setenv(flag, raw)


def _requires_restart(reader, flag):
    with pytest.raises(RuntimeError, match=flag) as error:
        reader()
    assert "restart" in str(error.value).lower()


def _required_selector_stays_required(monkeypatch):
    """An opt-in mode default must not change the existing no-default API."""
    kwargs = dict(modes=("resident", "stream"), meaning="test residency",
                  unset_help="declare the test residency explicitly")
    with pytest.raises(ValueError, match=REQUIRED) as error:
        flags.latched_mode(REQUIRED, **kwargs)
    assert kwargs["unset_help"] in str(error.value)
    monkeypatch.setenv(REQUIRED, " ReSiDeNt ")
    assert flags.latched_mode(REQUIRED, **kwargs) == "resident"
    monkeypatch.setenv(REQUIRED, "stream")
    _requires_restart(lambda: flags.latched_mode(REQUIRED, **kwargs), REQUIRED)


@pytest.mark.parametrize("flag,raw,want,repeat,changed,changed_want", [
    pytest.param(ONORM, None, False, None, "1", True, id="onorm-unset"),
    pytest.param(ONORM, "", False, "  ", "1", True, id="onorm-empty"),
    pytest.param(ONORM, " \t ", False, "", "1", True, id="onorm-blank"),
    pytest.param(ONORM, "0", False, " 0 ", "1", True, id="onorm-zero"),
    pytest.param(ONORM, "1", True, " 1 ", "0", False, id="onorm-one"),
    pytest.param(ONORM, " 1 ", True, "1", "0", False, id="onorm-trimmed"),
    pytest.param(SP, None, "off", "off", "auto", "auto", id="sp-unset"),
    pytest.param(SP, "off", "off", " OFF ", "force", "force", id="sp-off"),
    pytest.param(SP, "auto", "auto", " AUTO ", "off", "off", id="sp-auto"),
    pytest.param(SP, "force", "force", " FORCE ", "off", "off", id="sp-force"),
    pytest.param(SP, " FoRcE ", "force", "force", "auto", "auto", id="sp-normalized"),
    pytest.param(KDA, None, "off", "off", "on", "on", id="kda-unset"),
    pytest.param(KDA, "off", "off", " OFF ", "on", "on", id="kda-off"),
    pytest.param(KDA, "on", "on", " ON ", "off", "off", id="kda-on"),
    pytest.param(KDA, " On ", "on", "on", "off", "off", id="kda-normalized"),
])
def test_accepted_settings_and_default_latch(monkeypatch, flag, raw, want,
                                             repeat, changed, changed_want):
    if flag != ONORM:
        _required_selector_stays_required(monkeypatch)
    reader = READERS[flag]
    _setting(monkeypatch, flag, raw)
    assert reader() == want
    _setting(monkeypatch, flag, repeat)
    assert reader() == want  # normalized spellings remain idempotent

    _setting(monkeypatch, flag, changed)
    _requires_restart(reader, flag)
    _setting(monkeypatch, flag, repeat)
    assert reader() == want  # a refused mutation did not replace the latch

    # The public serving reset must own the latch, not a new reader-local cache.
    flags.reset_for_tests(flag)
    _setting(monkeypatch, flag, changed)
    assert reader() == changed_want


@pytest.mark.parametrize("flag,invalid,valid,want,changed,accepted", [
    *[pytest.param(ONORM, raw, "1", True, "0", ("0", "1"),
                   id=f"onorm-invalid-{label}")
      for label, raw in (("true", "true"), ("yes", "yes"), ("mode", "on"),
                         ("two", "2"))],
    *[pytest.param(SP, raw, "auto", "auto", "force", ("auto", "off", "force"),
                   id=f"sp-invalid-{label}")
      for label, raw in (("empty", ""), ("blank", " \t "), ("true", "true"),
                         ("boolean", "1"), ("mode", "on"), ("unknown", "sometimes"))],
    *[pytest.param(KDA, raw, "on", "on", "off", ("on", "off"),
                   id=f"kda-invalid-{label}")
      for label, raw in (("empty", ""), ("blank", " \t "), ("true", "true"),
                         ("boolean", "1"), ("mode", "auto"), ("unknown", "sometimes"))],
])
def test_invalid_diagnostics_do_not_poison_latch(monkeypatch, flag, invalid,
                                                valid, want, changed, accepted):
    reader = READERS[flag]
    _setting(monkeypatch, flag, invalid)
    with pytest.raises(ValueError, match=flag) as error:
        reader()
    message = str(error.value)
    assert all(value in message for value in accepted), message
    assert repr(invalid.strip().lower()) in message, message

    _setting(monkeypatch, flag, valid)
    assert reader() == want  # only the first *successfully parsed* value latches
    _setting(monkeypatch, flag, invalid)
    with pytest.raises(ValueError, match=flag):
        reader()  # parsing remains strict even after a successful first read
    _setting(monkeypatch, flag, valid)
    assert reader() == want
    _setting(monkeypatch, flag, changed)
    _requires_restart(reader, flag)


@pytest.mark.parametrize("flag,enabled,want", [
    pytest.param(ONORM, "1", True, id="onorm"),
    pytest.param(SP, "force", "force", id="sp"),
    pytest.param(KDA, "on", "on", id="kda"),
])
def test_unsetting_an_enabled_flag_requires_restart(monkeypatch, flag, enabled, want):
    reader = READERS[flag]
    monkeypatch.setenv(flag, enabled)
    assert reader() == want
    monkeypatch.delenv(flag)
    _requires_restart(reader, flag)
