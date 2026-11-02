"""Whether this thread is encoding the encoder-identity fixtures.

:mod:`tessera.encoder_identity` computes the encoder's behavioural identity by
running the real exporter, and so the real encoder, over fixed fixtures. While
it does, nothing may stamp that identity (it is not known yet), and the
encoder's refit diagnostic must not count the fixture encodes as refits of the
unit being measured. Both ask :func:`building`.

The flag lives here, apart from the identity, because the encoder reads it:
the encoder importing :mod:`tessera.encoder_identity`, which imports the
exporter, which imports the encoder, made every one of them and the serving
stack one import cycle (#1588).
"""
from __future__ import annotations

import threading

_BUILDING = threading.local()


def building() -> bool:
    """True while :func:`tessera.encoder_identity.encoder_fixture_id` is encoding its fixtures.

    The fixture build is the one path that must stamp no identity: the digest
    is not known while it is being computed, and a manifest that carried it
    would make the hash a function of itself.  ``build_unit_artifact`` asks
    :func:`tessera.encoder_identity.stamped_fixture_id`, which reads this.
    """
    return getattr(_BUILDING, "active", False)


class fixture_build:
    """Marks the fixture encodes so they stamp no identity, and forbids nesting.

    Re-entry would mean a call path routed the fixture build back through the
    stamping path -- the recursion this guard exists to make impossible -- so
    it raises rather than quietly returning ``None`` twice.
    """

    def __enter__(self):
        if building():
            raise RuntimeError(
                "the encoder fixture build re-entered itself: something asked "
                "for the encoder identity while it was being computed"
            )
        _BUILDING.active = True
        return self

    def __exit__(self, *exc):
        _BUILDING.active = False
        return False
