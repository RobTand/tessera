"""Issues #66-68: encoder-fit caps must not stop a descent that is still moving.

Each test pins the RULE -- converge on the computed quantity, caps are
backstops -- with the expected value derived from the code's own converged
reference, not from a roster of names.
"""

from tessera.alphabet import GAUSSIAN_SOURCE, _lloyd_levels


def test_lloyd_default_reaches_the_converged_levels():
    """#66: the default budget must not truncate Lloyd while levels move."""
    source = GAUSSIAN_SOURCE(1 << 12)
    assert _lloyd_levels(source, 32) == _lloyd_levels(source, 32, iterations=5000)
