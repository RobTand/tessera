"""Issues #66-68: encoder-fit caps must not stop a descent that is still moving.

Each test pins the RULE -- converge on the computed quantity, caps are
backstops -- with the expected value derived from the code's own converged
reference, not from a roster of names.
"""

from tessera.alphabet import E4M3_GRID, GAUSSIAN_SOURCE, _lloyd_levels
import tessera.alphabet as alphabet


def test_lloyd_default_reaches_the_converged_levels():
    """#66: the default budget must not truncate Lloyd while levels move."""
    source = GAUSSIAN_SOURCE(1 << 12)
    assert _lloyd_levels(source, 32) == _lloyd_levels(source, 32, iterations=5000)


def test_mass_balanced_fit_uses_the_full_sample(monkeypatch):
    """#67: Lloyd targets are fit on the full source, not every 4th sample."""
    seen = []
    real = alphabet._lloyd_levels

    def spy(source, size, *args, **kwargs):
        seen.append(len(source))
        return real(source, size, *args, **kwargs)

    monkeypatch.setattr(alphabet, "_lloyd_levels", spy)
    samples = GAUSSIAN_SOURCE(1 << 10)
    alphabet._mass_balanced_blocks(E4M3_GRID, samples, 8, 32)
    assert seen == [len(samples)]
