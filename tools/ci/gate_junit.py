"""Keep xdist scheduling suffixes outside test policy identities."""


def test_name(name):
    original, separator, group = name.rpartition('@')
    return original if separator and ']' not in group else name
