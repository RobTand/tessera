"""Temporary new-test probe for the native Bazel cutover gate."""


def test_new_cpu_target_is_executed():
    assert sum((2, 3, 5, 7)) == 17
