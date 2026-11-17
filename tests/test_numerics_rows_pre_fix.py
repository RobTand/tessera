"""Terminal exceptions decide whether a pre-fix failure tests behavior."""
import xml.etree.ElementTree as ET

import pytest

from experiments.numerics_rows.pre_fix import selection_result


@pytest.mark.parametrize("text,message,kind,behavioral", [
    ("# An ImportError would be incidental.\nE   AssertionError: feature absent\ntest.py:21: AssertionError", "", "AssertionError", True),
    ("E   ImportError: inner\nE   AssertionError: feature absent\ntest.py:21: AssertionError", "", "AssertionError", True),
    ("E   AssertionError: inner\nE   ModuleNotFoundError: missing\ntest.py:21: ModuleNotFoundError", "", "ModuleNotFoundError", False),
    ("E   ImportError: cannot import name", "", "ImportError", False),
    ("E   Failed: DID NOT RAISE Invalid", "", "Failed", True),
    ("", "Failed: DID NOT RAISE Invalid", "Failed", True),
    ("E   KeyError: missing field", "", "KeyError", False),
    ("E   TypeError: unsupported argument", "", "TypeError", False),
    ("unrecognized traceback", "", None, False),
])
def test_pre_fix_uses_the_terminal_exception(text, message, kind, behavioral):
    xml = ET.fromstring("<testsuite><testcase name='case'><failure/></testcase></testsuite>")
    failure = xml.find("testcase/failure")
    failure.text = text
    failure.set("message", message)
    result = selection_result(xml, 1, 1)
    assert result["terminal_exception_types"] == [kind]
    assert result["matched"] is result["behavioral_failures"] is behavioral


def test_pre_fix_refusal_types_apply_only_to_the_declared_test():
    xml = ET.fromstring("<testsuite><testcase name='case[cut]'><failure>E   Invalid: refused</failure></testcase></testsuite>")
    assert not selection_result(xml, 1, 1)["matched"]
    assert not selection_result(xml, 1, 1, {"other": ["Invalid"]})["matched"]
    assert selection_result(xml, 1, 1, {"case": ["Invalid"]})["matched"]
    xml.find("testcase/failure").text = "E   ImportError: absent"
    assert not selection_result(xml, 1, 1, {"case": ["ImportError"]})["matched"]
