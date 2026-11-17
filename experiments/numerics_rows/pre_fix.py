"""Check pre-fix failures by terminal exception, not traceback source text."""
import re


def terminal_type(failure):
    text = failure.text or ""
    types = re.findall(r"(?m)^\S+:\d+: ([\w.]+)$", text)
    if not types:
        types = re.findall(r"(?m)^E\s+([\w.]+):", text)
    if not types:
        types = re.findall(r"^([\w.]+):", failure.get("message", ""))
    return types[-1].rsplit(".", 1)[-1] if types else None


def selection_result(xml, exit_code, count, expected_refusals=None):
    cases = xml.findall(".//testcase") if xml is not None else []
    failures = [(case, failure) for case in cases for failure in case.findall("failure")]
    errors = xml.findall(".//testcase/error") if xml is not None else []
    terminal = [terminal_type(failure) for _, failure in failures]
    unexpected = [{"node": case.get("name"), "type": kind}
        for (case, _), kind in zip(failures, terminal)
        if kind in (None, "ImportError", "ModuleNotFoundError") or kind not in {
            "AssertionError", "Failed", *(expected_refusals or {}).get(case.get("name", "").split("[", 1)[0], [])}]
    return {"selected": len(cases), "expected_selected": count, "expected_failures": count,
        "failures": len(failures), "errors": len(errors) if xml is not None else -1,
        "selection_error": "selected 0" if not cases else None,
        "terminal_exception_types": terminal, "non_behavioral_failures": unexpected,
        "behavioral_failures": not unexpected,
        "matched": xml is not None and exit_code == 1 and len(cases) == count == len(failures) and not errors and not unexpected,
        "failure_lines": [failure.text or "" for _, failure in failures]}
