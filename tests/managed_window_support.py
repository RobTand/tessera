"""Clock and process observations shared by live Envelope controls."""
from pathlib import Path


class FakeClock:
    """One fake clock for managed_window during a test."""

    def __init__(self, monotonic=100.0):
        self.monotonic_value, self.unix_value = monotonic, 1_800_000_000.0
        self.sleeps = []

    def time(self):
        return self.unix_value

    def monotonic(self):
        return self.monotonic_value

    def sleep(self, seconds):
        assert seconds >= 0, "no wait may compute a negative sleep"
        self.sleeps.append(seconds)
        self.monotonic_value += seconds
        self.unix_value += seconds

    def advance(self, seconds):
        self.monotonic_value += seconds
        self.unix_value += seconds


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False


