"""Event-driven local device/process boundaries for the real two-rank protocol.

This proves lifecycle control and retained evidence, not OS process teardown,
Docker, CUDA, a physical memory floor or broker ownership release. The separate
subprocess scenarios retain the OS process-group coverage.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Event

import managed_window as window
import rank_window


class ProtocolClock:
    """Protocol polling yields to publication events; only a probe timeout advances time."""

    def __init__(self):
        self.condition = Condition()
        self.elapsed = 0.0

    def time(self):
        return window.WINDOW_SECONDS + self.elapsed

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        if not self.condition.wait(timeout=30):
            raise AssertionError("rank protocol made no progress after an explicit event")


class LocalBoundary:
    def __init__(self, simulation, rank, name, envelope):
        self.simulation, self.rank, self.name, self.envelope = simulation, rank, name, envelope
        self.guard = None
        self.active = None

    def record(self, operation, arm):
        self.simulation.events.append((operation, self.rank, arm))

    def preflight(self):
        if self.name == "preflight" and self.rank == 1:
            raise window.Refused("simulated preflight refusal")
        return dict(image="fixture-image", src_sha256="a" * 64, config_sha256="b" * 64)

    def start(self, arm):
        self.active = arm["arm"]
        self.simulation.active.add((self.rank, self.active))
        self.record("start", self.active)
        return dict(cid=f"simulated-rank{self.rank}-{self.active}")

    def tick(self):
        if self.guard:
            self.guard()
        if self.name == "floor" and self.rank == 1:
            raise window.Refused("simulated physical memory floor refusal")

    def ready(self, arm):
        self.tick()
        self.record("ready", arm["arm"])
        return dict(fabric="socket")

    def probes(self, arm, peer):
        stage = arm["arm"]
        self.record("probe", stage)
        log = self.simulation.rdv / f"{stage}.probes.log"
        if self.name in ("probe_failure", "timeout"):
            log.write_text("partial\n")
            if self.name == "timeout":
                self.simulation.clock.elapsed = self.envelope.end - self.envelope.cleanup_seconds
                self.simulation.clock.condition.notify_all()
                self.envelope.remaining()
            raise window.Refused("simulated probe command failed: 7")
        log.write_text("two full passes and screens simulated\n")

    def cleanup(self, arm):
        stage = arm["arm"]
        if self.active is not None:
            self.simulation.active.remove((self.rank, self.active))
            self.record("stop", self.active)
            self.active = None
        if self.name == "copy_failure" and self.rank == 1:
            raise window.Refused("simulated evidence copy failed")
        self.record("cleaned", stage)
        return dict(containers_empty=True, gpu_descendants_empty=True)


class Lifecycle:
    def __init__(self, root, arms, clock):
        self.rdv = root / "rdv"
        self.rdv.mkdir()
        self.arms, self.clock = arms, clock
        self.events = []
        self.active = set()
        self.outcomes = []

    def run(self, root, name, mode, first_rank):
        queue = root / "queue"
        entered = [Event(), Event()]

        def rank(ordinal):
            with self.clock.condition:
                entered[ordinal].set()
                key, nonce = str(ordinal + 1) * 64, str(ordinal + 3) * 32
                owned = dict(rank=ordinal, action_key=key, nonce=nonce,
                             scope_id=window.scope_name(key, nonce), host=window.HOSTS[ordinal],
                             container_owner=f"simulated-owner-{ordinal}", claimed_unix=self.clock.time(),
                             run_id="simulated-fresh-run", input_sha256="a" * 64,
                             window_end_unix=self.clock.time() + window.WINDOW_SECONDS)
                window.atomic_json(queue / "claimed" / f"{key}.json",
                                   dict(action_key=key, claimed_unix=owned["claimed_unix"],
                                        claimed_host=owned["host"], container_owner=owned["container_owner"],
                                        resource_scope=dict(action_key=key, nonce=nonce, scope_id=owned["scope_id"])))
                envelope = window.Envelope(owned["window_end_unix"])
                adapter = LocalBoundary(self, ordinal, name, envelope)
                try:
                    return rank_window.run_rank(dict(fabric="socket", window_mode=mode), owned, queue, self.rdv,
                                                [dict(arm=arm) for arm in self.arms], adapter, envelope)
                finally:
                    (queue / "claimed" / f"{key}.json").unlink()
                    self.clock.condition.notify_all()

        (queue / "claimed").mkdir(parents=True)
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(rank, first_rank)
            assert entered[first_rank].wait(timeout=30), "first rank did not enter the protocol"
            second = workers.submit(rank, 1 - first_rank)
            first.result(timeout=30)
            second.result(timeout=30)
        self.outcomes = [window.read_json(self.rdv / f"outcome-rank{ordinal}.json") for ordinal in (0, 1)]
        return self


def scenario(monkeypatch, root, name, *, mode, arms, first_rank=0):
    clock = ProtocolClock()
    simulation = Lifecycle(root, arms, clock)
    atomic_json = window.atomic_json

    def publish(path, value):
        with clock.condition:
            atomic_json(path, value)
            if value.get("stage") == "failed":
                simulation.events.append(("failed", value["rank"], "failed"))
            clock.condition.notify_all()

    with monkeypatch.context() as patch:
        patch.setattr(window, "time", clock)
        patch.setattr(rank_window, "time", clock)
        patch.setattr(window, "atomic_json", publish)
        patch.setattr(rank_window, "atomic_json", publish)
        return simulation.run(root, name, mode, first_rank)
