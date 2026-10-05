"""Run all three control arms in one admitted LOCAL rank action; never launch a peer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time

from managed_window import (WINDOW_SECONDS, CLEANUP_SECONDS, MEMORY_POLICY, Envelope, HOSTS, Refused,
                            seal_check, check_memory_policy,
                            Rendezvous, atomic_json, read_json, require_claim)
import tp2_recipe as recipe


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def available_gib() -> float:
    lines = Path("/proc/meminfo").read_text().splitlines()
    return int(next(line.split()[1] for line in lines if line.startswith("MemAvailable:"))) / 1048576


def scope_memory(identity):
    """Read the real owned cgroup's host charge; CUDA accounting is not inferred."""
    unified = [line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
               if line.startswith("0::")]
    if len(unified) != 1:
        raise Refused("Window4 sampler has no unique cgroup v2 membership")
    parts = Path(unified[0]).parts[1:]
    if identity["scope_id"] not in parts:
        raise Refused("Window4 memory sample is outside its owned aggregate scope")
    scope = Path("/sys/fs/cgroup").joinpath(*parts[:parts.index(identity["scope_id"]) + 1])
    return dict(unix=time.time(), rank=identity["rank"], mem_available_gib=available_gib(),
                scope_id=identity["scope_id"], cgroup_path=str(scope),
                scope_memory_current_bytes=int((scope / "memory.current").read_text()),
                scope_memory_peak_bytes=int((scope / "memory.peak").read_text()),
                note="Host cgroup charge only; CUDA coverage is unproven. Peak is claim-lifetime, current is sampled.")


def fabric_from_log(log: str) -> str:
    banners = set(re.findall(r"Using network ([A-Za-z_]+)", log))
    if banners == {"Socket"}: return "socket"
    if banners == {"IB"}: return "roce"
    raise Refused(f"missing/mixed/unknown actual NCCL fabric banners: {sorted(banners)}")


class LocalArm:
    """Only local Docker, the ordinary PB shim and already-admitted child clients."""
    def __init__(self, config: dict, identity: dict, envelope: Envelope, rdv: Path):
        self.config, self.identity, self.envelope, self.rdv = config, identity, envelope, rdv
        self.rank = identity["rank"]
        self.work = Path("/home/rob/tmp") / ("ga702-" + identity["run_id"] + "-" + identity["nonce"])
        self.ext = self.work / "ext"
        self.ext.mkdir(parents=True, exist_ok=False)
        self.ext.chmod(0o777)
        self.active = None
        self.containers = []
        self.last_sample = 0
        self.last_maintenance_sample = 0
        self.memory_summary = dict(rank=self.rank, host=identity["host"], samples=0,
                                   baseline_gib=None, minimum_gib=None)
        self.server_terminations = {}
        self.abort = None
        self.image_env = {}
        self.guard = None

    def command(self, argv, **kwargs):
        return self.envelope.run(argv, **kwargs)

    def headroom(self):
        """Wait at most 900 seconds for the D30 107 GiB predicate on this host.

        The existing envelope, peer cancellation and cleanup reserve still bound
        every poll. Retain every wait sample and terminal reason, including errors.
        A ready wait never substitutes for the later synchronous launch recheck.
        """
        threshold_gib = MEMORY_POLICY["start_gib"]
        wait_bound_seconds = MEMORY_POLICY["headroom_wait_seconds"]
        deadline = time.monotonic() + wait_bound_seconds
        started_unix = time.time()
        samples = []
        reason, terminal = "error", None
        try:
            while True:
                sample = dict(unix=time.time(), monotonic=time.monotonic(),
                              available_gib=available_gib())
                samples.append(sample)
                left = self.envelope.remaining()  # raises TimeoutError at the deadline
                if sample["available_gib"] >= threshold_gib:
                    self.tick()  # live guard recheck immediately before declaring ready
                    left = self.envelope.remaining()  # recheck lifetime after the guard
                    if time.monotonic() >= deadline:
                        reason = "headroom_timeout"
                        break
                    reason = "ready"
                    break
                if time.monotonic() >= deadline:
                    reason = "headroom_timeout"
                    break
                self.tick()  # existing guard every poll; peer cancel/floor propagate
                left = self.envelope.remaining()  # recompute after the guard/tick
                bound_left = deadline - time.monotonic()
                if bound_left <= 0:
                    reason = "headroom_timeout"
                    break
                time.sleep(min(getattr(self, "poll_seconds", .2), left, bound_left))
        except Refused as exc:
            reason, terminal = "lifecycle_cancelled", exc
        except TimeoutError as exc:
            reason, terminal = "deadline", exc
        finally:
            ended_unix = time.time()
            report = dict(threshold_gib=threshold_gib, wait_bound_seconds=wait_bound_seconds,
                          initial_available_gib=samples[0]["available_gib"] if samples else None,
                          last_available_gib=samples[-1]["available_gib"] if samples else None,
                          started_unix=started_unix, ended_unix=ended_unix,
                          elapsed_seconds=ended_unix - started_unix,
                          samples=samples, reason=reason)
            with (self.rdv / f"headroom-preflight-rank{self.rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(report, sort_keys=True) + "\n")
        if terminal is not None:
            raise terminal
        if reason != "ready":
            raise Refused("local MemAvailable below 107 GiB preflight")
        return samples

    def preflight(self):
        self.headroom()
        if available_gib() < MEMORY_POLICY["start_gib"]:
            raise Refused("local MemAvailable below 107 GiB preflight")
        runtime = Path(self.config["ts"]) / "src"
        env = dict(os.environ, PYTHONPATH=str(runtime), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MAX_JOBS="1")
        record = json.loads(self.command([sys.executable, "-m", "tessera.serving.runtime_image",
                                         "resolve", "--image", self.config["image"]], env=env, tick=self.tick).stdout)
        seal_check("image reference", self.config["image"], record.get("resolved_reference"),
                   where="Window4 local preflight", refusal=Refused("local image does not resolve to the frozen full image reference"))
        from_source = self.command([sys.executable, "-m", "tessera.serving.runtime_image", "container-env"],
                                   env=env, input_text=json.dumps(record), tick=self.tick).stdout
        self.image_env = dict(line.split("=", 1) for line in from_source.splitlines())
        self.image_id = record["local_id"]
        self.assert_empty()
        current = recipe.inputs(os.environ, live=True,
                   runner=lambda argv, **kw: self.command(argv, tick=self.tick, **kw))
        recipe.check_control_record(self.config, current, where="Window4 local preflight",
                                  refusal=Refused("immutable control/source inputs changed during the window"))
        return dict(image=self.config["image"], image_id=record["local_id"],
                    src_sha256=self.config["src_sha256"], config_sha256=self.config["config_sha256"])

    def tick(self):
        sample = None
        try:
            now = time.monotonic()
            if now - self.last_sample < MEMORY_POLICY["sample_seconds"]:
                if self.guard:
                    self.guard()
                return
            self.last_sample = now
            mem = available_gib()
            sample = dict(unix=time.time(), monotonic=now, rank=self.rank,
                          mem_available_gib=mem, arm=(self.active or {}).get("arm"))
            summary = getattr(self, "memory_summary", dict(rank=self.rank, samples=0,
                                  baseline_gib=None, minimum_gib=None))
            if summary["baseline_gib"] is None:
                summary.update(baseline_gib=mem, baseline_unix=sample["unix"])
            if summary["minimum_gib"] is None or mem < summary["minimum_gib"]:
                summary.update(minimum_gib=mem, minimum_unix=sample["unix"])
            summary.update(samples=summary["samples"] + 1, last_sample=sample,
                           policy=MEMORY_POLICY, identity=self.identity)
            self.memory_summary = summary
            # Local decision precedes EVERY shared-filesystem operation in this tick.
            if mem < MEMORY_POLICY["abort_below_gib"]:
                raise Refused("local MemAvailable below 2 GiB physical memory floor")
            if self.guard:
                self.guard()
            self._persist_memory(sample)
            sample = None  # already durable; do not append it again on a later error
            if self.config.get("window_mode") in recipe.BENCHMARK_PAIRS:
                scope = dict(scope_memory(self.identity), arm=(self.active or {}).get("arm"),
                             mem_available_gib=mem)
                with (self.work / "memory-scope.jsonl").open("a") as stream:
                    stream.write(json.dumps(scope, sort_keys=True) + "\n")
            if now - getattr(self, "last_maintenance_sample", 0) < 5:
                return
            self.last_maintenance_sample = now
            for path, cap in ((self.ext, 6 * (1 << 30)), (self.work, int(6.3 * (1 << 30))),
                              (self.rdv if self.config.get("window_mode") in recipe.BENCHMARK_PAIRS else self.rdv / "arms",
                               (1 << 30) if self.config.get("window_mode") in recipe.BENCHMARK_PAIRS else int(.2 * (1 << 30)))):
                if directory_bytes(path) > cap:
                    raise Refused(f"declared disk/output cap exceeded: {path}")
            if self.active and self.active.get("cid"):
                value = self.inspect_owned(self.active["cid"])
                if value.get("State", {}).get("Running") is not True:
                    raise Refused("owned local server exited before probes completed")
        except BaseException as exc:
            errors = getattr(self, "abort_errors", [])
            self.abort_errors = errors
            # Start the local stop before a potentially blocked peer-marker write.
            # Every diagnostic/termination failure remains secondary to the trigger.
            for name, operation in (
                ("owned TERM", lambda: self._term_server(self.active["cid"]) if self.active and self.active.get("cid") else None),
                ("failure publication", lambda: self.abort(error=f"{type(exc).__name__}: {exc}") if getattr(self, "abort", None) else None),
                ("memory persistence", lambda: self._persist_memory(sample) if sample is not None else None)):
                try:
                    operation()
                except BaseException as secondary:
                    errors.append(dict(operation=name, error=f"{type(secondary).__name__}: {secondary}", unix=time.time()))
            print(json.dumps(dict(event="rank_abort", rank=self.rank,
                                  error=f"{type(exc).__name__}: {exc}", secondary_errors=errors), sort_keys=True), flush=True)
            raise

    def _persist_memory(self, sample):
        with (self.work / "memwatch.txt").open("a") as stream:
            stream.write(f"{sample['unix']} rank{self.rank} MemAvailable_GiB={sample['mem_available_gib']}\n")
        with (self.rdv / f"memory-samples-rank{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
        atomic_json(self.rdv / f"memory-summary-rank{self.rank}.json", self.memory_summary)

    def _container_control(self, cid, operation, *, check=True):
        if self.envelope.end - time.monotonic() <= 5:
            return self.envelope.run_container_control(cid, operation, check=check)
        argv = (["docker", "inspect", cid] if operation == "inspect" else
                ["docker", "kill", "--signal", operation, cid])
        return self.command(argv, cleanup=True, check=check, limit=5)

    def _term_server(self, cid):
        """Signal only an inspected exact-attempt container; never names/foreign scope."""
        stops = getattr(self, "server_terminations", {})
        self.server_terminations = stops
        if cid in stops:
            return stops[cid]
        value = self.inspect_owned(cid)
        record = dict(cid=cid, scope_id=self.identity["scope_id"], signals=[],
                      started_unix=time.time(), state=value["State"])
        stops[cid] = record
        if value["State"].get("Running") is True:
            result = self._container_control(cid, "TERM", check=False)
            record["signals"].append(dict(signal="SIGTERM", unix=time.time(),
                                         monotonic=time.monotonic(), returncode=result.returncode,
                                         output=result.stdout))
            if result.returncode and self.inspect_owned(cid)["State"].get("Running"):
                raise Refused("exact owned container SIGTERM failed")
        atomic_json(self.rdv / f"container-termination-rank{self.rank}.json", stops)
        return record

    def _stop_server(self, cid):
        record = self._term_server(cid)
        term = next((event for event in record["signals"] if event["signal"] == "SIGTERM"), None)
        stop_at = min((term["monotonic"] if term else time.monotonic()) +
                      MEMORY_POLICY["term_grace_seconds"], self.envelope.end)
        while True:
            # Inspect/control retain a bounded cleanup-only budget after expiry.
            value = self.inspect_owned(cid)
            if value["State"].get("Running") is not True:
                break
            # An expired work envelope must not suppress this exact-owned KILL.
            if time.monotonic() >= stop_at:
                result = self._container_control(cid, "KILL")
                record["signals"].append(dict(signal="SIGKILL", unix=time.time(),
                                             monotonic=time.monotonic(), returncode=result.returncode))
                value = self.inspect_owned(cid)
                if value["State"].get("Running") is True:
                    raise Refused("exact owned container remains running after SIGKILL")
                break
            left = stop_at - time.monotonic()
            time.sleep(min(.2, max(0, left)))
        record.update(state=value["State"], ended_unix=time.time(),
                      deadline_shortened_grace=stop_at == self.envelope.end)
        atomic_json(self.rdv / f"container-termination-rank{self.rank}.json", self.server_terminations)
        return record

    def start(self, arm):
        # Source checks and the peer barrier may outlive the successful preflight.
        available = available_gib()
        atomic_json(self.rdv / f"{arm['arm']}-launch-headroom-rank{self.rank}.json",
                    dict(unix=time.time(), rank=self.rank, mem_available_gib=available,
                         threshold_gib=MEMORY_POLICY["start_gib"]))
        if available < MEMORY_POLICY["start_gib"]:
            raise Refused("local MemAvailable below 107 GiB preflight")
        out = self.work / arm["arm"]
        out.mkdir()
        out.chmod(0o777)
        cidfile = self.work / (arm["arm"] + ".cid")
        argv = recipe.container(self.config, arm, self.identity, out, self.ext, cidfile, self.image_env)
        self.active = dict(arm=arm["arm"], out=str(out), cidfile=str(cidfile),
                           name=argv[argv.index("--name") + 1], launch_argv=argv)
        atomic_json(self.work / (arm["arm"] + ".launch.json"), self.active)
        result = self.command(argv, tick=self.tick)
        cid = cidfile.read_text().strip()
        if not re.fullmatch("[a-f0-9]{64}", cid) or result.stdout.strip() != cid:
            raise Refused("Docker launch did not return its exact cidfile identity")
        self.active["cid"] = cid
        value = self.inspect_owned(cid)
        self.active["pid"] = value["State"]["Pid"]
        self.containers.append(dict(self.active))
        atomic_json(self.work / (arm["arm"] + ".launch.json"), self.active)
        return dict(cid=cid, scope_id=self.identity["scope_id"], pid=value["State"]["Pid"])

    def inspect_owned(self, cid):
        value = json.loads(self._container_control(cid, "inspect").stdout)[0]
        labels = value.get("Config", {}).get("Labels", {})
        if (value.get("Id") != cid or labels.get("prismabuild.scope") != self.identity["scope_id"]
                or labels.get("prismabuild.action") != self.identity["container_owner"]
                or labels.get("org.prismaquant.graph-window") != self.identity["run_id"]
                or labels.get("org.prismaquant.attempt") != self.identity["nonce"]
                or value.get("HostConfig", {}).get("CgroupParent") != self.identity["scope_id"]):
            raise Refused("container identity/labels/parent do not belong to this exact PB attempt")
        return value

    def ready(self, arm):
        end = min(time.monotonic() + 1800, time.monotonic() + self.envelope.remaining())
        while time.monotonic() < end:
            self.tick()
            ready = self.command(["curl", "-sf", "-m", "3", "http://10.100.96.2:8142/v1/models"],
                                 check=False, tick=self.tick, limit=4)
            if ready.returncode == 0:
                logs = self.command(["docker", "logs", self.active["cid"]], tick=self.tick).stdout
                observed = fabric_from_log(logs)
                if observed != self.config["fabric"]:
                    raise Refused(f"actual local NCCL fabric {observed} differs from requested {self.config['fabric']}")
                # Canonical spelling of the singleton banner the local log actually contained.
                self.fabric_banner = "Using network " + ("Socket" if observed == "socket" else "IB")
                return dict(fabric=observed, banner=self.fabric_banner, image_id=self.image_id)
            time.sleep(min(1, self.envelope.remaining()))
        raise TimeoutError("local serve readiness deadline")

    def probes(self, arm, peer_meta):
        if self.config.get("window_mode") in recipe.BENCHMARK_PAIRS:
            from eager_benchmark import probes
            return probes(self, arm, peer_meta)
        name = arm["arm"]
        out = Path(self.active["out"])
        root = Path(self.config["ts"])
        args = dict(arm=name, eager=arm["eager"], tensor_parallel_size=2,
                    fabric_requested=self.config["fabric"],
                    fabric_observed=f"rank0:{self.fabric_banner};rank1:{peer_meta['banner']}",
                    compilation_json=arm["compilation"], spec_json=arm["spec"],
                    kernel_json='{"enable_flashinfer_autotune":false}', max_model_len=8448,
                    max_num_seqs=4, kv_cache_memory_bytes=2147483648,
                    tessera_env="TESSERA_FUSED_E4M3_MMA=e4m3", long=1,
                    serve_rank0=shlex.join(recipe.serve(self.config, arm, 0)),
                    serve_rank1=shlex.join(recipe.serve(self.config, arm, 1)),
                    image=self.config["image"], image_resolved_reference=self.config["image"],
                    image_id=self.image_id, image_id_peer=peer_meta["image_id"],
                    equal_script_sha256=self.config["equal_script_sha256"],
                    producer_commit=self.config["producer_commit"], producer_sha256=self.config["producer_sha256"],
                    host="sparklina+sparky", tree=self.config["ts"], tree_sha=self.config["source_commit"],
                    src_sha256=self.config["src_sha256"], hooks_sha256=self.config["hooks_sha256"],
                    model=self.config["artifact"], pb_action=self.identity["action_key"],
                    pb_nonce=self.identity["nonce"], pb_scope=self.identity["scope_id"], started=time.time())
        with (out / f"engine-args-{name}.txt").open("w") as stream:
            stream.write("".join(f"{key}={value}\n" for key, value in args.items()))
        env = dict(os.environ, T508_HOST="10.100.96.2", T508_MODEL="glm53-artifact",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        for suffix, cases in (("", None), ("-r2", None), ("-long", recipe.LONG_CASES)):
            command = [sys.executable, str(root / "experiments/glm53_508_graph_qual/equal-508.py"),
                       "8142", str(out), name + suffix]
            child_env = env if cases is None else dict(env, T702_LONG="1", T702_MAX_MODEL_LEN="8448")
            if cases is not None: command.append(cases)
            with (out / (name + suffix + ".eq.txt")).open("w") as log:
                self.command(command, env=child_env, stdout=log, tick=self.tick,
                             limit=self.envelope.remaining())
        # Dispatch totals rewrite once per second; this delay also consumes the same envelope.
        until = time.monotonic() + 3
        while time.monotonic() < until:
            self.envelope.remaining(); self.tick(); time.sleep(.2)
        with (out / (name + ".metrics.txt")).open("w") as stream:
            self.command(["curl", "-sf", "-m", "10", "http://10.100.96.2:8142/metrics"],
                         stdout=stream, tick=self.tick, limit=12)
        with (out / f"engine-args-{name}.txt").open("a") as stream:
            stream.write("rc=0\n")

    def verify_profiles(self, arm, probe):
        command = [sys.executable, str(Path(self.config["client_source"]) / "comparison_inputs.py"),
                   "--manifest", self.config["profile_manifest"], "verify-profile",
                   "--rank", str(self.rank), "--directory", probe["profile_dir"],
                   "--events", probe["events"], "--invocation", probe["invocation"]]
        proof = json.loads(self.command(command, tick=self.tick, limit=self.envelope.remaining()).stdout)
        if proof["rank"] != self.rank or proof["invocation"] != probe["invocation"]:
            raise Refused("Window4 profile verifier returned a different rank/invocation")
        atomic_json(self.rdv / "arms" / arm["arm"] / f"profile-verified-rank{self.rank}.json", proof)
        return proof

    def assert_empty(self):
        # Scope AND owner, never a campaign/name-only drain. Foreign work is untouched.
        containers = self.command(["docker", "ps", "-aq", "--filter",
                    f"label=prismabuild.scope={self.identity['scope_id']}", "--filter",
                    f"label=prismabuild.action={self.identity['container_owner']}"], cleanup=True, limit=10).stdout.strip()
        if containers:
            raise Refused(f"owned scope still contains containers: {containers}")
        pids = self.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            cleanup=True, limit=10).stdout.splitlines()
        for pid in pids:
            pid = pid.strip()
            if not pid.isdigit(): raise Refused("GPU descendant inventory unreadable")
            try:
                groups = Path(f"/proc/{pid}/cgroup").read_text()
            except FileNotFoundError:
                continue  # it ceased to exist between inventory and ancestry read
            if self.identity["scope_id"] in groups:
                raise Refused(f"owned GPU descendant {pid} remains")
        return dict(containers_empty=True, gpu_descendants_empty=True, scope_id=self.identity["scope_id"])

    def cleanup(self, arm):
        errors = []
        active = self.active
        if active:
            cidfile = Path(active["cidfile"])
            cid = active.get("cid") or (cidfile.read_text().strip() if cidfile.exists() else None)
            try:
                # A timed-out CLI may have created its named container before writing the cidfile.
                if cid is None:
                    found = self.command(["docker", "ps", "-aq", "--no-trunc", "--filter",
                                          f"name=^{active['name']}$"], cleanup=True, limit=10).stdout.strip()
                    if found:
                        if not re.fullmatch("[a-f0-9]{64}", found): raise Refused("ambiguous launch identity")
                        cid = found
                if cid:
                    self._stop_server(cid)
                    logname = f"{arm['arm']}.rank1.engine.log" if self.rank else f"{arm['arm']}.engine.log"
                    try:
                        with (Path(active["out"]) / logname).open("w") as stream:
                            self.command(["docker", "logs", cid], stdout=stream, cleanup=True, limit=15)
                    except Exception as exc:
                        errors.append(f"log capture: {exc}")
                    self.command(["docker", "rm", cid], cleanup=True, limit=20)
            except Exception as exc:
                errors.append(f"exact container cleanup: {exc}")
            try:
                dest = self.rdv / "arms" / arm["arm"]
                dest.mkdir(parents=True, exist_ok=True)
                copied = []
                for path in Path(active["out"]).iterdir():
                    if not path.is_file(): raise Refused(f"unexpected output directory: {path}")
                    if (dest / path.name).exists(): raise Refused(f"refuse output overwrite: {path.name}")
                    self.command(["cp", "--", str(path), str(dest / path.name)], cleanup=True, limit=30)
                    copied.append(dest / path.name)
                self.command(["cp", "--", str(self.work / "memwatch.txt"),
                              str(dest / f"{arm['arm']}.rank{self.rank}.memwatch.txt")], cleanup=True, limit=10)
                atomic_json(dest / f"ownership-rank{self.rank}.json", dict(self.identity, container=active))
                if self.config.get("window_mode") in recipe.BENCHMARK_PAIRS:
                    target = dest / f"{arm['arm']}.rank{self.rank}.memory-scope.jsonl"
                    self.command(["cp", "--", str(self.work / "memory-scope.jsonl"), str(target)], cleanup=True, limit=10)
                    copied.append(target)
                copied += [dest / f"{arm['arm']}.rank{self.rank}.memwatch.txt", dest / f"ownership-rank{self.rank}.json"]
                with (dest / f"{arm['arm']}.rank{self.rank}.SHA256SUMS").open("w") as sums:
                    self.command(["sha256sum", "--", *(str(path) for path in copied)],
                                 stdout=sums, cleanup=True, limit=30)
            except Exception as exc:
                errors.append(f"partial evidence copy: {exc}")
        self.active = None
        try:
            physical = self.assert_empty()
        except Exception as exc:
            physical = dict(containers_empty=False, gpu_descendants_empty=False)
            errors.append(f"physical local cleanup: {exc}")
        if errors:
            raise Refused("; ".join(errors))
        return physical


def run_rank(config, owned, queue, rdv, arms, adapter, envelope, *, poll_seconds=.2):
    """The real finite protocol; CPU scenarios substitute only the LOCAL device adapter."""
    meeting = Rendezvous(rdv, owned, queue, envelope, poll_seconds=poll_seconds)
    adapter.guard = meeting.check
    adapter.abort = lambda **fields: meeting.publish("failed", **fields)
    adapter.poll_seconds = poll_seconds
    outcome = dict(owned, simulation=not isinstance(adapter, LocalArm), completed_arms=[],
                   ownership_released=False, local_cleanup=[], returncode=1)
    current = None
    probe_rank = 1 if config.get("window_mode") in recipe.BENCHMARK_PAIRS else 0
    try:
        meeting.bind_peer(tick=adapter.tick)
        owned["window_end_unix"] = envelope.end_unix
        atomic_json(meeting.path, owned)
        if isinstance(adapter, LocalArm):
            event = dict(event="both_halves_claimed", identities=[owned, meeting.peer],
                         requested_pb_timeout_s=5400, effective_pb_timeout_s=None, peer_wait_seconds=3600,
                         runtime_commit=config["source_commit"], producer_commit=config["producer_commit"])
            atomic_json(rdv / f"both-claimed-rank{owned['rank']}.json", event)
            print(json.dumps(event, sort_keys=True), flush=True)
        for arm in arms:
            current = arm
            meeting.check()
            envelope.remaining()
            metadata = adapter.preflight()
            stage = arm["arm"]
            meeting.publish(stage + "-preflight", metadata=metadata)
            peer = meeting.wait(stage + "-preflight", tick=adapter.tick)
            # Agreement between live gang halves is comparability, not a recorded-run seal.
            for key in ("image", "src_sha256", "config_sha256"):
                if metadata.get(key) != peer["metadata"].get(key):
                    raise Refused(f"rank preflight differs in {key}")
            meeting.check()  # live peer authority immediately before every local launch
            envelope.remaining()
            started = adapter.start(arm)
            meeting.publish(stage + "-started", container=started)
            meeting.wait(stage + "-started", tick=adapter.tick)
            ready = adapter.ready(arm)
            if ready["fabric"] != config["fabric"]:
                raise Refused("local observed fabric differs from frozen tuple")
            meeting.publish(stage + "-ready", **ready)
            peer_ready = meeting.wait(stage + "-ready", tick=adapter.tick)
            if peer_ready["fabric"] != config["fabric"]:
                raise Refused("peer observed fabric differs from frozen tuple")
            if owned["rank"] == probe_rank:
                finished = adapter.probes(arm, peer_ready) or {}
                meeting.publish(stage + "-probes", returncode=0, **finished)
            else:
                while not (rdv / f"{stage}-probes-rank{probe_rank}.json").exists():
                    envelope.remaining(); meeting.check(); adapter.tick(); time.sleep(poll_seconds)
                finished = meeting.checked(rdv / f"{stage}-probes-rank{probe_rank}.json", probe_rank)
                if finished["returncode"] != 0: raise Refused("head probe failed")
            if config.get("window_mode") in recipe.BENCHMARK_PAIRS and isinstance(adapter, LocalArm):
                profile = adapter.verify_profiles(arm, finished)
                meeting.publish(stage + "-verified", profile=profile)
                meeting.wait(stage + "-verified", tick=adapter.tick)
            current = None
            try:
                physical = adapter.cleanup(arm)
            except BaseException as exc:
                outcome["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                raise
            outcome["local_cleanup"].append(dict(arm=stage, **physical))
            meeting.publish(stage + "-cleaned", **physical)
            # Final cleanup is evidence, not fresh admission. Check exact failed
            # markers before accepting the peer final-cleaned record, even if it exited.
            if arm != arms[-1]:
                meeting.wait(stage + "-cleaned", tick=adapter.tick)
            else:
                peer_cleaned = rdv / f"{stage}-cleaned-rank{1-owned['rank']}.json"
                while True:
                    envelope.remaining()
                    for rank in (0, 1):
                        failure = rdv / f"failed-rank{rank}.json"
                        if failure.exists():
                            value = meeting.checked(failure, rank)
                            raise Refused(f"rank {rank} failed: {value.get('error')}")
                    if peer_cleaned.exists():
                        meeting.checked(peer_cleaned, 1-owned["rank"])
                        break
                    adapter.tick()
                    time.sleep(min(poll_seconds, envelope.remaining()))
            outcome["completed_arms"].append(stage)
        outcome["returncode"] = 0
    except BaseException as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        outcome["returncode"] = 124 if isinstance(exc, TimeoutError) else 1
        try:
            meeting.publish("failed", error=outcome["error"])
        except BaseException as secondary:
            outcome["failure_publication_error"] = f"{type(secondary).__name__}: {secondary}"
    finally:
        if outcome["returncode"] != 0:
            envelope.tighten(time.time() + CLEANUP_SECONDS)
        if current is None and not outcome["local_cleanup"] and "cleanup_error" not in outcome:
            current = dict(arm="unstarted")
        if current is not None:
            try:
                physical = adapter.cleanup(current)
                outcome["local_cleanup"].append(dict(arm=current["arm"], **physical))
            except BaseException as exc:
                outcome["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                outcome["returncode"] = 1
        # Do not let native gang teardown shorten the peer server's TERM grace:
        # a failed rank returns only after BOTH payloads acknowledge owned cleanup.
        if outcome["returncode"] != 0 and meeting.peer is not None:
            try:
                meeting.publish("failed-cleaned", local_cleanup=outcome["local_cleanup"],
                                cleanup_error=outcome.get("cleanup_error"))
                peer_cleaned = rdv / f"failed-cleaned-rank{1-owned['rank']}.json"
                while not peer_cleaned.exists():
                    try:
                        require_claim(meeting.peer, queue)
                    except BaseException as peer_ended:
                        outcome["peer_cleanup_acknowledgement"] = dict(available=False,
                            reason=f"peer claim ended: {type(peer_ended).__name__}: {peer_ended}")
                        break
                    left = envelope.remaining(cleanup=True) - 5  # preserve broker-stop margin
                    if left <= 0:
                        raise TimeoutError("peer cleanup acknowledgement deadline; broker-stop margin begins")
                    time.sleep(min(poll_seconds, left))
                if peer_cleaned.exists():
                    meeting.checked(peer_cleaned, 1-owned["rank"])
                    outcome["peer_cleanup_acknowledgement"] = dict(available=True)
            except BaseException as exc:
                outcome["cleanup_error"] = f"failed peer cleanup acknowledgement: {type(exc).__name__}: {exc}"
        outcome["window_end_unix"] = envelope.end_unix
        outcome["process_terminations"] = getattr(envelope, "terminations", [])
        outcome["container_terminations"] = getattr(adapter, "server_terminations", {})
        outcome["memory_summary"] = getattr(adapter, "memory_summary", None)
        outcome["abort_errors"] = getattr(adapter, "abort_errors", [])
        outcome["ended_unix"] = time.time()
        if outcome["ended_unix"] > envelope.end_unix:
            outcome["cleanup_error"] = "payload cleanup outside whole-window envelope"
            outcome["returncode"] = 1
        outcome["invocation_failed"] = outcome["returncode"] != 0
        atomic_json(rdv / f"outcome-rank{owned['rank']}.json", outcome)
        print(json.dumps(outcome, sort_keys=True), flush=True)
    return outcome["returncode"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, choices=(0, 1), required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--role-preflight", action="store_true", help="CPU-only real role/source/image/shell check; start no container")
    args = ap.parse_args()
    # Public identity is stamped by the broker; a truthy action-key alone is not admission.
    key = os.environ.get("PRISMABUILD_ACTION_KEY", "")
    nonce = os.environ.get("PRISMABUILD_ACTION_NONCE", "")
    scope = os.environ.get("PRISMABUILD_ACTION_SCOPE", "")
    if not key or not nonce or not scope:
        raise Refused("rank_window requires an admitted PB action/attempt/scope")
    setup = read_json(args.run)
    if recipe.sha(args.run) != os.environ.get("GRAPH_WINDOW_INPUT_SHA256"):
        raise Refused("window inputs differ from sealed action environment")
    policy_path = args.run.parent / "memory-policy.json"
    if recipe.sha(policy_path) != setup["memory_policy_sha256"]:
        raise Refused("rank memory policy differs from its own recorded digest")
    check_memory_policy(read_json(policy_path), where="Window4 rank")
    if socket.gethostname() != HOSTS[args.rank]:
        raise Refused("rank claimed on the wrong physical host")
    queue = Path(os.environ["PRISMABUILD_QUEUE_ROOT"])
    row = read_json(queue / "claimed" / (key + ".json"))
    if not args.role_preflight and not row.get("gpu_admission"):
        raise Refused("model rank has no admitted GPU evidence; a CPU action cannot launch it")
    owned = dict(rank=args.rank, action_key=key, nonce=nonce, scope_id=scope, host=HOSTS[args.rank],
                 container_owner=os.environ["PRISMABUILD_CONTAINER_OWNER"], claimed_unix=row["claimed_unix"],
                 run_id=setup["run_id"], input_sha256=recipe.sha(args.run),
                 window_end_unix=row["claimed_unix"] + WINDOW_SECONDS)
    require_claim(owned, queue)
    envelope = Envelope(owned["window_end_unix"])
    current = recipe.inputs(os.environ, live=True, runner=envelope.run)
    recipe.check_control_record(setup["config"], current, where="Window4 rank",
                              refusal=Refused("frozen source/control differs from prepared window inputs"))
    config = setup["config"]  # stored data, never an identity-triggered regeneration
    rdv = args.run.parent
    if args.role_preflight:
        image = json.loads(envelope.run([sys.executable, "-m", "tessera.serving.runtime_image", "resolve",
                                       "--image", config["image"]],
                                      env=dict(os.environ, PYTHONPATH=str(Path(config["ts"]) / "src"))).stdout)
        seal_check("role image identity", config["image"], image.get("resolved_reference"),
                   where="Window4 role", refusal=Refused("role preflight image differs from frozen image"))
        for arm in setup["arms"]:
            command = recipe.container(config, arm, owned, rdv / "unused-out", rdv / "unused-ext",
                                       rdv / "unused-cid", {})
            envelope.run(["bash", "-n", "-c", command[-1]])
        sampler = None
        parsers = []
        if config.get("window_mode") in recipe.BENCHMARK_PAIRS:
            sampler = scope_memory(owned)
            for arm in setup["arms"]:
                local_config = dict(config, profile_dir=str(Path(config["profile_dir"]) / arm["arm"]))
                serve = recipe.serve(local_config, arm, args.rank)
                document = serve[serve.index("--profiler-config") + 1]
                parser = ["docker", "run", "--rm", "--network", "none",
                          "--name", f"window4-cpu-parser-{owned['nonce']}-{arm['arm']}",
                          "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
                          "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
                          "-e", "OPENBLAS_NUM_THREADS=1", "--entrypoint", "python3", config["image"], "-c",
                          "import json,sys; from pydantic import TypeAdapter; "
                          "from vllm.config import ProfilerConfig; "
                          "c=TypeAdapter(ProfilerConfig).validate_json(sys.argv[1]); "
                          "print(json.dumps(dict(profiler=c.profiler,torch_profiler_dir=c.torch_profiler_dir,"
                          "torch_profiler_with_stack=c.torch_profiler_with_stack,"
                          "torch_profiler_record_shapes=c.torch_profiler_record_shapes,ignore_frontend=c.ignore_frontend)))",
                          document]
                actual = json.loads(envelope.run(parser, limit=30).stdout.strip().splitlines()[-1])
                expected = json.loads(document)
                if any(actual.get(key) != value for key, value in expected.items()):
                    raise Refused("pinned image parsed a different profiler configuration")
                parsers.append(dict(arm=arm["arm"], actual=actual, command=parser))
            present = envelope.run(["docker", "ps", "-aq", "--filter", f"label=prismabuild.scope={owned['scope_id']}",
                                    "--filter", f"label=prismabuild.action={owned['container_owner']}"]).stdout.strip()
            if present:
                raise Refused("CPU profiler-parser container remains in the owned scope")
        proof = dict(owned, config=config, image=image, native_gpu_work=False, model_containers_started=0,
                     cpu_parser_containers_started=len(parsers), profiler_parsers=parsers, real_cgroup_sample=sampler,
                     requested_pb_timeout_s=120, effective_pb_timeout_s=None,
                     model_window_seconds=WINDOW_SECONDS, peer_wait_seconds=3600,
                     rendered_arms=[a["arm"] for a in setup["arms"]])
        atomic_json(rdv / f"cpu-role-preflight-rank{args.rank}.json", proof)
        print(json.dumps(proof, sort_keys=True), flush=True)
        return 0
    local = LocalArm(config, owned, envelope, rdv)
    def interrupted(signum, frame):
        raise Refused(f"rank interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return run_rank(config, owned, queue, rdv, setup["arms"], local, envelope)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as exc:
        print(f"rank refused: {exc}", file=sys.stderr)
        raise SystemExit(3)
