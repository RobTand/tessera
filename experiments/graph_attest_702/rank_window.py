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

from managed_window import (WINDOW_SECONDS, Envelope, HOSTS, Refused,
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
        self.image_env = {}
        self.guard = None

    def command(self, argv, **kwargs):
        return self.envelope.run(argv, **kwargs)

    def preflight(self):
        if available_gib() < 114:
            raise Refused("local MemAvailable below unchanged 114 GiB preflight")
        runtime = Path(self.config["ts"]) / "src"
        env = dict(os.environ, PYTHONPATH=str(runtime), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MAX_JOBS="1")
        record = json.loads(self.command([sys.executable, "-m", "tessera.serving.runtime_image",
                                         "resolve", "--image", self.config["image"]], env=env, tick=self.tick).stdout)
        if record.get("resolved_reference") != self.config["image"]:
            raise Refused("local image does not resolve to the frozen full image reference")
        from_source = self.command([sys.executable, "-m", "tessera.serving.runtime_image", "container-env"],
                                   env=env, input_text=json.dumps(record), tick=self.tick).stdout
        self.image_env = dict(line.split("=", 1) for line in from_source.splitlines())
        self.image_id = record["local_id"]
        self.assert_empty()
        # Repeat the frozen small-file/source checks between arms; never restamp them.
        if recipe.inputs(os.environ, live=True,
                         runner=lambda argv, **kw: self.command(argv, tick=self.tick, **kw)) != self.config:
            raise Refused("immutable control/source inputs changed during the window")
        return dict(image=self.config["image"], image_id=record["local_id"],
                    src_sha256=self.config["src_sha256"], config_sha256=self.config["config_sha256"])

    def tick(self):
        if self.guard:
            self.guard()
        if time.monotonic() - self.last_sample < 5:
            return
        self.last_sample = time.monotonic()
        mem = available_gib()
        with (self.work / "memwatch.txt").open("a") as stream:
            stream.write(f"{time.time()} rank{self.rank} MemAvailable_GiB={mem}\n")
        if self.config.get("window_mode") == recipe.EAGER_MODE:
            sample = dict(scope_memory(self.identity), arm=(self.active or {}).get("arm"))
            with (self.work / "memory-scope.jsonl").open("a") as stream:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
        if mem < 16:
            raise Refused("local 16 GiB physical memory floor breached")
        for path, cap in ((self.ext, 6 * (1 << 30)), (self.work, int(6.3 * (1 << 30))),
                          (self.rdv if self.config.get("window_mode") == recipe.EAGER_MODE else self.rdv / "arms",
                           (1 << 30) if self.config.get("window_mode") == recipe.EAGER_MODE else int(.2 * (1 << 30)))):
            if directory_bytes(path) > cap:
                raise Refused(f"declared disk/output cap exceeded: {path}")
        if self.active and self.active.get("cid"):
            inspect = self.command(["docker", "inspect", self.active["cid"]], limit=5)
            value = json.loads(inspect.stdout)[0]
            if value.get("State", {}).get("Running") is not True:
                raise Refused("owned local server exited before probes completed")

    def start(self, arm):
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
        value = json.loads(self.command(["docker", "inspect", cid], cleanup=True, limit=10).stdout)[0]
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
        if self.config.get("window_mode") == recipe.EAGER_MODE:
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
                    self.inspect_owned(cid)
                    logname = f"{arm['arm']}.rank1.engine.log" if self.rank else f"{arm['arm']}.engine.log"
                    try:
                        with (Path(active["out"]) / logname).open("w") as stream:
                            self.command(["docker", "logs", cid], stdout=stream, cleanup=True, limit=15)
                    except Exception as exc:
                        errors.append(f"log capture: {exc}")
                    self.command(["docker", "rm", "-f", cid], cleanup=True, limit=20)
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
                if self.config.get("window_mode") == recipe.EAGER_MODE:
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
    outcome = dict(owned, simulation=not isinstance(adapter, LocalArm), completed_arms=[],
                   ownership_released=False, local_cleanup=[], returncode=1)
    current = None
    probe_rank = 1 if config.get("window_mode") == recipe.EAGER_MODE else 0
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
            if config.get("window_mode") == recipe.EAGER_MODE and isinstance(adapter, LocalArm):
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
            # Cleanup evidence is not admission: allow a terminal peer only after the final arm.
            if arm != arms[-1]:
                meeting.wait(stage + "-cleaned", tick=adapter.tick)
            outcome["completed_arms"].append(stage)
        outcome["returncode"] = 0
    except BaseException as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        outcome["returncode"] = 124 if isinstance(exc, TimeoutError) else 1
        meeting.publish("failed", error=outcome["error"])
    finally:
        if current is None and not outcome["local_cleanup"] and "cleanup_error" not in outcome:
            current = dict(arm="unstarted")
        if current is not None:
            try:
                physical = adapter.cleanup(current)
                outcome["local_cleanup"].append(dict(arm=current["arm"], **physical))
            except BaseException as exc:
                outcome["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                outcome["returncode"] = 1
        outcome["window_end_unix"] = envelope.end_unix
        outcome["ended_unix"] = time.time()
        if outcome["ended_unix"] > envelope.end_unix:
            outcome["cleanup_error"] = "payload cleanup outside whole-window envelope"
            outcome["returncode"] = 1
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
    config = recipe.inputs(os.environ, live=True, runner=envelope.run)
    if config != setup["config"]:
        raise Refused("frozen source/control differs from prepared window inputs")
    rdv = args.run.parent
    if args.role_preflight:
        image = json.loads(envelope.run([sys.executable, "-m", "tessera.serving.runtime_image", "resolve",
                                       "--image", config["image"]],
                                      env=dict(os.environ, PYTHONPATH=str(Path(config["ts"]) / "src"))).stdout)
        if image.get("resolved_reference") != config["image"]:
            raise Refused("role preflight image differs from frozen image")
        for arm in setup["arms"]:
            command = recipe.container(config, arm, owned, rdv / "unused-out", rdv / "unused-ext",
                                       rdv / "unused-cid", {})
            envelope.run(["bash", "-n", "-c", command[-1]])
        sampler = None
        parsers = []
        if config.get("window_mode") == recipe.EAGER_MODE:
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
