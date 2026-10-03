"""Finite original-token_sum binding diagnostics (#859); no serving/timing claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BANK = Path('/mnt/shared/astra-resume-20261002/t8_performance/small-perf-common-d620660e4702')
SOURCE = 'd620660e4702b7798b0a695bdbde4c82993582bcc7469de8e304734c77257618'
LIBS = {
    'value': ('tessera_routed_fused_value', 'f23bdcdf31e8fb9ddacdeb747735b03f407ea8623065bc73c5c9047e33f1b8eb', 0, 0, 0),
    'e4m3': ('tessera_routed_fused_e4m3', '7f10ab3948b9c360589fe9be051f1db6d9780c4169c818a9d248bb87a9547580', 1, 0, 0),
    'e4m3mma': ('tessera_routed_fused_mma_e4m3', '6abdcac9505614b15ee78ddb0add804f3eaff260c85d262e9eb0bb7b83845e8d', 1, 1, 0),
    'e2m1': ('tessera_routed_fused_e2m1', '15098c6e73cf95ac9b8be735f65467e26bec46cd3becb83b35d88b052758fb2f', 0, 0, 1),
}
BEFORE = False
SOURCE_DIRECTORY = "frozen-candidate-source"
BASELINE_HASHES = {
    "value": "97c5f4ca8a722721c114f8bca43eb04e755b9eacfafa3c6689411fdb229abbef",
    "e4m3": "dc2bc9e08752c013319c1c638798a8ea019e2fa7a830e523b7e380512a52099e",
    "e4m3mma": "d68633d236bab421e574f6b5fc73bbcf3bf0a87141ad833690d4dba308f223c1",
    "e2m1": "597cab7e707df53c77cf0e8406093ac14410964b03eecdf9565890ecc7de402b",
}

def binding(path):
    raw = path.read_bytes()
    return dict(path=str(path), offset=0, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def prepare(out):
    source_root = BANK / SOURCE_DIRECTORY
    hashes = json.loads((source_root / 'SOURCE-SHA256.json').read_bytes())
    entries = []
    for name, expected in hashes.items():
        row = binding(source_root / 'src' / name)
        if row['sha256'] != expected:
            raise ValueError('frozen package source differs: ' + name)
        entries.append(row)
    entries.append(binding(source_root / 'pyproject.toml'))
    for library, (module, expected, fp8, mma8, fp4) in LIBS.items():
        directory = BANK / (module + '_sm_121_tessera_guarded_v1')
        elf = binding(directory / (module + '.so'))
        if elf['sha256'] != expected:
            raise ValueError('retained ELF differs: ' + library)
        entries.append(elf)
        for name in ('build.ninja', 'native-finalization.json'):
            entries.append(binding(directory / name))
    manifest = dict(schema='prismaquant.prismabuild.data_manifest.v1',
                    produced_by=dict(tool='TS859 existing-bank qualification', action_key=os.environ['PRISMABUILD_ACTION_KEY']),
                    annotations=dict(row_id="token-sum-859-original-binding", phases=[dict(name="four-native-libraries", bytes=sum(e["bytes"] for e in entries), cumulative_bytes=sum(e["bytes"] for e in entries))]),
                    mount_prefix='/mnt/shared', entries=entries, entry_count=len(entries),
                    total_bytes=sum(e['bytes'] for e in entries))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(readset=str(out), sha256=binding(out)['sha256'], entry_count=len(entries), total_bytes=manifest['total_bytes'])))

def refusal(lib, routed, out, message):
    try:
        lib.token_sum(routed, out, 8)
    except RuntimeError as exc:
        if message not in str(exc):
            raise
        return str(exc).split('\n')[0]
    raise AssertionError('original token_sum accepted invalid input')

def consume(manifest, out, gpu):
    import torch
    from experiments.t8r_speed.pb_staged_store import NativeCallback, StagedInputs
    out.mkdir(parents=True, exist_ok=False)
    reader = StagedInputs(manifest)
    owners, live_fds, results = [], [], []
    try:
        local_source = out / 'source'
        for path, offset in reader.entries:
            original = Path(path)
            source_root = BANK / SOURCE_DIRECTORY
            if original.is_relative_to(source_root):
                target = local_source / original.relative_to(source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(reader.read(path, offset))
        kernel = local_source / 'src/tessera/serving/csrc/routed_fused_window.cu'
        if binding(kernel)['sha256'] != SOURCE:
            raise ValueError('wrong original binding source')
        body = kernel.read_text().split('void token_sum(', 1)[1].split('void token_sum_shared(', 1)[0]
        if BEFORE:
            if "out.is_cuda()" in body or "out.device() == routed.device()" in body:
                raise ValueError("causal-before original binding already has a device guard")
        elif not (body.index("out.is_cuda() && out.device() == routed.device()") < body.index("if (vecs == 0) return;") < body.index("token_sum_kernel<<<")):
            raise ValueError("output device gate does not precede return/launch")
        sys.path.insert(0, str(local_source / 'src'))
        from tessera import routed_fused as rf
        from tessera.serving.backend import PlatformMismatchError
        if gpu:
            if not torch.cuda.is_available():
                raise ValueError('actual original-token_sum controls require CUDA')
            device = torch.device('cuda', torch.cuda.current_device())
        else:
            if torch.cuda.is_available() or torch.cuda.is_initialized():
                raise ValueError('CPU diagnostic unexpectedly exposes CUDA')
            os.environ['TESSERA_PLATFORM_TOKEN'] = 'sm_121'
        for library, (module, expected, fp8, mma8, fp4) in LIBS.items():
            directory = BANK / (module + '_sm_121_tessera_guarded_v1')
            flags = reader.read(directory / 'build.ninja').decode()
            finalization = json.loads(reader.read(directory / 'native-finalization.json'))
            for name in ('post_cflags', 'cuda_post_cflags'):
                if next(line for line in flags.splitlines() if line.startswith(name + ' =')).split('=', 1)[1].strip():
                    raise ValueError('unexpected native postflags')
            for name, expected_flag in [('FP8', fp8), ('MMA8', mma8)]:
                if f'-DTESSERA_ROUTED_FUSED_{name}={expected_flag}' not in flags:
                    raise ValueError('native family flags differ')
            if fp4 and '-DTESSERA_ROUTED_FUSED_FP4=1' not in flags:
                raise ValueError('FP4 native flag absent')
            owner = NativeCallback(reader, directory / (module + '.so'), rf, out / library,
                                   expected_sha256=expected, source_sha256=SOURCE,
                                   module=module, source_module=module)
            try:
                def no_compile(*args):
                    raise AssertionError('qualification must not rebuild retained native code')
                if gpu:
                    lib = rf.build_library(module, module, no_compile)
                    platform_refusal = None
                else:
                    # The real build owner loads the retained compile-gate artifact,
                    # then MUST refuse serving handoff on the absent platform.
                    try:
                        rf.build_library(module, module, no_compile)
                    except PlatformMismatchError as exc:
                        platform_refusal = str(exc)
                    else:
                        raise AssertionError('CPU loader accepted a CUDA serving platform')
                    lib = owner.module
                    if lib is None:
                        raise AssertionError('build owner refused before native diagnostic load')
                owner.attest_mapped(lib)
                cpu = torch.empty((0, 16), dtype=torch.bfloat16)
                checks = {'cpu_routed_refusal': refusal(lib, cpu, cpu, 'routed must be')}
                if gpu:
                    routed = torch.empty((1, 16), dtype=torch.bfloat16, device=device)[:0]
                    if BEFORE:
                        # Safe historical acceptance: zero vectors return before
                        # any kernel receives the CPU output pointer.
                        lib.token_sum(routed, cpu, 8)
                        torch.cuda.synchronize(device)
                        checks["zero_cpu_output_accepted_before"] = True
                        results.append(dict(library=library, checks=checks, platform_refusal=platform_refusal,
                                            native_finalization=finalization, build_ninja_sha256=hashlib.sha256(flags.encode()).hexdigest()))
                        continue
                    checks['zero_cpu_output_refusal'] = refusal(lib, routed, cpu, "on routed's CUDA device")
                    good = torch.empty((0, 16), dtype=torch.bfloat16, device=device)
                    lib.token_sum(routed, good, 8)
                    for name, bad in [('rank', good.reshape(-1)), ('dtype', good.float()), ('width', torch.empty((0, 8), dtype=torch.bfloat16, device=device))]:
                        checks[name + '_refusal'] = refusal(lib, routed, bad, 'out must be')
                    # Small exact powers of two give an independent FP32 fixed-order
                    # reference without tolerance or shared-fold arithmetic.
                    data = (torch.arange(8 * 16, device=device).reshape(8, 16) % 8 - 4).bfloat16()
                    output = torch.empty((1, 16), dtype=torch.bfloat16, device=device)
                    for name, bad in [("out_contiguity", torch.empty((1, 32), dtype=torch.bfloat16, device=device)[:, ::2]),
                                      ("out_rows", torch.empty((2, 16), dtype=torch.bfloat16, device=device))]:
                        checks[name + "_refusal"] = refusal(lib, data, bad, "out must be")
                    for name, bad in [("routed_rank", data.reshape(-1)), ("routed_dtype", data.float()),
                                      ("routed_contiguity", torch.empty((8, 32), dtype=torch.bfloat16, device=device)[:, ::2])]:
                        checks[name + "_refusal"] = refusal(lib, bad, output, "routed must be")
                    short = torch.empty((0, 12), dtype=torch.bfloat16, device=device)
                    checks["width_quantum_refusal"] = refusal(lib, short, short, "H must be a multiple of 8")
                    lib.token_sum(data, output, 8)
                    torch.cuda.synchronize(device)
                    expected_output = data.float().sum(0, keepdim=True).bfloat16()
                    if not torch.equal(output.view(torch.int16), expected_output.view(torch.int16)):
                        raise AssertionError('same-device original-token_sum bits differ')
                    checks['same_device_nonempty_bits'] = True
                    checks['actual_device'] = str(device)
                    checks['visible_device_count'] = torch.cuda.device_count()
                    checks['different_device_refusal'] = 'NOT_EXERCISED: single-device matrix; no inferred CUDA index'
                else:
                    if torch.cuda.is_initialized():
                        raise AssertionError('CPU original refusal initialized CUDA')
                results.append(dict(library=library, checks=checks, platform_refusal=platform_refusal,
                                    native_finalization=finalization, build_ninja_sha256=hashlib.sha256(flags.encode()).hexdigest()))
            finally:
                if owner.module is not None:
                    live_fds.append(owner.fd)
                owner.finish(torch.cuda.synchronize if gpu else lambda: None, keep_load_fd=True)
                owners.append(dict(owner.record))
        if len(results) != 4 or len(set(live_fds)) != 4:
            raise AssertionError('incomplete distinct four-library population')
        report = dict(schema='tessera.token_sum_859_qualification.v1', action_key=os.environ['PRISMABUILD_ACTION_KEY'],
                      arm="before" if BEFORE else "fixed",
                      gpu=gpu, source_sha256=SOURCE, readset_sha256=reader.manifest_sha256,
                      results=results, native_owners=owners, source_reads=reader.reads,
                      serving_qualification=False, performance_qualification=False)
        raw_report = json.dumps(report, indent=2) + "\n"
        retained_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if len(raw_report.encode()) > 1 << 20 or retained_bytes + len(raw_report.encode()) > 32 << 20:
            raise ValueError("finite qualification output bound exceeded")
        (out / "result.json").write_text(raw_report)
    finally:
        try:
            if gpu:
                torch.cuda.synchronize()
        finally:
            for fd in live_fds:
                os.close(fd)
            reader.close()
    print(json.dumps(dict(passed=4, gpu=gpu, report_sha256=binding(out / 'result.json')['sha256'], sdk_released=reader.closed)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'cpu', 'gpu'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument("--before", action="store_true")
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    global BANK, SOURCE, SOURCE_DIRECTORY, BEFORE
    BEFORE = args.before
    if BEFORE:
        if args.mode == "cpu":
            raise ValueError("historical causal-before population is GPU-zero-only")
        BANK = BANK.parent / "terminal-855-shipping-71fe64f2"
        SOURCE = "71fe64f23d304156faf95f30157321989a4195428f6cb073efccb924fc1247f4"
        SOURCE_DIRECTORY = "frozen-common-source"
        for name, fields in LIBS.items():
            LIBS[name] = (fields[0], BASELINE_HASHES[name], *fields[2:])
    if args.mode == 'prepare':
        prepare(args.out)
    else:
        consume(args.manifest, args.out, args.mode == 'gpu')

if __name__ == '__main__':
    main()
