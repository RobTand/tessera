"""Finite #793 observer around the existing routed adapter, not a decoder/launcher."""
import hashlib
from pathlib import Path

GROUPS = 'experts.R1024.L10,experts.R1088.L11,experts.R832.L42'
MODULES = tuple(f'model.language_model.layers.{layer}.mlp.experts' for layer in (10, 11, 42))


def require_options(args, *, stubbed):
    if (args.artifact != '/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported'
            or args.groups != GROUPS or args.ms != '1,2048' or not args.no_graph
            or args.ncu or not args.hash_only or not args.input_manifest
            or not args.profile_native_file or args.single_routing_file or args.routing
            or (args.warmup, args.iters, args.power_s) != (0, 0, 0)
            or stubbed):
        raise ValueError('stageprev numeric qualification requires the exact original three-group '
                         'TP2rank0 population, staged inputs/native, no graph/timing/stub overrides')


def record(tensor):
    import torch
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return raw, {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                 'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}


def observe(fn, xa, directory):
    """Reuse the paired qualification's exact-class/exact-instance observation pattern."""
    import torch
    from tessera.routed_fused import FusedRoutedWindowMoE
    native = fn.native_adapter
    if type(native) is not FusedRoutedWindowMoE or native.library != 'e4m3mma':
        raise ValueError('stageprev numeric owner is not the production MMA8 routed adapter')
    if native.uniform is None:
        raise ValueError('stageprev observes the uniform launch; this owner has several classes')
    if len(xa) != 3:
        raise ValueError('stageprev requires exactly x, ids and weights')
    owner = type(native.uniform)
    original = owner.launch
    captured = {}

    def capture(kernel, mode, *args, **kwargs):
        original(kernel, mode, *args, **kwargs)
        if kernel is not native.uniform:
            return
        if mode not in (0, 2) or mode in captured:
            raise ValueError('unexpected or duplicate native stage')
        captured[mode] = kwargs['out'].clone()

    owner.launch = capture
    try:
        first = fn(*xa)
        torch.cuda.synchronize()
        stages = captured.copy()
        captured.clear()
        second = fn(*xa)
        torch.cuda.synchronize()
        if set(stages) != {0, 2} or set(captured) != {0, 2}:
            raise ValueError('native role population differs')
        if (not torch.equal(first.view(torch.uint8), second.view(torch.uint8))
                or any(not torch.equal(stages[m].view(torch.uint8), captured[m].view(torch.uint8))
                       for m in (0, 2))):
            raise ValueError('repeated intermediate/final words differ')
    finally:
        owner.launch = original
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for role, tensor in [('gate_up', stages[0]), ('down_routes', stages[2]), ('out', first)]:
        raw, entry = record(tensor)
        path = directory / (role + '.bin')
        with path.open('xb') as handle:
            handle.write(raw)
        outputs[role] = dict(entry, path=str(path))
    runs = {}
    for role, projection in [('gate_proj', native.uniform.gate), ('up_proj', native.uniform.up),
                             ('down_proj', native.uniform.down)]:
        runs[role] = projection.runs.cpu().tolist()
    inputs = {name: record(tensor)[1] for name, tensor in zip(("x", "ids", "weights"), xa)}
    return {'outputs': outputs, 'repeat_equal': True, 'run_tables': runs, 'inputs': inputs,
            'native_extra_resident_bytes': int(native.resident_bytes()),
            'scope': 'original selected mixed operator words only; not all-rung/served quality'}
