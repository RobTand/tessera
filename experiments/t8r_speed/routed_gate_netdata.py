"""Reuse box_power_window for both boxes' already elapsed diagnosis phases."""
import importlib.util
import json
import math
import sys
from pathlib import Path


def retain_query(owner, address, context, dimensions, start, end, points):
    """Keep each diagnostic response/error without losing other host evidence."""
    try:
        response = owner._fetch(address, context, dimensions, start, end, points)
        raw = response.get('raw_doc', response['doc'])
        return {'query': response['url'], 'stats': owner._stats(response['doc'], dimensions),
                'coverage': response.get('coverage'), 'raw_response': raw,
                'update_every_s': raw.get('db', {}).get('update_every'),
                'returned_bucket_s': raw.get('view', {}).get('update_every')}
    except Exception as error:
        return {'error': f'{type(error).__name__}: {error}', **getattr(error, 'evidence', {})}


def main():
    root = Path(sys.argv[1])
    source = Path(__file__).resolve().parents[1] / 'box_power_window.py'
    spec = importlib.util.spec_from_file_location('existing_box_power_owner', source)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    action = [float(t) for t in (root/'action-window.txt').read_text().split()]
    phases = {'action': action}
    timing = root/'timing/bench_t8r.json'
    if timing.exists():
        data = json.loads(timing.read_text())
        for number, result in enumerate(data['results']):
            if result.get('conditioning', {}).get('window_unix'):
                phases[f'conditioning:{number}'] = result['conditioning']['window_unix']
            for key, cell in result.get('cells', {}).items():
                if cell.get('wall_window_unix'):
                    phases[f'events:{number}:{key}'] = cell['wall_window_unix']
                if cell.get('power', {}).get('window_unix'):
                    window = cell['power']['window_unix']
                    # Keep the historical alias and every finite comparison arm.
                    phases['steady-power'] = window
                    phases[f"steady-power:{number}:{key}"] = window
    report = {'schema':'tessera.routed_gate_netdata.v1', 'phases':{}}
    for phase, (a,z) in phases.items():
        start,end = math.floor(a),math.ceil(z)
        boxes = {}
        for host,address in [('sparky','192.168.1.180'),('sparklina','192.168.1.110')]:
            try:
                # Preserve the owner's explicit queries, update_every and points;
                # never infer fast cadence from an averaged returned series.
                boxes[host] = owner.collect(address,start,end,max(4,(end-start)//10))
                for context, dimensions in [('system.cpu_some_pressure', ('some 10',)),
                    ('system.memory_some_pressure', ('some 10',)),
                    ('system.memory_full_pressure', ('full 10',)),
                    ('system.io', ('reads', 'writes')),
                    ('nvidia_smi.gpu_clock_freq', ('sm',)),
                    ('nvidia_smi.gpu_temperature', ('temperature',))]:
                    boxes[host][context] = retain_query(owner, address, context, dimensions,
                                                        start, end, max(4, (end-start)//10))
            except Exception as error:
                boxes[host]={'error':f'{type(error).__name__}: {error}'}
        report['phases'][phase]={'window_unix':[a,z], 'query_window':[start,end], 'boxes':boxes}
    # Work/J stays held until attributable sample coverage and instrument
    # agreement have actually been reviewed; wrapper success never promotes it.
    report['energy_status']='HOLD_pending_coverage_and_instrument_agreement'
    (root/'netdata.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
