"""Reuse box_power_window for both boxes' already elapsed diagnosis phases."""
import importlib.util
import json
import math
import sys
from pathlib import Path


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
        for result in data['results']:
            for cell in result.get('cells', {}).values():
                if cell.get('power', {}).get('window_unix'):
                    phases['steady-power'] = cell['power']['window_unix']
    # Fixed paired diagnostic phases reuse the same box-side owner; ordinary
    # legacy/single-population paths retain their existing phase names.
    for name in ('A1', 'B1', 'B2', 'A2'):
        arm = root / name / 'arm.json'
        if arm.exists():
            data = json.loads(arm.read_text())
            phases[name] = data['steady']['window_unix']
    report = {'schema':'tessera.routed_gate_netdata.v1', 'phases':{}}
    for phase, (a,z) in phases.items():
        start,end = math.floor(a),math.ceil(z)
        boxes = {}
        for host,address in [('sparky','192.168.1.180'),('sparklina','192.168.1.110')]:
            try:
                # Preserve the owner's explicit queries, update_every and points;
                # never infer fast cadence from an averaged returned series.
                boxes[host] = owner.collect(address,start,end,max(4,(end-start)//10))
                for context,dimension in [('system.cpu_some_pressure','some 10'),
                    ('system.memory_some_pressure','some 10'),('system.memory_full_pressure','full 10')]:
                    response=owner._fetch(address,context,(dimension,),start,end,max(4,(end-start)//10))
                    boxes[host][context]={'query':response['url'],
                        'stats':owner._stats(response['doc'],(dimension,))}
            except Exception as error:
                boxes[host]={'error':f'{type(error).__name__}: {error}'}
        report['phases'][phase]={'window_unix':[a,z], 'query_window':[start,end], 'boxes':boxes}
    # Work/J stays held until attributable sample coverage and instrument
    # agreement have actually been reviewed; wrapper success never promotes it.
    report['energy_status']='HOLD_pending_coverage_and_instrument_agreement'
    (root/'netdata.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
