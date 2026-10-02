"""Private source instrumentation for the #855 CUDA schedule experiment.

This transforms an explicitly supplied frozen source copy, never the serving
checkout. It is not a runtime option. The resulting source hash must be bound
to the experiment's native artifact and is not a shipping binary.
"""
import argparse
import hashlib
import json
from pathlib import Path


def instrument(source: str) -> str:
    if 'terminal_probe_entered' in source:
        raise ValueError('source is already instrumented')
    init = 'unsigned item_idx = 0;    // the descriptor slot is item_idx & 1'
    consumer = 'bar_sync(BAR_FULL0 + stage, THREADS);'
    terminal = 'bar_arrive(BAR_FULL0 + (gc & 1), THREADS);'
    for text, expected in [(init, 2), (consumer, 4), (terminal, 2)]:
        if source.count(text) != expected:
            raise ValueError(f'probe requires reviewed source anchors: {text!r}')
    source = source.replace(init, init + '''
    // PRIVATE #855 PROBE: one announcement counter per consumer warp.
    __shared__ unsigned terminal_probe_entered[8];
    if (tid < 8) terminal_probe_entered[tid] = 0;
    __syncthreads();''')
    source = source.replace(consumer, '''{
                    // Delay BEFORE announcing/entering this FULL phase. An
                    // announcement proves only entry, not barrier completion.
                    const unsigned long long begin = clock64();
                    while (clock64() - begin < 1000000ULL) __nanosleep(64);
                    if (lane == 0)
                        atomicExch(&terminal_probe_entered[(tid - PRODUCER_THREADS) >> 5], gc + 1);
                    bar_sync(BAR_FULL0 + stage, THREADS);
                }''')
    source = source.replace(terminal, '''// Stop all producers before the potentially illegal arrival.
                bar_sync(BAR_PROD, PRODUCER_THREADS);
                if (tid == 0 && gc >= 2) {
                    for (int warp = 0; warp < 8; ++warp) {
                        if (atomicAdd(&terminal_probe_entered[warp], 0U) < gc - 1) {
                            printf("TESSERA_TERMINAL_PHASE_VIOLATION gc=%u warp=%d\\n", gc, warp);
                            asm volatile("trap;");
                        }
                    }
                }
                bar_sync(BAR_PROD, PRODUCER_THREADS);
                bar_arrive(BAR_FULL0 + (gc & 1), THREADS);''')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    if args.destination.exists() or args.source.resolve() == args.destination.resolve():
        parser.error('destination must be a new private source file')
    original = args.source.read_bytes()
    if hashlib.sha256(original).hexdigest() != args.source_sha256:
        parser.error('source digest does not match the frozen input')
    modified = instrument(original.decode()).encode()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open('xb') as handle:
        handle.write(modified)
    print(json.dumps({'schema': 'tessera.terminal_barrier_probe.v1',
                      'source': str(args.source), 'destination': str(args.destination),
                      'source_sha256': hashlib.sha256(original).hexdigest(),
                      'instrumented_sha256': hashlib.sha256(modified).hexdigest(),
                      'delay_cycles': 1000000, 'shipping_binary': False}))


if __name__ == '__main__':
    main()
