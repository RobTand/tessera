"""Execute the CUDA source's terminal branches under a delayed-consumer schedule.

This CPU protocol regression is not a CUDA numerical or termination result.
The PTX named-barrier contract forbids a warp arriving twice before reset:
https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'src/tessera/serving/csrc/routed_fused_window.cu'


def terminal_branches():
    """Extract complete production branches, including any conditional waits."""
    source = SOURCE.read_text()
    marker = 'if (item >= total_items) {'
    branches = []
    start = 0
    while (start := source.find(marker, start)) >= 0:
        opening = source.index('{', start)
        depth = 1
        end = opening + 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        branches.append(source[opening + 1:end - 1])
        start = end
    assert len(branches) == 2, 'review all persistent producer terminal paths'
    return branches


HARNESS = r'''
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
constexpr int BAR_FULL0 = 1, BAR_EMPTY0 = 3, THREADS = 512;
struct Waiting {};
bool full_pending[2] = {}, empty_released[2] = {};
int32_t desc[16];
void __threadfence_block() {}
void bar_sync(int barrier, int count) {
    if (count != THREADS || barrier < BAR_EMPTY0 || barrier > BAR_EMPTY0 + 1)
        throw std::runtime_error("unexpected wait in terminal branch");
    if (!empty_released[barrier - BAR_EMPTY0]) throw Waiting{};
}
void bar_arrive(int barrier, int count) {
    if (count != THREADS || barrier < BAR_FULL0 || barrier > BAR_FULL0 + 1)
        throw std::runtime_error("unexpected arrival in terminal branch");
    int stage = barrier - BAR_FULL0;
    if (full_pending[stage])
        throw std::runtime_error("FULL reused before delayed consumer reset");
    full_pending[stage] = true;
}
// These two bodies are copied verbatim from the CUDA source by the fixture.
@FUNCTIONS@
int main(int argc, char** argv) {
    const int which = std::atoi(argv[1]);
    const unsigned gc = std::atoi(argv[2]);
    const int slot = 0;
    for (int& value : desc) value = 23;
    // Delay consumers before FULL(gc-2). The ordinary producer's final
    // quantum gc-1 waited EMPTY(gc-3), so publishing the final two quanta
    // and claiming exhaustion is a permitted schedule. For gc=0/1 the
    // terminal parity has never been used and must not wait on EMPTY.
    for (unsigned q = (gc >= 2 ? gc - 2 : 0); q < gc; ++q)
        full_pending[q & 1] = true;
    bool blocked = false;
    auto terminal = [&] { if (which == 0) terminal0(gc, slot); else terminal1(gc, slot); };
    try {
        try { terminal(); } catch (Waiting&) { blocked = true; }
        if (gc >= 2) {
            if (!blocked) throw std::runtime_error("terminal did not await reused stage");
            if (desc[slot * 8] != 23)
                throw std::runtime_error("terminal descriptor published before acknowledgment");
            full_pending[(gc - 1) & 1] = false;
            empty_released[(gc - 1) & 1] = true;
            bool still_blocked = false;
            try { terminal(); } catch (Waiting&) { still_blocked = true; }
            if (!still_blocked || desc[slot * 8] != 23)
                throw std::runtime_error("terminal did not await matching EMPTY parity");
            full_pending[(gc - 2) & 1] = false;
            empty_released[(gc - 2) & 1] = true;
            terminal();
        } else if (blocked) {
            throw std::runtime_error("first-use terminal waited on nonexistent chunk");
        }
        if (desc[slot * 8] != -1 || !full_pending[gc & 1])
            throw std::runtime_error("terminal sentinel was not published");
        // The next consumer FULL observes the sentinel and terminates.
        full_pending[gc & 1] = false;
        std::cout << "terminal acknowledged; gc=" << gc << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
'''


@pytest.fixture(scope='module')
def terminal_build(tmp_path_factory):
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    assert compiler, 'a C++ compiler is required for the source-executed protocol regression'
    root = tmp_path_factory.mktemp('terminal-barrier')
    binaries = {}

    def build(family=None, mutation=None):
        key = family, mutation
        if key not in binaries:
            branches = terminal_branches()
            guard = 'if (gc >= 2) bar_sync(BAR_EMPTY0 + (gc & 1), THREADS);'
            if mutation:
                assert guard in branches[family]
                replacement = '' if mutation == 'missing-wait' else 'bar_sync(BAR_EMPTY0 + (gc & 1), THREADS);'
                branches[family] = branches[family].replace(guard, replacement, 1)
            functions = '\n'.join(
                f'void terminal{i}(unsigned gc, int slot) {{ const int tid = 0;\n{body}\n}}'
                for i, body in enumerate(branches))
            source, binary = root / f'terminal{len(binaries)}.cpp', root / f'terminal{len(binaries)}'
            source.write_text(HARNESS.replace('@FUNCTIONS@', functions))
            subprocess.run([compiler, '-std=c++17', '-O0', str(source), '-o', str(binary)],
                           check=True, capture_output=True, text=True)
            binaries[key] = binary
        return binaries[key]

    return build


@pytest.fixture(scope='module')
def terminal_binary(terminal_build):
    return terminal_build()


@pytest.mark.parametrize('family', [0, 1], ids=['value-e4m3', 'fp4'])
@pytest.mark.parametrize('chunks', [0, 1, 2, 3, 4, 5, 8, 9])
def test_terminal_waits_for_delayed_consumer(terminal_binary, family, chunks):
    # 0 covers CTAs that claim no work; 1 is a first-use protocol control.
    # 2/3 cover the minimum dense split and odd parity; >=4 routed items.
    result = subprocess.run([str(terminal_binary), str(family), str(chunks)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('family', [0, 1], ids=['e4m3-paired', 'fp4-protocol-control'])
@pytest.mark.parametrize('pairs,items', [(0, 0), (0, 3), (3, 0), (1, 1), (3, 1),
                                       (3, 2), (3, 3), (4, 1), (4, 3), (64, 5)])
def test_paired_terminal_waits_for_delayed_consumer(terminal_binary, terminal_build, family, pairs, items):
    quanta = pairs * items
    result = subprocess.run([str(terminal_binary), str(family), str(quanta)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    mutation = 'missing-wait' if quanta >= 2 else 'unused-wait'
    diagnostic = ('FULL reused before delayed consumer reset' if quanta >= 2
                  else 'first-use terminal waited on nonexistent chunk')
    result = subprocess.run([str(terminal_build(family, mutation)), str(family), str(quanta)],
                            capture_output=True, text=True, timeout=5)
    with pytest.raises(AssertionError, match=diagnostic) as failure:
        assert result.returncode == 0, result.stderr
    print(f'unsafe terminal {family}/{mutation}, paired quanta={quanta}: {failure.value}')
