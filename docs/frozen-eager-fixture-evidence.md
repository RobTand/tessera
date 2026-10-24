# Frozen eager selector validation

Scope: historical source custody and selector behavior only; no kernel numerical,
performance or served qualification. Reused the fixture and loader from
`9f262ee38be30a239da4af923dd48f8ae14dd0e6`, excluding optional-prerequisite skip
paths. Acceptance file `tests/test_frozen_eager_fixture.py` remains unchanged.
Its retained research RED records the unavailable historical tree before any
selector executed (original target: one failure; combined: six failures and
247 passes). No retries or weakened assertions repair that input failure.

## CPU acceptance

Executed all five frozen-fixture acceptance cases with two workstealing workers,
`PYTHONPATH=src`, OMP/MKL/OpenBLAS threads one and scratch under `$TMPDIR`:
5 passed; CPU Torch 2.11.0+cpu, no CUDA, zero skips, zero uncollected modules,
zero device allocations. This includes the actual offline, no-history target.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8ce0e4ca29b806897c65d0534271c90c01a762f1e8ea8fd1a05601a47204a922","size":888}},"operation":"87a00e8a2948365d803dec5dd2d80fcf63e6bfa2a117e823b1a384eb9d69c2e1","receipt":"18cc47486bec4536f6e88500e56e0a211b7529dcb1250674671317eab99b5187"}]

## Configured GPU decision (before execution)

First execute only the named target on the configured GPU arm. This decides
whether the frozen selector child retains the sealed CUDA Torch/Triton
prerequisites on that actual arm, which CPU cannot establish. Expected budget:
under one GPU minute. No kernels or numerical qualification are requested.
If successful, run the five consecutive acceptance executions and the bounded
two-worker combined selection on each arm; stop the sequence on any failure.
The GPU follow-up is expected under two GPU minutes and establishes repetition,
offline import custody and preservation of the eager-levers module and guards,
not numerical coverage. CPU acceptance already establishes fixture integrity;
GPU follow-up checks its compatibility with genuine CUDA dependencies.

## Executed GPU pilot

Named target: 1 passed, CUDA Torch 2.13.0+cu130, NVIDIA GB10 visible,
zero skips, zero uncollected modules, zero device allocations (2.15 seconds).

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9b11e42aa67b13e66b0d2a48c7e47f107fb47da8f7de04c5ed3fcc74ac19dd3f","size":557}},"operation":"d93260d3701ee205e49eb5416b38a99040ce6e6b99659f2252dd58f8cf05ae88","receipt":"09f1d8e266b20d4166d84a5a7379c02c21a57a4bb2e32f501234157c5710a867"}]

## Consecutive acceptance and bounded xdist

Executed the O3 command verbatim on each arm: export `PYTHONPATH=src` and
OMP/MKL/OpenBLAS thread limits one; five invocations of the named target
with `--durations=10 --basetemp="$TMPDIR/pytest-$run"`, stopping on failure;
then the complete eager-levers module, frozen-fixture acceptance and both
mandatory guards with `-q -n 2 --dist worksteal --durations=10
--basetemp="$TMPDIR/pytest-xdist"`.

Read both retained stdout artifacts in full: each records five separate
`1 passed` results followed by `253 passed`. No failures, skips or uncollected
modules on either arm; zero device allocations throughout. CPU: Torch
2.11.0+cpu, NO CUDA. GPU: Torch 2.13.0+cu130, one NVIDIA GB10 visible. Combined
xdist selections completed in 126.73 seconds on CPU and 36.50 seconds on GPU;
GPU repetitions took about two seconds each, within the recorded budget.
These are selector/source-custody checks, not CUDA numerical coverage.

CPU repetitions and combined selection:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0530f44a13806a729dd592edd28cff86e41b619feb08fd90b539718157bb454d","size":5067}},"operation":"d0062ef563436052b73397fd74a098d75ed6c56d9ac00936fe5a36d59ca025a9","receipt":"3de5b3a5af686b580577a9b47fbe652d3b8518d0f9709aac4ff552867a86c3ba"}]

GPU repetitions and combined selection:

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"dd2ade124293fb25af161defbc0d3fe08033d0e67c9d8b497e7c85b84e45e8ac","size":4527}},"operation":"94fae0c6d086cca0b60df24c36bd71014bb89964cf5b7de000cc491657ddeb05","receipt":"c2533596701314ec1d27ebb5af5ec07d46eb06e2a69533ab3e4f8291286fc774"}]

No tests removed or tautological tests found in the changed eager-levers module.
Only the old optional Torch skip was removed so missing required dependencies
fail visibly. The benchmark pin and historical selector assertions are unchanged.
Full merge-result validation remains the coordinator's responsibility.
