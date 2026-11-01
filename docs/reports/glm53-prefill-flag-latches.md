# GLM53 prefill flag latches: verification

Change base: `e83b338234e6dfb0315e5d90c5662a0eea93ce5f`.
The supplied pre-fix acceptance receipts report 34 failures at the missing
`RuntimeError` assertion (`tests/test_glm53_prefill_flag_latches.py:44`) on
both arms. The acceptance file is unchanged by this implementation.

## Checks

Executed through `run_tests` separately on x86 and GPU:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_glm53_prefill_flag_latches.py tests/test_glm53_prefill.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Both arms: **90 passed**, zero skipped, zero modules not collected, zero
device allocations. These are Python configuration/regression checks, not
CUDA numerical qualification. x86 reported no CUDA; GPU reported a GB10
with torch 2.13.0+cu130.

x86:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"42c55df9ff9e2d292111a0d7e9870100f0385fdf9f7bd998f89efa5559f661fe","size":1540}},"operation":"e55ac23a1072cef14958851f530e32699e84b383073f5ec8b69a8c744e901f9d","receipt":"762b6a096a5f2212c7a17af7ff1374bb8fd8575a0601cd8c90dcac469ffb08b2"}]

GPU:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9243f2cb8b0e9b4ca8e48b2f8be112a46a0a31cc671002fa0f31f5c62bc6256d","size":1453}},"operation":"094c28c3618c3422078594a0679865229dd944066264be728b870de98fef9938","receipt":"d302c5d37a16c1686a6e54ca8bc286df2235d1fdbe296cdf7150b365ee8326d2"}]

Also executed on both arms:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_serving_residency_modes.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Both arms: **7 passed**, zero skipped, zero modules not collected, zero
device allocations.

x86:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e2934e801ee153ed283bada18a6a59f2ecfe8abb813f1919657f6bfbee75692d","size":975}},"operation":"4e76749b7e9de84752a54b6ef423a048c8925c92d76779b7f34978d405eabf1c","receipt":"0bf6053a8746cc24040251367ee81257c587650492e24308008abe24438b7105"}]

GPU:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"88be5c0f78004945ddc1023192822a3379bbe8ef186d1271bd92080a6850dc49","size":888}},"operation":"d8ca49cab2a8ad6e1b3fe64bf352cc5e9b89bf0d858402fc5d409160d43f240e","receipt":"ffdae382e5f21c1179d4f8f779104ad27026b50bdc49acc41e2317e254d1972c"}]

## Change-boundary and documentation review (O4)

Source/diff review confirms the only production behavior changes are the
three reported readers and the optional absent-only `latched_mode` default.
The existing mandatory `lane.serve_mode` caller is unchanged. Validation
still precedes latching. Selection, installation caches, stock fallbacks,
source qualification, custom ops and ONORM → SP → KDA ordering are unchanged.
The existing prefill tests reset only these three flags before/after each
test; the combined KDA test resets before its independent enabled scenario.
Architecture §5.1.3 and helper documentation describe first-successful-read
latching and distinguish configuration errors from interface declines.
This review is distinct from test execution.

Ran `python3 tools/impacted_tests.py --ref e83b338234e6dfb0315e5d90c5662a0eea93ce5f --json`
against the tracked working-tree changes. It returned `narrowed`, but its
conftest/import reachability conservatively selected a broad test population.
That broad population was not run here; the full merge-result campaign is
left to the coordinator as instructed. The targeted checks above passed;
they do not establish that broad population's result.
