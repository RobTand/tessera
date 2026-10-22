# W3 replay admission implementation evidence (#1260 / combined #1271 repair)

## Scope and contract

This piece implements the producer/shared-admission portion of W2/W3. The
producer retains full dispatch records under their original basenames;
`tessera.graph_receipt` independently applies owner-local, descriptor-exact
successful-return coverage under the authoritative arm name and TP scope.
Historical marginals remain diagnostics, never proof. Actual non-Cartesian
inventories are validated without constructing size-by-class products. All
recorded relevant counters/dimensions, including unused entries, must be valid
integers; legacy totals cannot be promoted into successful-return coverage.
Malformed consumer inputs return refusal reasons. Scope, fabric, W1 population,
producer identity and byte-integrity behavior remain in place.

The hook/serving portion was separately owned at the time of the initial W3
actions below. Composition and instrumentation repair are now validated in
[owner outcomes evidence](graph-owner-outcomes-implementation.md), including
both-arm O1/O2/O3 receipts, non-Cartesian producer/verify coverage and the completed
initial implementation review's disposition. The architecture describes the
combined wire contract; this W3 piece alone does not implement capture observation.

Positive fixtures in test_graph_receipt.py, test_graph_receipt_population.py and
test_graph_attest_tp2.py now contain owner evidence; zero-replay negatives also
remove strict successful coverage. Assertions and existing negative populations
are preserved. Protected acceptance files were not modified.

## Supplied before evidence (research actions, not rerun here)

At base ff95111ee8fed33e9a4ea23e59c46444cd8fd61b the final retained acceptance
selection failed all 73 cases on x86 in 58.77s and on GPU-arm CPU stand-ins in
14.25s. Missing owner_evidence, premature serving counter increments, producer
false equality and direct-verifier acceptance were representative failures.
Existing targeted regression passed 184 tests on each arm. Both selections had
zero skips, zero uncollected modules and zero device-allocating tests. x86
reported NO CUDA; GPU visibility is not evidence of GPU execution.

### x86 acceptance before
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"523393b5e4783ff40e864a1e8f801a5adfdfe58b40cb2468d32649841bd1f7af","size":176411}},"operation":"bfeaf912f55a7da1efebc26115245899ca8c45b248d5fa98c92c70a487b9bb1a","receipt":"7b769e966ac4381f89a8de9688952748f0a8be7196835c50bd5e9b0507504b43"}]

### GPU-arm CPU-stand-in acceptance before
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c09ce1037010109ae9d4a9ab0796cca130655726b20b61a64e26fc240975df33","size":174448}},"operation":"7e0162ea781995ac6abb5631ae8a1f66149ede33e3948cf28f478614d6b4ce24","receipt":"b6f16473333233259e9e0e294db8d9a60975b2309c9edca88157614f9416f6a8"}]

### x86 targeted regression before
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"2d9bdbfc45e9364cd3c539728f29375bb9ad0a229d476ec49eb0433191d4f356","size":1981}},"operation":"d123994b3496f6c8349bed169d092e4994752a26da8baf3bd67218652ba945b4","receipt":"1299619986d0afd294e19bd7209cdb74bcb33bf4958fb1a332ec42fcca8d50b7"}]

### GPU-arm targeted regression before
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ea333af9a79068f86a824f5666dba74862584670bce441be605fb53158e83203","size":1840}},"operation":"53040b510a2e460994affb7b026b8df00673173a96604cc0aac3df45feb501b0","receipt":"23909e82532798cced47eea31cc2cc33e4862ad0bb51788ffc52c94dc47e580f"}]

## Final W3 actions

All checks used run_tests with plain commands, native thread limits of one,
two workstealing workers and isolated TMPDIR basetemp. They ran independently
on both configured arms. No local fallback, nested launcher, package repair or
environment modification was used. These are CPU-only checks on either arm.
All final runs reported zero skips, zero uncollected modules and zero tests
allocating on a device. x86 reported torch 2.11.0+cpu and NO CUDA; GPU arm
reported torch 2.13.0+cu130 and a visible GB10, not device test execution.

### O2 command
```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_replay_coverage.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```
Final result: x86 61 passed in 40.71s; GPU-arm CPU stand-ins 61 passed in 13.77s.

### Migrated regression command
```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_receipt.py tests/test_graph_receipt_population.py tests/test_graph_attest_tp2.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```
Final result: x86 106 passed in 103.75s; GPU arm 106 passed in 17.49s.

### Producer identity command
```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_attest_producer_identity.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```
Result: x86 27 passed in 7.32s; GPU arm 27 passed in 2.71s.

## Composition and review disposition (updated after instrumentation repair)

The initial W3-only actions did not validate the composed O1/O3 paths. After
composition, separate-session pipeline review `068cafcf80d34257a28c7fcfb7daa6f2`
passed the prescribed selections but found stale-writer publication and final-bound
size-diagnostic defects in the observer. It therefore refused O1/O4, not the
shared admission rule. Both findings are now repaired and reproduced in retained
regressions. [Owner outcomes evidence](graph-owner-outcomes-implementation.md)
records their before/after receipts and final combined O1/O2/O3 validation on
both arms: **11 / 61 / 187 passed per arm**, with zero skips, uncollected modules
or device allocations. O3 includes the complete serving regression and two new
repair cases. W1 protections and all protected acceptance files remain unchanged.
These final observed results supersede the initial composition-pending status;
they do not retroactively make the initial review an approval.

Post-repair completed-contract review and its approval remain with the normal
pipeline, not self-certified by these implementation documents or satisfied by
the research-test review. No publication or workflow transition was performed.
This work makes no device-completion, expanded equality, compiled qualification,
#1159 measurement qualification or #1226 admission-acceptance claim.

## Action receipts

### Earlier O2 x86
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8c8c81dd05b522408ef208d6c3d031cb064e6fd5139932b350e32f2d79e2df2d","size":1812}},"operation":"5311148042c93b096e4cc5f757ffd465101f133312ba904e74726537e4085c86","receipt":"f8e21c3051c351e5767dadcdc601e1ad50e0840d2d895c94a4b36c063d0f15b0"}]

### Earlier O2 GPU arm
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"45202627994494ad055e1e6f1e48b733971a75214b42415bb7a1f47d3a52d8ce","size":1699}},"operation":"8cb3160525b76c7ee9a62f3559599445a113dca024acdf3e53675aaa3d83ea99","receipt":"41220ad985f96efdc260fbfa4dab6d94a14871d1644f5e2787968122eaf26ecf"}]

### Earlier migrated regression x86
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4d689f800f762d5087fb72a560f3a5d39a32d6b938446e48d20e37abb6e59704","size":1856}},"operation":"b9fadc53aa085b5a401322dca552eee150a563aba9700c0df2141250a816a560","receipt":"02412418099a2c96ebe51d5b08f1c665b642bde19206069a4d5c270c68a60592"}]

### Earlier migrated regression GPU arm
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e522d5abfd0d0aba18864140f5034ba1e9e2d460a50f422598ccc2ebccaf55c3","size":1769}},"operation":"3659cb45dbb4d1f37f98ef517b00e28803625196b17c00c74ce0d7be61ce5c8e","receipt":"b5fc047043337f992b9e68e812fac0c00899f2c81c027b1a6e167f2416a92885"}]

### Producer identity x86
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b9d10f5c2dbefddd62827ed9c112625ee0ff7014d435f37ce740a35d388b1897","size":1861}},"operation":"8497ef190a888a3dffeadc30b1438b8b5f6fb586d909b406f1a7cf7499958bc7","receipt":"b100027235e5509039b02c13ce5229e9638be81d1a6bc6a452fd3314b561f7d7"}]

### Producer identity GPU arm
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f8ada566942cb58047e307978284f338b40ade24003c376a4311e3b0d2468877","size":1774}},"operation":"664d3db19f2d821869a544d42061902488a0ac3ee09f40d840e7966453331ad6","receipt":"1d3899db3434400b787694398678c298b46bced0263164eac294695b42d35371"}]

### Intermediate failed action (61 failed, 28.67s): added aggregate validation initially used incorrect aggregate field names. Corrected captured_sizes/replayed_sizes and pre-JSON integer keys; final O2 receipts above supersede it.
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9b5fdac5a7f201211a04682342373ac978ec93fcff58ebf219547fad9396ddf6","size":126095}},"operation":"4b0ddb100b1a236832724816955cc51a60138d3f215e64706d05153206a01a1e","receipt":"7a505d4fa8fd32f54ac292ef21fd0e33a8fdd0d5d772d120bebb514674a7fdab"}]

The final reruns below followed centralizing diagnostic validation before the
producer aggregates its historical marginals. Earlier green runs are retained
above for action history, not substituted for these final reruns.

### Final O2 x86
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"883f8d9165d385816391f83f3eee7be7b2e617c3d7c7e558fdf33cf61fbfcd25","size":1830}},"operation":"bd79e0455f5bef3d58a68c770cd4dc548a2446ff0357864d17519b574c64818c","receipt":"79df1d7d5a3c1f20f7efef2d8c662804ed4403a183b12e7241223dadb388b683"}]

### Final O2 GPU arm
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ba914f80fdc5ac68852510bbaf5af06c488d64362600d2fa1133defbd18e7a2a","size":1716}},"operation":"b0fe2a6b97ff5a68d78d939fb5686d83a783aaf969c2388ca04bcf69786be654","receipt":"30c2c6082fe14bfdbbca62d906540a6bdf13bbb764f2d1d3de622d1d26fc9783"}]

### Final migrated regression x86
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6f4d3f958a94a80ad1d8d54ba7d336c8efb5d828241f4948ccd033c63ed05c22","size":1908}},"operation":"a2c063ab8d0ba253e00672b5ade2067d2e9b896e5e7ddc8a97ab879cd3dc1598","receipt":"035705096992c18aef947d0e3b97226e12ae6d548ae713ed27d5f62f54da031e"}]

### Final migrated regression GPU arm
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b1d8d1b224d2689253dda1f1279430e9f9470b9fc2ad86b6899e9830e1820106","size":1768}},"operation":"86886e0460df509f12caf6868c52ae6355bb87baa6a2e686b23be1e7cc64a202","receipt":"e1c36f5dabe2811a21d46264b34598592cc94e32fee2a483691d25764c917120"}]
