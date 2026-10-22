# W3 replay admission implementation evidence (#1260 / combined #1271 repair)

Latest combined repair status, action receipts and review disposition are in
[the post-composition appendix](#post-composition-review-repair-and-final-targeted-actions).
Earlier piece-local checkpoints below are retained as history, not current
claims that composition is still unavailable.

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


## Post-composition review, repair and final targeted actions

### Reviewed source and rejection disposition

The normal pipeline separate-session review `068cafcf80d34257a28c7fcfb7daa6f2`
reviewed composed head `b48a5b6d74b3dccdab9b13119c2862086b2ccaf1`
(work-history seq 264299). It found O2/O3 met and O1/O4 not met; the prescribed
selections passed but additional probes revealed two blocking defects. This
supersedes the earlier piece-local statements that no hook implementation or
hook evidence was present. The separate hook evidence is now available at
[owner outcomes](graph-owner-outcomes-implementation.md).

The supplied independent review reported stale-publication failures on both
arms (operations `f168d3deabbc11fd95b391f5438346c35288d9b2d55d80be9920d1d8426f9184`
and `d9ca0c17dbe14d96b925a8850c17323c9b959d1f7e65046276651e4254366761`),
and non-Cartesian token-inventory failures (operations
`80c8bb21aea2d17ab572acc3d8a9cb58e60f3b85df8f070c6af80e2d025cd134`
and `8fef33f2c5b1f30552cffbaaee9b95d92677be2e821f21474a3a4408a5d90e13`).
Its temporary independent-review-evidence.md was not found in this repair
session's checkout or TMPDIR. The unchanged final repair regressions below
independently reproduce both failures on the reviewed source and retain complete
new failing-before receipts; the review operation IDs are supplied provenance,
not reconstructed Test evidence lines.

**P1 stale publication:** a flush copied qualifying owner evidence before taking
the writer lock, then could publish after recapture had published invalidation.
The repair holds the existing writer lock over snapshot construction, diagnostic
copying, temporary-file writing and atomic replacement as a single transaction.
Recapture's forced invalidation still completes before underlying capture work;
an older writer cannot subsequently replace it. The state-lock order is writer
then state; dispatch releases its state lock before flushing. Fork reset still
replaces inherited locks before a child event. Return/exception behavior is
unchanged, and neither host return nor atomic file replacement means device
completion.

**P2 non-Cartesian inventory:** owner descriptors accumulated across bounds, but
the historical token diagnostic consulted the graph map left on the final bound.
The repair derives that diagnostic from the token union of the complete observed
owner inventory. The verifier's consistency and exact-tuple requirements are
unchanged; no refusal was weakened to tolerate contradictory evidence.

`tests/test_graph_owner_publication.py` adds two separate CPU regressions without
editing protected acceptance files. The first pauses a real flush immediately
after releasing its state snapshot lock, then drives real recapture and a second
publication. A lock-attempt handshake distinguishes the serialized and buggy
orders, so thread scheduling cannot silently turn the buggy order into a pass.
The second captures token 1 only at bound 192 and token 2 only at bounds 640 and
8448 through the real layered hook/serving path. It checks inventory, successful
returns and diagnostic union, then passes actual hook output through complete W1
population, real producer, retention and shared verification. Removing token 1's
successful tuple refuses despite unchanged historical dispatch/class totals.

### Measured repair snapshot

Actions below measure the reviewed head plus this uncommitted hook repair and
new regression file (TrustLadder owns the eventual commit). Relevant SHA-256:

| File | SHA-256 |
| --- | --- |
| experiments/glm53_508_graph_qual/digest/usercustomize.py | 792f9557c90e322c5979ab21a1edbbad91baac3a7422c9faceb4ca0e2c3e15ff |
| src/tessera/serving/glm53_graphs.py | 8bc4862d275ffcdbc3bb5484ec7ff08e48076c6629182a5ddca29d859e64a2bb |
| src/tessera/graph_receipt.py | 8d547883c3d791787e01f4673f717fe1e6e55963e161ca40ff581b03e9d44e6e |
| experiments/graph_attest_702/receipt.py | 76fdea4d62068896e655508e03cc27dfa0f405f987db337d00907fc26f746a4f |
| tests/test_graph_owner_publication.py | 9b6c259573f92af9e3c660d5c46021c040326e5157010e112c3b027ca614b45a |

The pre-fix actions temporarily restored the hook from the current reviewed HEAD,
with the repaired bytes kept outside the worktree. Those bytes were restored
before final after actions. Tests do not read git history or construct a separate
environment. No historical commit before the checkout base is needed.

### Exact commands and populations

All commands used run_tests, independently on x86 and arm="gpu", with the arm's
own interpreter/packages. There was no nested launcher, local fallback, package
or environment repair. All pre-fix and final runs below report **zero skips,
zero uncollected modules and zero tests allocating on a device**. x86 reports
**NO CUDA**, torch 2.11.0+cpu; the GPU arm reports a visible GB10 and torch
2.13.0+cu130, not GPU execution of these CPU stand-ins.

Repair regression command (both before and after):
```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_owner_publication.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```
Before: 2 failed on x86 (7.67s), 2 failed on GPU arm (2.59s), at the stale
capture_complete and missing token-union assertions. After: 2 passed on x86
(10.78s), 2 passed on GPU arm (2.91s).

O1 command:
```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_owner_hooks.py -k 'not actual_hook_inventory' -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```
After: 11 passed on x86 (9.11s); 11 passed on GPU arm (2.57s).

O2 command:
```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_replay_coverage.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```
After: 61 passed on x86 (46.24s); 61 passed on GPU arm (13.94s).

Full combined O3 command:
```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_owner_hooks.py::test_actual_hook_inventory_survives_complete_population_producer_verify tests/test_glm53_graphs.py tests/test_graph_receipt_population.py tests/test_graph_receipt.py tests/test_graph_attest_tp2.py tests/test_graph_attest_producer_identity.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```
After: 185 passed on x86 (72.67s); 185 passed on GPU arm (18.19s). This is the
complete prescribed composed selection, not the earlier 106-case migration
selection or a claim inferred from check submission. The retained x86 O3 stdout
was also read back with read_test_artifact to check outcome/population evidence.

### Completed review versus post-repair re-review

The supplied normal-pipeline separate-session review of the completed combined
contract **rejected** that source for the two defects above and missing combined
evidence retention. This appendix records their repair disposition and final
combined action results; it does not turn the rejection into approval. An internal
preparatory code review found no production-code defect in these repairs but
identified a timeout-dependent test interleaving; the test was strengthened with
the explicit lock-attempt handshake and rechecked before/after on both arms.
That internal review is not the required normal-pipeline review.

Post-repair separate-session approval and substantive final documentation review
remain for the normal pipeline (existing tracking issue #1312); re-review is
requested with this repair. No new competing review workstream or issue is
created here, and no workflow transition/publication is performed. Until that
review is recorded, O4 is not self-certified complete. Scope/image/model/source,
public verifier, launch/rank configuration and byte-integrity boundaries remain
unchanged. These results qualify no measurement, expanded equality population,
compiled execution or device completion.

### Final before/after action receipts

#### Repair pre-fix x86 — 2 failed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"072dcca861cf69f85016e94495b3cbe3b4f976ca9db8e3d9375ea1fbf53a9fe9","size":9704}},"operation":"4c8e0b7a18197bb00e84048c1b859b036c7c471aa674926b9c134219083d9776","receipt":"3fbb8bf6e3f0dcfcab8b80930010c5d15a8e3fe09fb726369a4eb5514f35244e"}]

#### Repair pre-fix GPU arm — 2 failed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3baa89d80e33278498eaff4267fa8610ae43ddc4e7f194c82a9f044fc5284627","size":9229}},"operation":"1a2ff38839035392a5098a4b2c95c1cbf112d06052c2be2352d0f7945e8354bf","receipt":"a2bc5460093ea1e8a959c5da95b2ab621f1b0640f4765b321d3aeb5713f450f2"}]

#### Repair final x86 — 2 passed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c330807c50238c0ebef34befc988223c0e28f52445248ac1a5a4ee0a74b2e58f","size":808}},"operation":"fadc20e96f8b35a4e918f1fd1460dd199d2871062beedd086f3d519d54590595","receipt":"71fdd7680a2dac8ec50d4b4ee4ecbc22d271ad4f4995e44bc899675a5ea877d7"}]

#### Repair final GPU arm — 2 passed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ad89650fa990c75d1f5c35166b9336c4c87e9a8d9bc99710369fe2ad185bd0ba","size":720}},"operation":"be429296d5847bceb0dc9150be4d89c52736134abed419faee42fafaa2046603","receipt":"ef38e5f000a96cf79e8abdb033fcd684658197d926fa464627e3fe8d0712ca72"}]

#### Combined O1 final x86 — 11 passed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"695fd50a9d6e122654b6e0e0bf3ad88c04316250ad577d46df2fcd7871ad8bc9","size":1644}},"operation":"0e4716a26ce591b77d4eb2e8de00934e7c6a1d7101649e394a9df56664044fc3","receipt":"25c3992d368355d47f88ca6ba640ec6beab7359d9ec281a38d0a9da5bb941f4c"}]

#### Combined O1 final GPU arm — 11 passed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1df3ec289594a01a001b7c92e592b5334037ba8e79dc0475234560da6ea5ae77","size":1557}},"operation":"548672809ada2b317f3849c587ef1cc33ab61bb9580fb2358107e9039c78f677","receipt":"e7b61fe462f34de99811619d603003fd8a92a204f9458b78b453e9ea6c98aa2c"}]

#### Combined O2 final x86 — 61 passed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c808a34d1648884774b59988ab18850352d5b2fb0800870abe3a78f5069b702a","size":1810}},"operation":"e87121081710ab8164bcb0a4ffa8f5e8cb91b126c046f67b6ea25f46c7d8019e","receipt":"a3dd79baf31e48e3dc6efd5dc442eaa45c83203c45145d27ec80aa81f9ef134e"}]

#### Combined O2 final GPU arm — 61 passed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"b35c878d011ed800fa2ec332b6ea945e9922c8e51ab2e892b495c5054b1ebd37","size":1715}},"operation":"86603a6e4f7337a5b1d09ce1da78ae48679f5da79cd53b67adcc5a2295e57cfa","receipt":"73bda2b25dfc740da2b82fc41a7d0682a84d89ccb269fe8cdf4fc543eaaacaed"}]

#### Combined O3 final x86 — 185 passed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"308afee251e36b3c52da613b8c7ca9b13b0a9e1d1e6d1aebe2652c15c759b61e","size":1943}},"operation":"2f1e54c8658a0b526cf11dae414d19a2ef76391e688b3262c56d6123b2b6472e","receipt":"9615267e3fd6bb2704f5b46dbc54eba0eef412ae0dfec40e411ac0d58c0958bd"}]

#### Combined O3 final GPU arm — 185 passed
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"fda48e0947f26f6ea90af84d54555d1c3ce301ab90c8f5adc4c5a41e1035c0c8","size":1838}},"operation":"13ce91ebf633b3952a114df9192a449cc029c8ab178171887fbe364558681176","receipt":"690aa91476f8428524a1d4b5eb4cadb54c1c79d3ff924f394efbfa45e38f9ed6"}]
