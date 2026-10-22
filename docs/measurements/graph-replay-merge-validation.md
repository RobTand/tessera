# Replay repair composition validation

Resolved the replay-admission merge against owner-outcomes repair HEAD facc55cc.
Kept the HEAD observer implementation: it includes the same complete-inventory
union and serialized publication as replay-admission, plus state-locked inventory
updates, rate-limit rechecking and dirty-state retry on publication failure.
Kept the owner repair evidence unchanged and retained the admission repair evidence
and its separate publication regressions. Earlier source hashes in those documents
identify their piece-local snapshots, not this composed observer.

The exact O1/O2/O3 commands and publication-regression command are listed in
`graph-replay-admission-implementation.md`; reran each unchanged through run_tests
on x86 and GPU, with isolated TMPDIR basetemp. Final results:

| Selection | x86 | GPU arm |
| --- | --- | --- |
| O1 | 11 passed, 8.71s | 11 passed, 2.67s |
| O2 | 61 passed, 58.04s | 61 passed, 13.97s |
| O3 | 187 passed, 57.86s | 187 passed, 18.74s |
| Publication regressions | 2 passed, 9.93s | 2 passed, 3.19s |

All eight final runs report zero skips, zero uncollected modules and zero device
allocations. x86 reports NO CUDA (torch 2.11.0+cpu); GPU reports visible GB10
(torch 2.13.0+cu130). These are CPU stand-ins, not device-completion, GPU numerical,
expanded equality or compiled-qualification evidence. O3 now includes the two
owner repair regressions in test_glm53_graphs.py, in addition to the earlier
185-case selection. Protected acceptance files were not edited.

An initial conflict-marker removal accidentally truncated the observer and yielded
9 failed / 2 passed on O1 x86 (IndentationError at line 839). Restored the complete
stage-2 observer and reran successfully; the failed action below is not a product
pre-fix reproduction. Historical failing-before evidence remains in the two piece
evidence documents. Normal-pipeline post-repair separate-session approval remains
pending as already tracked by #1312; this merge validation does not self-certify O4.

## Initial merge-resolution failure
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"31d500dd6e045e87194b0003c2e456e1e2e9b04ee9d798785243c38dc8653f5e","size":28808}},"operation":"772f1938fbcbdba768830da85fecfc59399310f9f3020203b24fc49a70d1c140","receipt":"920468d49ec23eb3800daba6c1ba48ca1d6c9d34c520092a08fba1fd1bea7368"}]

## Final O1 x86
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"66ce04200f5fe2c35dc83cce44890361be79db6f2ca90a3e955e2cd773760ee4","size":1644}},"operation":"f55f539e0d558f4d6a6ac90340f03012f580db6c9205e93d6bf9384a1f551831","receipt":"30065ccb3301b9097e205cbf34963c7820bafc4c92a27cefb06ee4df3afc4f1a"}]

## Final O1 GPU
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4c2826872e3fda4795ca44d4f95738af5816953a54039f858b31f3cb36c8b51b","size":1557}},"operation":"89e090de333c8af02201ab88f049655042b868a62fa4b5d0fc6fccccb5bde730","receipt":"f252852bf991a152c9f76dbb1532eda6ae442bd2897cf645e7c2dc81567b3c03"}]

## Final O2 x86
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"430410fde3647a0c332bb4519dec0e4cd63b691a4af01cbb05e0b29d1d582475","size":1822}},"operation":"1fe34093dcd4ed8a9c2b060672a80d82f81ca74d9e1ac1e925deae5cd07e0f8d","receipt":"7847abeff549f46f1686107dc227c1ffc2d3ddf2d454e45ada5bab319fcefc96"}]

## Final O2 GPU
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3c49abbe38a36e837b3f62d62f04458afcabc09371ac050d1c6ea65f8df4ec56","size":1700}},"operation":"94a5a1f9f928da7f9a24a939d04e411d93e02caae131092729f8e31027743b42","receipt":"5ed29dc6fda44b50bc9e4f051c62c44acaf148dbf85a04f1f1659931c389fcd0"}]

## Final O3 x86
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"93c53e8f08a71fc856de75db2f2d35ff44e7476f0900b2af953b5614d85920e8","size":1932}},"operation":"1ca26a797d103cebddd2d2fc67d98f6f7f48b2140325931607c0d93b43671de5","receipt":"66badee28db4386be7cd6f2b4a92fecf72ea7eed0713b1ffcd729e57fc9e30ca"}]

## Final O3 GPU
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"e984675fd6b9ea7f0e72925f004f78b71fa891862801595d06f6b23f5c8230cd","size":1850}},"operation":"fbdc2a4d2dd59aaaa2a2035dad6f65cbba2570b2cedba9b84de5037902ea8b49","receipt":"e04a94e7fde2bea34bdf7493055848a90b82002b1fdb5f6cc205c9f856ffadb3"}]

## Final publication regressions x86
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4d3770a55784a0efabdc7279656e76d5875c6acc67cfd6beece493954f0a916a","size":807}},"operation":"c76f103a92033fd905609013510c6084affdc49c18fde9a6a612ff872b7432d1","receipt":"c867303d705286ed8254844b73158460343e9f46406b04f8379ce97e8c03db95"}]

## Final publication regressions GPU
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"a80c4341a9d9875b546a9107ec3180fc90ebb8f024acd3e69fd9bace2826183b","size":720}},"operation":"d735d21c787bd26750a99df2fb013705de32941fcdc91e687c6b558055629d4f","receipt":"77ab3f2c4d2a248fc6b71cc55591c96e04f799bd3d27a80ff6051f4ee0bc12ce"}]
