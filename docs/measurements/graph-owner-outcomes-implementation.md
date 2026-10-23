# W2 owner outcomes implementation evidence

Scope: instrumentation portion of #1259/#1260 (overlap #1271), based on
ff95111ee8fed33e9a4ea23e59c46444cd8fd61b. Initial actions below were piece-local.
The repair section records final combined O1/O2/O3 validation after composition
at b48a5b6d74b3dccdab9b13119c2862086b2ccaf1 plus this repair. Admission and
ARCHITECTURE.md are retained from the admission piece; see also
[admission evidence](graph-replay-admission-implementation.md). Separate-session
review found two defects, addressed below; post-repair approval belongs to the
normal pipeline, not this implementation document.
No protected acceptance files were edited. W1 population protection is unchanged.

## Contract and reviewer handoff

Dispatch records retain pid, captured, counts and optional tessera_classes.
The additive owner_evidence map uses full Class@instance-hex process-local keys
(bookkeeping, not an identity seal). Each owner has capture_complete,
capture_error, captured, replays and replay_failures. Descriptors record mode,
tokens, num_reqs, max_seq_len (null for unsplit; actual applied bound for split).
Replay records add count; failures add error and count. Dispatch totals are
attempt diagnostics, never successful tuple coverage.

Both base and model capture wrappers share per-owner nesting depth. The
outermost entry invalidates earlier captured/replay credit immediately and flushes
that nonqualifying state before calling capture work (including interrupted work). Nested
wrappers accumulate actual graph keys, deduplicated by descriptor; they do not
invent a Cartesian product. A model failure after super returns still fails the
whole transaction. Serving's outer per-bound capture and model validation wrappers
compose that same observer via its wrapper attribute, so a successful None return
from layered capture is not a failure and per-bound calls do not reset one
another. Any nested failed/incomplete capture leaves the current transaction
nonqualifying; a later fresh capture clears that state. Incomplete captures have
an explicit error, while exceptions retain repr(error) and propagate unchanged.

The replay observer runs only inside execute_model/sample_tokens serving scopes,
excludes dummy calls, restores nested scope state, and reads the selected graph
map's bound by identity. Only an underlying successful return (including None)
adds a replay count. Exceptions add replay_failures and are re-raised. Serving
REPLAYS similarly increments only after the target/drafter underlying call returns.
Unsplit and eager paths retain their existing execution behavior. Capture/dispatch
flush I/O errors do not replace engine return values or exceptions.

PID changes reset observer evidence, diagnostics, locks and thread-local scope
before the child's first accounting event, not only at output flush. Serving
CAPTURED/REPLAYS and their lock are reset as well. Inherited graph objects do not
constitute a fresh child capture. The configured GA_DISPATCH_LOG prefix and atomic
replace publication are unchanged; no rank/path/launch assumptions were added.

All claims here are host observations with CPU stand-ins: successful replay
return is **not device completion**, numerical equality, compiled qualification,
#1159 measurement qualification or #1226 acceptance. Producer/shared admission
must independently refuse insufficient legacy diagnostics, and preserve rank /
process / owner boundaries.

## Supplied failing-before and regression evidence

These are supplied research results, not new runs performed by this piece.
The combined 73-case acceptance selection failed on both arms (x86 58.77s;
GPU-arm CPU stand-ins 14.25s). Representative failures: missing owner_evidence;
failed target/drafter calls changing REPLAYS from 1 to 2; invalid evidence accepted
by producer and verify. Existing targeted regression/W1 coverage passed 184 on
each arm. Both selections reported zero skips, zero uncollected modules, zero
device allocations; x86 reported NO CUDA. GPU visibility did not imply GPU work.

Acceptance x86:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"523393b5e4783ff40e864a1e8f801a5adfdfe58b40cb2468d32649841bd1f7af","size":176411}},"operation":"bfeaf912f55a7da1efebc26115245899ca8c45b248d5fa98c92c70a487b9bb1a","receipt":"7b769e966ac4381f89a8de9688952748f0a8be7196835c50bd5e9b0507504b43"}]

Acceptance GPU arm:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"c09ce1037010109ae9d4a9ab0796cca130655726b20b61a64e26fc240975df33","size":174448}},"operation":"7e0162ea781995ac6abb5631ae8a1f66149ede33e3948cf28f478614d6b4ce24","receipt":"b6f16473333233259e9e0e294db8d9a60975b2309c9edca88157614f9416f6a8"}]

Existing regression x86:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"2d9bdbfc45e9364cd3c539728f29375bb9ad0a229d476ec49eb0433191d4f356","size":1981}},"operation":"d123994b3496f6c8349bed169d092e4994752a26da8baf3bd67218652ba945b4","receipt":"1299619986d0afd294e19bd7209cdb74bcb33bf4958fb1a332ec42fcca8d50b7"}]

Existing regression GPU arm:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"ea333af9a79068f86a824f5666dba74862584670bce441be605fb53158e83203","size":1840}},"operation":"53040b510a2e460994affb7b026b8df00673173a96604cc0aac3df45feb501b0","receipt":"23909e82532798cced47eea31cc2cc33e4862ad0bb51788ffc52c94dc47e580f"}]

## Initial piece-local action checks (superseded by composed repair checks below)

Each command was submitted through run_tests directly, without nested launcher,
package repair or device campaign. Temporary files use the run's isolated TMPDIR.

### O1 protected hooks

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_owner_hooks.py -k 'not actual_hook_inventory' -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

x86: 11 passed in 7.92s, NO CUDA (torch 2.11.0+cpu).
GPU arm: 11 passed in 2.62s, visible GB10 (torch 2.13.0+cu130).
Both: exit 0; zero skips, zero uncollected modules, zero device allocations.
The actual_hook_inventory integration case is intentionally excluded here and
remains assigned to post-composition O3 validation.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"131c9e1ed2aaaa75a25316fc9875f7c61e41f31d4acbc4af24d39cddb1c4a476","size":1644}},"operation":"029f1b3bb4b01983566f11abc1b1cb1ccc70371733380c838d950f48dc7bf6b8","receipt":"f72ec98be85805c5853d69b79cc5997af369dc68b7e61efb5360847996fe872d"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7c7f35650f1954fbb15cbf61370ea361b6dc745cbb480c75bc0e25371bee2985","size":1557}},"operation":"d0e66ed292c0a40f3f822bba4610d9ebe5d88c8788a34087d49cdd37c5031eec","receipt":"f5864f164ff08be4619b4ff44780a91951c096ee13d8085f2b7fe7ebbcb0dbf1"}]

### Focused serving regression

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_glm53_graphs.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

x86: 51 passed in 7.20s, NO CUDA. GPU arm: 51 passed in 2.44s, GB10 visible.
Both exit 0; zero skips, zero uncollected modules, zero device allocations.
These are CPU protocol/serving stand-ins, not CUDA numerical/capture execution.
No final requested check was unavailable or failed. Earlier iterations also
passed; the hook receipts cover final source; the serving receipts cover unchanged
serving source before the final observer-only capture-entry flush addition.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"42c70c1fcba66bb3cec9057fcac5d705d9ccf57162b650e55c37fd587a04bec9","size":684}},"operation":"d0b994c3ba5bebe2431822b5ab3bbd0883054ecf7a83f39c9c8dd5465d704784","receipt":"e33fc1419aae5c60f711874e7d82b433f86f6789cb4d718e01a5622de00a9c51"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"f4b8098ea8f469167c8e606a9facf071d3b6fb1f7cb685e5adf550b552e4090e","size":490}},"operation":"d803ef9b772939de97f33c747b90c1da8371270ab5bc0373274db66172d427c8","receipt":"1ee8ac0f83088f4be4bf986a213b653617c18d91d0bab7f71cafd28d1dc68cbf"}]

## Separate-session review and repair disposition

Pipeline review `068cafcf80d34257a28c7fcfb7daa6f2` inspected composed head
b48a5b6d74b3dccdab9b13119c2862086b2ccaf1. It passed the prescribed O1/O2/O3
commands but refused O1 and O4 after finding two additional instrumentation
failures. This was completed-implementation review, not the earlier research-test
review, and was **not an approval**. The reported stale-writer reproductions were
operations f168d3deabbc11fd95b391f5438346c35288d9b2d55d80be9920d1d8426f9184
and d9ca0c17dbe14d96b925a8850c17323c9b959d1f7e65046276651e4254366761;
non-Cartesian diagnostic reproductions were operations
80c8bb21aea2d17ab572acc3d8a9cb58e60f3b85df8f070c6af80e2d025cd134 and
8fef33f2c5b1f30552cffbaaee9b95d92677be2e821f21474a3a4408a5d90e13.
Those operation IDs are supplied review history; new, fully retained failing-before
and passing-after evidence for both defects follows.

1. **Stale publication:** a flush could snapshot complete evidence before its
   write lock, pause, and overwrite a later recapture invalidation. Repair makes
   snapshot construction, throttling and atomic replacement one serialized writer
   transaction. Owner mutations and snapshots share the state lock; wrapped engine
   calls and I/O do not hold that state lock. Serving diagnostics are copied under
   their own lock without nesting it with the observer state lock. Publication
   errors retain dirty state for retry. A fresh capture replaces its evidence
   generation, so an old in-flight replay return cannot credit that new capture.
2. **Non-Cartesian diagnostic:** split capture leaves self.graphs pointing at the
   final bound. The historical captured diagnostic now derives sorted unique token
   sizes from the complete observed descriptor inventory, not that final map.
   Strict shared verification and its diagnostic consistency refusal are unchanged.

Added regressions live only in owned tests/test_glm53_graphs.py. The stale-writer
case keeps the real background daemon enabled and deterministically gates a
competing flush before acquiring the publication lock. A second thread begins
recapture and blocks inside the underlying capture after publishing invalidation.
The delayed writer must not restore capture_complete or replay credit; the shared
gate must refuse both during that block and after its deliberate capture failure.
The non-Cartesian case uses real layered hook/serving paths with token 1 only at
bound 192 and token 2 at bounds 640/8448. Layered capture and replay return None
successfully. The real producer and direct verifier accept the complete inventory;
removing each actual tuple's successful replay refuses, with old totals unchanged.
No new size/class cross-product or expanded equality claim is introduced.

### Repair failing-before and passing-after actions

Command (plain run_tests on each arm):
```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_glm53_graphs.py -k 'owner_recapture or owner_noncartesian' -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

Before changing the observer, both cases failed on both arms: x86 2 failed in
7.77s; GPU arm 2 failed in 2.56s. Inspected retained stdout shows
`during incomplete recapture: published capture_complete= True shared refusal= None`
and the token diagnostic `[2]` despite all three exact captured/replayed tuples.
After repair: x86 2 passed in 9.62s; GPU arm 2 passed in 3.42s. Every one of these
runs reported zero skips, zero uncollected modules and zero device allocations.

Before x86:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4373f1d6c02c999b46dda0c0377d15ea5cf1719ec249b9e3d80adaea38c1f114","size":13304}},"operation":"98632181578c585b98f052c2cdb5145144eff587a736dbaf10331b64e4de092a","receipt":"2e1a976e24e8c546ef4597f4311e7e54f3a454efb527090d979050ea5f15e611"}]

Before GPU arm:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"1c9ab70053ef35f4e4f7419028523a971b061644bf9c245f52b0bed925b4849e","size":12323}},"operation":"96a8ced38a2e1a86475c1d56f61e314b621944c10b02fca9ed66e3b9451cdda8","receipt":"830a772eb6e69b47da952fe91e7b329eea3700de1fc39b4dc1d63dda84a18b6c"}]

After x86:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"3fb6129442510da139382674fd6c19d0ca417352ff862e840f54a3ba8b624dbb","size":800}},"operation":"c2a4084931d4f1e186bc78b12fd52a46d7b0730795f3c2dbd49b422b4cbf45a0","receipt":"97c0e763d52d8560caa20293c85cd5a290d44970b0d45d0548e48779d18fa791"}]

After GPU arm:
[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"0e45aefcd423c97addca17001c859b9385a84bfd8a729f6c63d862f25993c0bd","size":713}},"operation":"edfa4869af08cc2fa8459b17109f8fab18e7056384873d7ae8d6123e9f5c5e84","receipt":"86ada212fae02c4b5a3086b135fc5e8aa29f2f7529d341554fc9275962e376a9"}]

## Final composed validation after repair

All commands below used run_tests directly with the composed admission code and
the repaired observer. Each ran on x86 and the configured GPU arm, with two
workstealing workers and isolated TMPDIR basetemp, without nested launcher,
environment repair, repin or device campaign. Every run exited 0 and reported
**zero skips, zero uncollected modules and zero device allocations**. x86 reported
NO CUDA with torch 2.11.0+cpu; GPU arm reported visible GB10 with torch
2.13.0+cu130. These are CPU-only bookkeeping/population/producer-verifier tests,
not numerical runtime qualification or evidence of device completion.

### O1

Same exact O1 command printed above: x86 11 passed in 6.95s; GPU arm 11 passed
in 2.63s. The two additional repair regressions are also covered by O3 below.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"fd1a9259c0627ad2c3e718da3928808c946478652255cd61d0c8cba0118107b1","size":1644}},"operation":"f01026acf50b3367fd3a13ae008be04f6de62f1f5aed13ba23c95e293b5a31c7","receipt":"ef63a1b239f0dbcd57f57669037b5adb588a677aba2d51ffdef5e764d70b8600"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"23e0af820220740f9fe4266f42d7c2049192618ea9c6ece65ae1b08a2744b925","size":1557}},"operation":"40c25712683801911282b8e62054ecde832614303268720dec8c02b2b6c1fd99","receipt":"8e1348c9a543fc7e38cd7c33941ae88c7f09b0257c690b1adf474d5b866394d1"}]

### O2

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_replay_coverage.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

x86 61 passed in 49.17s; GPU arm 61 passed in 13.83s. Admission code and its
counter/descriptor/legacy refusals are unchanged by the instrumentation repair.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"5fde3eff214f90ab7b8b5dcd0de667fa2b543727b7a3de3a2105ed55372969e7","size":1791}},"operation":"2797f6f70afd3afb97f413f33147237e9aca12a39141011fd4429471b0e612a6","receipt":"616b19968919afec008f718cfeb941beee19c1940c2cca3ec65dd495ec862101"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"6ad87d6aae7e53468e6a0bb6c178a93cf43712a3ae034e106f47483d1d4cb59b","size":1736}},"operation":"55e8fbeb974782e8ef2eece4c5468d2ee18305d338f0cab121d3f32f14c52f69","receipt":"76494114383919182bacc36904fddcb74545f63c22ee1e3cbb2f56f889561b4e"}]

### O3, including full serving regression

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_graph_owner_hooks.py::test_actual_hook_inventory_survives_complete_population_producer_verify tests/test_glm53_graphs.py tests/test_graph_receipt_population.py tests/test_graph_receipt.py tests/test_graph_attest_tp2.py tests/test_graph_attest_producer_identity.py -q -n 2 --dist worksteal --durations=10 --basetemp="$TMPDIR/pytest"
```

x86 **187 passed in 53.55s**; GPU arm **187 passed in 18.69s**. The population
is the previous 185 plus the two repair regressions, without discovery or
assertion weakening. The actual hook + complete W1 population + real producer +
shared verification path passes, including refusal after successful tuple removal.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"37cd6b6f66a5d2d1732804129a8828af9f4f192af26cd8cf3eaf1d4d426b385e","size":1946}},"operation":"86c410d1845743f5a061fd251ad18b05dba15642495cc298b68ef32d6200d456","receipt":"a207edd355d80cd1853a65335740aa9329265ac8948e65796025f485467e96de"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"d89048da59592e2f99dce40748da227fecc36c47020a7416b52a1426f4e088ac","size":1850}},"operation":"a62243249c844194312ed41c645a7c5187ef88c785513dd467d3d7f2b7901827","receipt":"222a32123fb961ac3363d4b79aa545646c588f6ea2cb60064830f8ddbf12d185"}]

### Focused serving rerun

Same focused serving command printed above: x86 **53 passed in 10.91s**; GPU
arm **53 passed in 3.53s** (previous 51 plus the two repair cases).

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"27d098e4e74a984d09e0673e658296ea39350ddbc4572790a23bd6e02c52c916","size":919}},"operation":"1cc1d2298f00b05ca6c7d5abb1c075f251a8f4912dcfc7159c3de2b39e357504","receipt":"2b441f93a70730c35ed5363c5f9921739be9a08fa797eefc58fa8dec9717f07e"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"4e7f8b032c4fc684470ef73e4bb9a1ba3a2ba18063be9dda06305918057a011f","size":714}},"operation":"8cd9492ee856f899876bd399b8b346aee6e6591f96864824fc7147bdb3f5fc28","receipt":"f832bd455e7d7f226f4ffa4f4c0d38ad1a340266cf895a5c7fffc5a28461f0fa"}]

No post-repair prescribed check failed or was unavailable. The completed initial
review's two findings are addressed by these changes and observed checks; this
is a repair disposition, not self-certification of approval. Final post-repair
separate-session approval remains a normal pipeline review action. No publication,
serving-default change, identity seal, expanded equality or compiled qualification
is performed or claimed.
