# W2 owner outcomes implementation evidence

Scope: instrumentation portion of #1259/#1260 (overlap #1271), based on
ff95111ee8fed33e9a4ea23e59c46444cd8fd61b. This piece changes only the real
import observer and serving accounting; admission, ARCHITECTURE.md, combined O3
validation and separate-session completed-contract review belong to the normal
pipeline/admission piece. Integration is **not yet claimed** by these results.
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

## Final action checks on this piece

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
