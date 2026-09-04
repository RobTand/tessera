# Integrated serve ownership and wrapper checks — 2026-09-04

This is targeted CPU integration evidence, not the final two-population merge
suite and not CUDA or served-quality evidence. Tessera #5 and #113 remain open
while their campaigns finish.

The measured source was `894305ee41b648e563bc1fa672948f5f6cf8c4af`, combining
master, the reviewed #113 controller, the qualified MoE export/census work,
teacher cleanup, and the two legacy-lock corrections. The merge retained the
#113 PID/start-time/nonce and serialized reaper protocol; it did not restore
the older MoE branch's mkdir recovery implementation. Both branches' distinct
build-identity tests were retained.

## PrismaBuild receipt

- Action: `7fe43a33cfe8ea2658c2c266aa97afe714d0b5f5820fbb892d951b578a411037`
- Receipt: `bc46f876d9cbe86b1788004b495cd2a45463b9a4a49b7f17ecb653b835c3dd19`
- Materialized snapshot: `e4b71dd2a3d59699eb748ef5d46a50d5ccc54126`
- Host/device: dl380g10, x86_64, torch `2.11.0+cpu`, no CUDA device.
- Mode/reservation: serial pytest, 2 CPUs / 4 GiB, CUDA masked by PrismaBuild.
- Result: **67 passed, 9 skipped, 0 uncollected modules**, pytest elapsed 13.06 s.
- Population: `/mnt/shared/tessera-runs/integration/astra-894305ee/surface.json`
- Selection: `/mnt/shared/tessera-runs/integration/astra-894305ee/selection.json`

The selector compared the exact fetched
`d5297a20093074c1ba1f7743c8850c93a19a01c0` tree with the parentless snapshot
and returned `narrowed`: `test_serve_lock.py` and
`test_serve_wrapper_cleanup.py`. The coordinator additionally ran
`test_serve_build_identity.py` and `test_runtime_image_pin.py` to check the
resolved wrapper merge. An earlier submission stopped before tests because
the snapshot had no `origin`; the successful submission fetched the exact
base through an explicit shared Git bundle.

Skip reasons, verbatim:

| Count | Reason |
| ---: | --- |
| 6 | `/home/rob/tessera-runs/compile-dispatch is not on this box` |
| 1 | `the two surviving compile caches from 2026-09-02 are not on this box` |
| 1 | `/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box` |
| 1 | `/home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box` |

## Separate pre-fix evidence

The legacy-directory publication regression failed in PrismaBuild
`49e2850ce76e`: `ln -s` successfully published *inside* the existing legacy
directory, so an acquire incorrectly returned 0 beside its live owner.
Commit `c27b66c` uses `ln -sT` at both publication sites. The independent
permission regression then failed in `ced3e8901c79`: denied `kill -0` permission
caused a live legacy owner to be removed (`assert 0 == 3`). Commit `f6920ef`
checks `/proc` presence instead. Both fixes were cherry-picked separately into
the integration branch as `61f7743` and `d77f8b6`.

The final whole-tree CPU/CUDA suite is still owed on the completed merge
result. This targeted run does not substitute for it.
