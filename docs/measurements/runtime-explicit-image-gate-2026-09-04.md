# Explicit runtime images are verified, not merely stamped

The #126 census now requires an exact image context. During that change,
review found the existing image helper gated only the default repository.
An explicit EUGR-style reference outside it could pass without matching
`RepoDigests`, and a local image with multiple manifest aliases could stamp
the first alias rather than the requested manifest.

`runtime_image.resolve` now verifies every explicit digest reference against
the daemon's `RepoDigests` and prefers that exact requested digest in its
stamp. `.Id` remains provenance only. The old dense default and its repository
policy do not change; unrelated floating-tag images remain merely stamped
and cannot supply an exact-runtime census context. Missing or mismatched
explicit references return the requested `docker pull` as their remedy.

PrismaBuild red action
`c1c51d61f1a21d9f387b2b47928d914778ddcfdfe538053dba1f084b8059ab72`
ran `tests/test_runtime_image_pin.py`: **5 failed, 13 passed in 1.21s**.
Every added case failed before the fix:

```text
test_an_explicit_digest_outside_the_default_repository_is_verified
tests/test_runtime_image_pin.py:189: AssertionError: assert False is True
test_an_explicit_digest_cannot_borrow_another_images_stamp[False]
test_an_explicit_digest_cannot_borrow_another_images_stamp[True]
tests/test_runtime_image_pin.py:198: Failed: DID NOT RAISE RuntimeImageError
test_explicit_digest_stamp_does_not_depend_on_repository_digest_order
tests/test_runtime_image_pin.py:212: AssertionError
test_explicit_requested_digest_is_stamped_even_when_the_default_pin_is_an_alias
tests/test_runtime_image_pin.py:219: AssertionError
```

Green action
`01ff4a01e74ff8b91ce3ede3e195b5f3a9d41b1ae58ee2260e0d4a7f8f27c98e`
ran the same file: **18 passed in 1.24s**, test return code zero. Both actions
used dl380g10, serial CPU, and recorded the test return code separately from
the successful result-publication wrapper to prevent automatic retries from
repeating a red test. Green result CAS:
`52f835b684c6cf71f009cc3f365d815e1f7a81520addd0abe4cf1283cdbdc436`.

Both populations:

```text
tessera surface: NO CUDA -- torch 2.11.0+cpu reports no CUDA device
tessera surface: 0 test(s) skipped, 0 module(s) not collected
tessera surface: this run did not exercise the CUDA-gated surface. Its pass count is not coverage of it.
```

No model bytes, encoder profile, default image, or quality threshold changed.
The active LFM encoders remain on their original frozen source; their Docker
commands were already exact digest references, so this provenance/gating fix
does not require re-encoding.

## Selected dependency coverage

Selector action
`48079ce7e8a72ffea4fa4a5fafaf56c1a03676aec8d1f5c4a3680bce7e55aa7c`
compared exact fetched base `24f042704fc6ed9abb6addcca247ee6160f98c88`
against the parentless snapshot of `a34de85`: `narrowed`, 28 test files.
The selector explicitly reported its direct tree comparison because the
snapshot has no merge base. Initial selector attempts used an unavailable
snapshot `origin` remote; fetching the exact base from the repository URL
resolved that dispatch error without a source change.

The already-green 18-test image file was not repeated. The other 27 files
ran in action
`149bfa41831a839c47984b8b14f209cb7c091873a2edc1cab47be8d60ab67b29`:
**403 passed, 291 skipped, 0 modules uncollected**, serial CPU on dl380g10,
torch `2.11.0+cpu`, 220.35 seconds, test return code zero. This is not CUDA
coverage. Result CAS:
`76edc0880a8cf06e90aed623a4b2546bd19aecce6d09b22793930ab1d39e6521`.
Verbatim skip reasons:

```text
81  the encoder is a CUDA path
79  needs a CUDA device
29  the Viterbi is CUDA
24  the kernel lane is a CUDA path
23  the lane is a CUDA kernel
14  the Tessera encoder is a CUDA path
10  the encoder is a GPU job
 6  needs CUDA
 6  /home/rob/tessera-runs/compile-dispatch is not on this box
 5  the kernel lane runs on CUDA
 5  the reach checkpoint is not on this box
 3  Qwen3-0.6B is not on this box
 2  no stock twin
 1  the two surviving compile caches from 2026-09-02 are not on this box
 1  /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2.log is not on this box
 1  /home/rob/tessera-runs/stock/serve_qwen_stock_tessera-k2-graph.log is not on this box
 1  the shipped checkpoint is not here
```
