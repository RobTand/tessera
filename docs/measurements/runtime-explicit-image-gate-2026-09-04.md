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
