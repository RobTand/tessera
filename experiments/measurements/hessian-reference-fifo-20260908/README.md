# Nonblocking Hessian reference refusal — Tessera #428 follow-up

Fix commit `3e35660b01f0f90b1b8297ca2f6f6cdfdf53e63a`, based on
`615e1c0302693447a9cd713bb1f9dddc21fbd378`. `_HeldFile` previously opened a path
with `O_RDONLY | O_NOFOLLOW` before checking the descriptor's regular-file
status. A FIFO could block that open indefinitely. The fix adds `O_NONBLOCK`
and retains the same `fstat`, regular-file, byte-cap and pathname checks.
Ordinary regular-file reads are unchanged.

The two new tests replace either the reference metadata path or a canonical
H input path with a FIFO. Each starts an owned child process, proves it reached
reference intake, and requires a regular-file refusal within a five-second
subprocess bound. Timeout kills and reaps that child. No FIFO writer or helper
process is used. The green run also executes the existing real regular-file
reader test, covering accepted metadata followed by verified H access.

| PrismaBuild action | Outcome |
|---|---|
| `5a05614e788003c47ae54fc3b2ec885f808186feb97295d4c2b49bdb37b48007` | Regression-first: 2 expected failures in 11.67 s; both FIFO opens reached intake and blocked until their five-second timeout. |
| `67f2568d87832cc670ddf86e6751aa995c0e933c92bc753a5e8ecf9f3bd3440d` | Fixed: 3 passed, 0 skipped, 0 modules missing, 2.70 s; no CUDA allocation. |

Both ran through published `pbrun.py` on the x86 tooling class (`dl380g10`),
with one CPU, 3 GiB memory, 60-second action timeout, one attempt, priority -10,
`PYTHONPATH=src`, and OMP/MKL/OPENBLAS threads set to one. Interpreter:
`/home/rob/venvs/pb-cpu/bin/python` (Torch 2.11.0+cpu). The red command was
`-m pytest -q tests/test_hessian_reference_fifo.py`; the green command appended
`tests/test_hessian_reference_capture.py::test_public_reader_binds_commitments_without_loading_h_then_verifies_access`.
No broad suite or native experiment was rerun.

The red and green snapshots carry identical regression test blob
`b47e7d12a3ee54d00d4666ce239aeb81c0be2d5d`. Red snapshot
`468a8c0a08cd25202a48a545d6c6a08f98e1190a` differs from the prior producer only
in the new test and PB closure metadata. Green snapshot
`b63a850ae9d3f8ea687f5ce978f548f1d54e466a` differs from the fix commit only in
PB closure metadata. Actual source-bundle and log bytes/hashes were checked.
Both terminal scopes were completely released with no OOM and no live process.
The green CAS receipt's canonical hash and its actual 385-byte result payload
hash were independently checked. Receipt:
`b31a7286ed66687a5d5f8b6cf700e65e2390432d9dacf98940256a7f5ddc06f3`;
result `25ec6f5602497de755bebf90cbf197edd3d1efb2ad92949418150482e8e5f22f`;
claim `cf783a7a2e4dc33b9a6f40e6518f08fc9d64a02681bbd9ec5bafd0798508b46b`.
The receipt was initially absent from the shared path despite publication in
stdout, then became visible and passed verification; no rerun or repair was
performed. This observation does not establish the cause of that delay.

Checked JSON files retain exact original log paths and hashes. Checked stdout
excerpts omit the trailing worker receipt JSON and trim line-end spaces.
These are bounded CPU refusal tests, not native timing or serving measurements.
