# Open implementation questions

- `rows.md:103` requires BF16 weight `2^-133` on each library. E4M3 and E2M1 cannot represent that weight.
  Proposed answer: run the operand case on the value library. Record the alphabet limit on the other libraries.
- `rows.md:6` says all rows wait for the freeze. No freeze record has arrived.
  Proposed answer: require the frozen rule hash and H1 receipt as explicit row inputs. Preparation can continue.
- `rows.md:103` asks for an injected error inside a Tessera launch, but the kernel must keep the cell hash.
  Proposed answer: inject an asynchronous device assertion inside the recorded launch boundary, without changing the cell binary.
- `rows.md:98` requires a 60-minute C1 job in each image. The code checks all alternate read tuples.
  Proposed answer: measure its time and peak memory before the lead accepts that job limit. No estimate is established.
- `rows.md:109-114` names checkpoint populations and canonical-cache additions, but gives no machine-readable module list.
  Proposed answer: the lead supplies one explicit population JSON per arm. The driver refuses an omitted manifest module.
- `pass-rule.md:56-65` requires separation of every element on every unit. `rows.md:79-84` limits some M values to columns only.
  Proposed answer: state whether full element separation at M=1,64,2048 can attach to a columns-only M unit.
- Tessera `AGENTS.md:239` requires `bbrun`. This checkout's host PATH has no `bbrun` executable.
  Proposed answer: provide the approved runner command on dl380g10. The CPU job records a missing command as a failed check.
