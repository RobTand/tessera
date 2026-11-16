# Open implementation questions

- `rows.md:103` requires BF16 weight `2^-133` on each library. E4M3 and E2M1 cannot represent that weight.
  Proposed answer: run the operand case on the value library. Record the alphabet limit on the other libraries.
- `rows.md:6` says all rows wait for the freeze. The heading still says draft 5.
  Proposed answer: require the frozen rule hash and H1 receipt as explicit row inputs. Preparation can continue.
