# Grader audit

## Tool-call search

`python3 audit_sequences.py` runs three deterministic strategies on each seed
from 0 through 511. Each call is checked for protected-state changes, state
changes after rejection, and reward inconsistent with a separate end-state
check that follows ticket receipts and independently reconciles stock.

| Strategy | Episodes | Tool calls | Reward 1 | Reward 0 at finish | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random requests, small and normal budgets | 512 | 9,684 | 0 | 268 | 254 |
| Guided prefixes of valid interleavings, then finish | 512 | 6,930 | 0 | 512 | 0 |
| Guided complete interleavings | 512 | 12,288 | 512 | 0 | 0 |

The request pool includes malformed envelopes and argument types, unknown and
protected IDs, bad warehouses, fabricated or mismatched receipts, request ID
lengths 0, 1, 80 and 81, and repeated IDs across different payloads. Guided
episodes exercise exact replays of successful and rejected mutations, including
after a state change, and conflicts under the same ID. They recorded 4,435
successful-result replays, 947 rejection replays and 5,382 conflicts.

## Findings and fixes

- **Rejected mutations did not bind their ID.** A rejected reservation could be
  replaced by a different reservation under the same ID. A ship call rejected
  before reservation could also execute on exact replay after reservation.
  Well-formed mutations now cache their tool-level result, including rejection.
  Validation errors do not bind an ID, and conflicts do not replace a result.
  A replayed rejection preserves the original error and tells the actor to use
  a new `request_id` to retry.
- **A missing shipped order raised `KeyError` rather than grading false.**
  Deleting that order while keeping its shipment reproduces the failure. The
  shipment check now rejects a missing order. This was a controller-side
  fixture, not a missing order created through the tool API.

The regression tests were committed before their respective fixes. The search
did not find an actor tool sequence earning reward 1 for an incomplete task or
changing protected state. It covered the current four-order generator, not all
possible sequences, future variants, the JSON-lines parser, or direct access to
the controller process. Reads inform the actor; the final-state grader does not
require or inspect them. No models were run.

## Mutation testing

`python3 audit_grade.py` forces one `grade()` predicate at a time to true or
false in memory (whole branches, individual Boolean operands, the generator
predicate, and final success). It reports every survivor. A failure or
unexpected exception in a test counts as detection. Use `--original-tests`
to run the same experiment against only the 19 original tests.

The original tests caught 36 of 65 mutants on the original grader. Every
survivor below forced the indicated predicate to **false**. “Whole” means the
entire compound predicate was disabled, separately from disabling its operands.

| Check | Original survivors (29 total) |
| --- | --- |
| Truncation | `self._truncated` |
| Shipment completeness | `shipped != self._eligible` |
| Population | Order keys differ; ticket keys differ |
| Shipment eligibility and uniqueness | Whole check; order is ineligible; order is already in `shipped` |
| Shipment SKU and quantity | Whole check; SKU differs; quantity differs |
| Warehouse and SKU | Whole check; warehouse is missing; shipped SKU is absent from warehouse |
| Stock | Reconciled stock is negative |
| Shipped order consistency | Whole check; order status differs; order receipt differs; order warehouse differs |
| Order existence and protection | Later `final is None`; whole protected-order comparison; order is ineligible; protected order differs |
| Immutable metadata | Whole `any(...)`; generator comparison for an immutable field |
| Protected tickets | Protected ticket differs |
| Eligible ticket consistency | Whole check; order ID differs; status differs; receipt differs |

On the fixed grader, the original tests caught 37 of 67; the missing-order
fix adds one more surviving false-forced operand (`final is None` inside
the shipment check). The expanded 39-test suite catches 65 of 67, including
that new operand. The two survivors are redundant under single mutations:

- `oid in shipped -> False`: two shipment dictionary entries have distinct
  receipt keys; both cannot match the same final order's receipt.
- Later `final is None -> False`: every eligible order must ship and already
  passed the missing-order check in the shipment loop. A missing protected
  order differs from its original record and is rejected by the next check.

The mutation command exits 1 whenever any mutant survives, including these
two. The unit test and sequence search commands exit 0 on success.
