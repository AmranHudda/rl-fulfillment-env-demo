# Grader audit

The first three sections record PR #2's historical **base/easy** audit. The
variant extension and current commands/results follow below.

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
the shipment check). The expanded 38-test suite catches 65 of 67, including
that new operand. The two survivors are redundant under single mutations:

- `oid in shipped -> False`: two shipment dictionary entries have distinct
  receipt keys; both cannot match the same final order's receipt.
- Later `final is None -> False`: every eligible order must ship and already
  passed the missing-order check in the shipment loop. A missing protected
  order differs from its original record and is rejected by the next check.

The mutation command exits 1 whenever any mutant survives, including these
two. The unit test and sequence search commands exit 0 on success.

## Six-variant extension

`audit_sequences.py --variant NAME --difficulty LEVEL` and
`audit_grade.py --variant NAME --difficulty LEVEL` accept all six variants and
the four named difficulties. Omitting flags retains the original sequence
strategy/output and mutation harness. After the grader refactor, the default
mutation command tests the thin dispatch wrapper (4/4 caught); use `--variant`
to include `_grade_base` and the stockout/cancellation/refund grader where
applicable. Historical 65/67 results above concern the original monolithic
grader, not that wrapper.

Reproduction (stdlib only):

```sh
python3 -m unittest -v
python3 audit_sequences.py
python3 audit_grade.py
for variant in base split duplicate stockout cancellation refund; do
  python3 audit_sequences.py --variant "$variant" --difficulty easy
  python3 audit_sequences.py --variant "$variant" --difficulty hard
  python3 audit_grade.py --variant "$variant" --difficulty medium
done
```

Mutation commands still exit 1 for documented survivors; do not mistake that
for a successful zero-survivor run. Each baseline fixture suite passed before
mutations. The fixtures use development seeds 7 and 11; no held-out seeds were
used. `test_variant_audit.py` also exercises the independent oracle and the
three sequence strategies for all four named difficulties.

### Sequence search

For **each** variant at easy and hard, 512 random, 512 guided-partial and 512
guided-complete episodes were checked (18,432 episodes total). All complete
episodes earned 1, all partial episodes earned 0, and no reward disagreed with
the independent final-state oracle in `audit_variants.py`. Base/easy retains
the original replay/conflict probes and exactly reproduces the table above.

Hard results (counts printed by the extended command):

| Variant | Random calls / truncated | Partial calls | Complete calls | Complete reward 1 |
| --- | ---: | ---: | ---: | ---: |
| base | 8,844 / 274 | 2,109 | 50,688 | 512/512 |
| split | 8,816 / 265 | 2,028 | 54,272 | 512/512 |
| duplicate | 9,033 / 262 | 2,021 | 58,368 | 512/512 |
| stockout | 8,393 / 286 | 2,014 | 52,736 | 512/512 |
| cancellation | 9,443 / 252 | 2,002 | 60,433 | 512/512 |
| refund | 8,754 / 268 | 2,003 | 47,104 | 512/512 |

Hard cancellation episodes observed 207 random, 15 partial and 512 complete
event deliveries. Protected mutations are now deliberately possible in
non-base tasks, so the oracle checks that such states cannot earn reward, not
that tools prevent them. Rejections without controller events must not change
world state. Random probes include malformed requests, IDs, quantities,
receipts and repeated request IDs; extended guided prefixes are short (1–5
calls), while complete trajectories use shuffled scripted solvers. This is not
exhaustive interleaving or replay coverage for the new variants.

### Findings and regression coverage

- PR #2's rejected-mutation request-ID binding and missing-shipped-order fixes
  survive the rebase. Regression tests cover both on every variant. The
  original 19 tests and PR #2's 19 audit tests were not edited.
- No additional real false-positive hole was found by the extended searches
  or final-state probes. Mutation survivors exposed fixture gaps, not evidence
  of an exploitable original grader: coordinated split SKU/stock/type probes,
  a valid but suboptimal stockout allocation, and an otherwise-correct
  stockout state omitting a normal order now detect their respective mutants.
- The error-ignoring tool-only `careless_actor` gets 0/100 reward-1 episodes on
  **each** of split/duplicate/stockout/cancellation/refund at easy and hard
  (0/1,000 total). Separate tests show otherwise-valid protected mutations
  actually execute before grading 0; base's authorization guards remain.
- Stockout's three short-SKU quantities are `(3s, s, 3s)`, with capacity `3s`
  in each warehouse and seeded integer `s` in 1..5. Exhaustive generation-time
  search gives optimum `6s` (the two big orders). Sorted and reversed ticket
  first-fit both consume one big order then the small order, yielding `4s`;
  smallest-first also yields `4s`. Those first three orders are always eligible,
  and their IDs sort in generation order. Extra orders use other SKUs, so the
  guarantee holds for every seed and every accepted order count, not just the
  sampled seeds. The solver independently plans from observable quantities and
  stock, without controller facts. This is a fixed small packing trap, not a
  growing combinatorial challenge at higher difficulty.

### Predicate mutations

Each run mutates one predicate at a time, including individual Boolean
operands, generator predicates and final success returns. Unexpected exceptions
count as detection, just as in PR #2. Known-field mutations, removed/extra
records, coherent bad receipt/stock/allocation states, incomplete work and
truncation are tested against the unmutated grader first.

| Variant (medium) | Final-state fixtures | Caught / total | Survived |
| --- | ---: | ---: | ---: |
| base | 590 | 68/89 | 21 |
| split | 790 | 82/89 | 7 |
| duplicate | 626 | 68/89 | 21 |
| stockout | 644 | 75/99 | 24 |
| cancellation | 640 | 93/117 | 24 |
| refund | 626 | 93/119 | 26 |

All survivors are accounted for below; “equivalent” is relative to the given
variant's controller facts and **one** mutation, not simultaneous deletion of
redundant guards. Names refer to predicates, not unstable source line numbers.

1. **All variants (two):** `oid in shipped -> False` and the later
   `final is None -> False` are redundant for the receipt/existence reasons
   given in the historical audit.
2. **All non-split variants (19):** forcing the split-fact branch false (one)
   or either constant in its allocation-only checks (18) cannot change
   execution: that branch is unreachable in these variants. This plus (1)
   explains all 21 base/duplicate survivors.
3. **Split (five more):** forcing the split-fact branch true is equivalent to
   its private constant. `not allocations -> False` is redundant with the
   positive original quantity and allocation-sum check. The three individual
   warehouse comparisons are redundant: both order/shipment warehouses must
   be `None` and must equal each other, so dropping any one comparison leaves
   the other two. Together with (1), these explain all seven split survivors.
4. **Stockout (three more):** disabling the delegated base shipment-set check,
   its eligibility operand, or its whole eligibility/uniqueness condition is
   redundant here. The wrapper passes `eligible & shipped`, making that set
   equality tautological; an ineligible shipment's final order cannot both
   match the shipment and remain unchanged under protected-order checking.
   Duplicate shipments still fail receipt consistency. The wrapper's optimum
   and normal-order-completeness predicates are both detected.
5. **Cancellation/refund (three more each):** the delegated order/ticket
   population checks and early shipment missing-order check are redundant:
   each wrapper already checks the full populations before delegation.
6. **Refund (two more):** its wrapper ticket-population check is also
   redundant with the delegated ticket-population check (single mutation).
   Its refund-ledger length check is redundant because distinct ledger receipt
   keys cannot both match one order's final receipt in the following loop.

These results are sampled evidence, not a proof of robustness against arbitrary
malformed controller memory, multiple mutants, arbitrary policies, process
access, the JSON-lines parser or all seed/difficulty combinations. No model
performance has been measured; process isolation is still absent.
