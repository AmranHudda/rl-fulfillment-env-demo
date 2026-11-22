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

Variant mutation commands now exit 0 if every surviving mutant has an exact
documented tool-invariant equivalence proof; any unexpected survivor or failed
baseline exits 1. They print caught, equivalent and unexpected counts separately,
not a zero-survivor claim. The historical mutation command above did not have
this allowlist. Fixtures use development seeds 7 and 11; no held-out seeds were
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

#### Cross-variant follow-up

The original per-variant survivor classification was too narrow: a predicate
unreachable in one variant can still be distinguished in another variant that
uses the same mutated function. Every run now includes the unchanged base/easy
and split/easy direct-state fixture suites, plus 24 tool-only trajectories:
the correct solver and careless actor for all six variants at easy and hard.
The oracle checks `grade()` after **every** tool call, including incomplete
prefixes, rejected moves and otherwise-valid protected mutations. New reachable
states are built entirely through the API; no controller setup or edited
snapshots are needed for these witnesses.

Eleven previously variant-local survivors have explicit reachable witnesses:
forcing the split-fact branch either way (two mutants), and forcing each of
the nine allocation predicates true. A completed base trajectory distinguishes
the forced-true split branch; a completed split trajectory distinguishes the
forced-false branch and the nine forced-true allocation checks. The regression
test runs the actual mutants against these tool trajectories. Existing direct
fixtures, retained rather than weakened, catch the remaining newly detected
mutants (disabled allocation checks, normal shipment completeness/eligibility,
population and early missing-order checks) across variants. These legacy
synthetic fixtures are not claimed to be reachable tool states.

| Variant (medium) | Before: caught / total; survivors | After: test cases | After: caught / total | Equivalent | Unexpected |
| --- | --- | ---: | ---: | ---: | ---: |
| base | 68/89; 21 | 954 | 84/89 | 5 | 0 |
| split | 82/89; 7 | 1,154 | 84/89 | 5 | 0 |
| duplicate | 68/89; 21 | 990 | 84/89 | 5 | 0 |
| stockout | 75/99; 24 | 1,008 | 94/99 | 5 | 0 |
| cancellation | 93/117; 24 | 1,004 | 112/117 | 5 | 0 |
| refund | 93/119; 26 | 990 | 112/119 | 7 | 0 |

Each medium run exits 0. The mutant catalog does not depend on difficulty:
the same static `grade`, `_grade_base` and variant-grade function objects supply
all predicates at easy/medium/hard/extreme and integer difficulties. A regression
test checks those identities across configurations; there is no different
easy/hard mutant set to enumerate. Reachable easy/hard witnesses are nevertheless
included in every medium run. No real reachable hole in the original grader
was found, so no grader logic was changed in this follow-up.

#### Remaining equivalent mutants: tool-invariant proofs

These seven allowlist entries apply across **all** variants, seeds and accepted
difficulties, not just the variant used to print a report. They concern one
predicate mutation at a time and reachable controller states. Induction starts
from generated worlds; the listed tool preconditions preserve each invariant.
Reads return copies, rejected mutations do not execute, and request-ID replay
returns a cached result rather than repeating a transition. No actor tool can
edit controller metadata or arbitrarily replace records.

| Function / predicate forced false | Why no reachable state distinguishes it |
| --- | --- |
| `_grade_base`: `oid in shipped` | `ship` requires reserved status and makes it shipped; no tool reopens an order, and pre-shipped refund orders cannot reserve, so shipment order IDs are always unique. |
| `_grade_base`: **later** `final is None` (occurrence 2) | No tool deletes orders or replaces a record with `None`, so every initial order always has a record. |
| `_grade_base`: `not allocations` | Split `ship` requires allocations summing to a positive immutable order quantity; shipped allocations therefore cannot be empty, and non-split variants never evaluate this predicate. |
| `_grade_base`: `final["warehouse"] is not None` | Split orders start with warehouse `None`; split reserve updates only allocations/status, ship preserves that warehouse, and no other split tool changes it. |
| `_grade_base`: `shipment["warehouse"] is not None` | Split ship copies the order's invariant `None` warehouse; no tool subsequently modifies a shipment. |
| `grade_refund`: `set(state["tickets"]) != set(initial["tickets"])` | No tool creates or deletes tickets, so their key sets cannot change. |
| `grade_refund`: `len(state["refunds"]) != len(refunds)` | Refund requires shipped status and makes it refunded with no reopening transition, so the ledger is injective in order ID; whenever the preceding value-set equality passes, its cardinality must equal the eligible-refund set's cardinality. |

The first five appear in all six runs; the last two additionally appear in
refund. `audit_grade.EQUIVALENT_MUTANTS` stores exactly these reasons, keyed by
function, predicate text, replacement constant and occurrence number. In
particular, the early missing-order check is **not** allowlisted by accidentally
matching the later predicate's text. New tests verify that distinction, exact
allowlist coverage, exit 0 for a documented equivalent, and exit 1 for an
undocumented survivor. An unrecognized surviving predicate is never silently
accepted. Sampling the trajectories checks the proofs' premises but is not
itself the proof of equivalence.

These results are sampled evidence, not a proof of robustness against arbitrary
malformed controller memory, multiple mutants, arbitrary policies, process
access, the JSON-lines parser or all seed/difficulty combinations. No model
performance has been measured; process isolation is still absent.
