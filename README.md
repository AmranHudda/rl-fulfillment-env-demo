# Fulfillment environment prototype

A small offline, in-memory tool environment for experiments with state-based
task verification. All orders, tickets and inventory are fictional.

This is a vertical slice, not a production RL platform, an OS sandbox, a trained
policy or evidence of model performance on long-horizon tasks.

## Task

Fulfill the tickets that do not require manual review. Read the order and stock,
reserve inventory, create a shipment and close the ticket with its real receipt.
Leave review-only tickets and unrelated orders unchanged.

The controller awards terminal reward only when eligible orders are shipped,
their tickets are closed with matching receipts, stock reconciles, and protected
state remains unchanged. An actor's claim of success is not a grading input.

## Run

Python standard library only. No credentials, network access or model calls.

```sh
python3 -m unittest -v
python3 demo.py
python3 environment.py 7
```

The last command is a JSON-lines interface. Its first response describes the task
and tools. Send requests such as:

```json
{"tool":"list_tickets","args":{}}
```

Mutations require a `request_id`. Once a well-formed mutation is attempted,
its ID is bound to the payload and its result, including a rejection by the
tool. Reusing the ID with the same request returns that result without
repeating the mutation; using it with another payload is rejected. To retry a
rejected mutation after the world changes, use a new ID. `finish` ends the
episode and returns the state-based reward.

## Task variants and difficulty

Defaults preserve the legacy base/easy world and observations.
`FulfillmentEnv(seed=0, max_steps=None, variant="base", difficulty="easy")`;
`reset(seed)` retains configuration and accepts keyword overrides.

| Variant | What it tests | Correct end state | Extra tools |
| --- | --- | --- | --- |
| `base` | Fulfillment and protected state | Eligible orders shipped; tickets closed with real receipts | None |
| `split` | Multi-warehouse allocations | Full allocations shipped; stock reconciled | `reserve` adds decimal-string `qty` |
| `duplicate` | Multiple tickets per order | Ship once; close every ticket with the same receipt | None |
| `stockout` | Limited-stock completion | Exactly one short-SKU order and ticket untouched; all other eligible work complete | None |
| `cancellation` | Mid-episode changes | Requested order cancelled, stock released, ticket closed; remaining eligible work complete | `cancel_order` |
| `refund` | Refund routing and review protection | Non-review refund tickets refunded and closed; fulfillment complete; protected orders unchanged | `refund` |

All variants require exact stock reconciliation and protect review/unrelated
state. Refunds do not restock inventory; refunding a protected order is
deliberately allowed by the tool but fails grading. Cancellation events arrive
before an action at a seeded step from 4 through
`max(5, (3 + tickets + 3 * eligible_orders) // 3)`; reserve/ship then refuse the
requested order. Close cancellation/refund tickets with their real operation
receipts, not shipment receipts.

| Difficulty | Orders |
| --- | ---: |
| `easy` | 4 |
| `medium` | 16 |
| `hard` | 32 |
| `extreme` | 64 |

An integer 4..64 is also accepted. About 25% of orders require review; some are
unrelated and protected. Base/easy defaults to 64 steps; all other configurations
use `8 * orders + 16`, leaving margin over the scripted solvers. Explicit
`max_steps` overrides this; `reset(seed, max_steps=None)` restores the default.

CLI: `python3 environment.py 7 --variant split --difficulty hard`
(or `--difficulty 24`). Omitting flags preserves the legacy interface.

Seed split: `DEV_SEEDS` 0..9999 is for development/tests;
`HELDOUT_SEEDS` 1,000,000..1,000,049 is reserved for evaluation. No test or solver
tuning uses held-out seeds; the constants do not enforce access restrictions.

## Evidence and limits

- 83 tests: 19 original tests and 19 PR #2 audit tests unchanged, plus 45 variant
  and integration tests.
- Each of the six variants has a scripted solver earning reward 1 on 100 dev
  seeds at every named difficulty and under alternative action orders, plus at
  least three grader-must-catch failure cases. Split also tests both warehouse
  reservation orders; cancellation is checked to fire in every tested solver
  episode.
- Worlds are solvable by construction. The scripted solvers know the rules:
  these checks show solvability and rejection of the listed failures, not an
  unbreakable grader. The [base-task audit](GRADER_AUDIT.md) sampled 1,536
  episodes without a false positive or protected-state change and detected
  65/67 predicate mutants (two redundant survivors). These historical audit
  results cover base only, not the variants. After the refactor, the unchanged
  sequence audit reproduces its base results; the unchanged mutation script
  catches 4/4 mutations of the thin grade-dispatch wrapper, not its delegated
  predicates. The merged suite also checks rejected-request binding and missing
  shipped orders across all variants. Reward depends on final state, not reads.
- Negative cases include fabricated receipts, missing/dangling reservations,
  protected orders, malformed requests, mutation replay, reset, exhausted step
  budgets and false completion claims. Trace snapshots retain past observations.
- Difficulty scales counts, not reasoning depth beyond that. No model runs have
  been performed and held-out seeds remain unused; these are not model
  performance or held-out benchmark results.
- `demo.py` prints tool transcripts from a scripted baseline and a negative
  control. These are explicitly not LLM or trained-policy rollouts.
- Actors are assumed to have only the listed tool API. Arbitrary code execution
  against the controller, process isolation and production persistence are not
  implemented.
- No real customer data, Fitscout code, private recordings, photos or credentials
  are included.
