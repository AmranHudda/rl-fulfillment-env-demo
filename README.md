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

Mutations require a `request_id`. Reusing it with the same request returns the
same result without repeating the mutation. Reusing it with another payload is
rejected. `finish` ends the episode and returns the state-based reward.

## Evidence and limits

- 19 tests, including scripted solutions across 100 seeds and immutable trace
  snapshots.
- Different valid action orders are accepted.
- Negative cases cover fabricated receipts, missing reservations, protected
  orders, malformed requests, mutation replay, reset, exhausted step budgets and
  false completion claims.
- `demo.py` prints tool transcripts from a scripted baseline and a negative
  control. These are explicitly not LLM or trained-policy rollouts.
- Actors are assumed to have only the listed tool API. Arbitrary code execution
  against the controller, process isolation and production persistence are not
  implemented.
- Seeds vary identifiers and stock counts; they do not yet establish broad task
  diversity or a held-out difficulty benchmark.
- No real customer data, Fitscout code, private recordings, photos or credentials
  are included.
