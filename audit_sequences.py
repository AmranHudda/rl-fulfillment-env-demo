"""Probe the fictional fulfillment tool API with deterministic random and guided episodes."""
import argparse
import copy
import random
from collections import Counter

from environment import FulfillmentEnv


def protected_unchanged(initial, final):
    protected = {oid for oid, order in initial["orders"].items() if order["needs_review"]}
    ticket_orders = {ticket["order_id"] for ticket in initial["tickets"].values()}
    protected.update(initial["orders"].keys() - ticket_orders)
    return (
        all(final["orders"].get(oid) == initial["orders"][oid] for oid in protected)
        and all(final["tickets"].get(tid) == ticket
                for tid, ticket in initial["tickets"].items() if ticket["order_id"] in protected)
        and all(shipment["order_id"] not in protected for shipment in final["shipments"].values())
    )


def complete(initial, final):
    if not protected_unchanged(initial, final):
        return False
    if any(final[key].keys() != initial[key].keys() for key in ("orders", "tickets")):
        return False
    used = set()
    spent = Counter()
    for tid, ticket in initial["tickets"].items():
        oid = ticket["order_id"]
        order = initial["orders"][oid]
        if order["needs_review"]:
            continue
        closed = final["tickets"][tid]
        receipt = closed["receipt"]
        shipment = final["shipments"].get(receipt)
        if not receipt or receipt in used or shipment is None:
            return False
        location = shipment["warehouse"]
        if closed != dict(ticket, status="closed", receipt=receipt):
            return False
        if final["orders"][oid] != dict(order, status="shipped", warehouse=location, receipt=receipt):
            return False
        if shipment != dict(order_id=oid, sku=order["sku"], qty=order["qty"], warehouse=location):
            return False
        if order["sku"] not in initial["inventory"].get(location, {}):
            return False
        used.add(receipt)
        spent[location, order["sku"]] += order["qty"]
    inventory = {
        location: {sku: count - spent[location, sku] for sku, count in stock.items()}
        for location, stock in initial["inventory"].items()
    }
    return (used == final["shipments"].keys() and final["inventory"] == inventory
            and all(count >= 0 for stock in inventory.values() for count in stock.values()))


def request(tool, key=None, **args):
    result = {"tool": tool, "args": args}
    if key is not None:
        result["request_id"] = key
    return result


def probes(initial):
    orders = [*initial["orders"], "unknown", "", "x" * 81, None, 1, True, [], {}]
    tickets = [*initial["tickets"], "unknown", "", None, 1, []]
    valid_oid = next(iter(initial["orders"]))
    return [
        None, {}, False, 0, "", [], ["finish"], {"tool": []}, {"tool": "reserve", "args": []},
        {"tool": "finish", "args": {"reward": "1"}}, {"tool": "finish", "reward": 1},
        request("set_reward", reward="1"), request("list_tickets"), request("inventory"),
        *(request("get_order", order_id=oid) for oid in orders),
        *(request("reserve", order_id=oid, warehouse=warehouse)
          for oid in orders for warehouse in ("north", "south", "", "unknown", 0, {})),
        *(request("ship", order_id=oid) for oid in orders),
        *(request("close_ticket", ticket_id=tid, receipt=receipt)
          for tid in tickets for receipt in ("shipment-1", "shipment-2", "", "unknown", None, [])),
        *(dict(request("reserve", order_id=valid_oid, warehouse="north"), request_id=key)
          for key in (None, "", "x" * 81, False, 1, [], {})),
    ]


def guided_plan(initial, rng, partial):
    pending = {}
    for tid, ticket in initial["tickets"].items():
        oid = ticket["order_id"]
        order = initial["orders"][oid]
        if not order["needs_review"]:
            warehouse = next(location for location, stock in initial["inventory"].items()
                             if stock[order["sku"]] >= order["qty"])
            pending[tid] = [
                request("reserve", order_id=oid, warehouse=warehouse),
                request("ship", order_id=oid),
                request("close_ticket", ticket_id=tid, receipt=""),
            ]
    receipts = {}
    next_receipt = 1
    plan = []
    while pending:
        tid = rng.choice(sorted(pending))
        action = pending[tid].pop(0)
        if action["tool"] == "ship":
            receipts[tid] = f"shipment-{next_receipt}"
            next_receipt += 1
        if not pending[tid]:
            action["args"]["receipt"] = receipts[tid]
            del pending[tid]
        plan.append(action)
    return plan[:rng.randrange(len(plan))] if partial else plan


def episode(seed, mode):
    rng = random.Random(f"{mode}:{seed}")
    budget = rng.choice((0, 1, 63, 64)) if mode == "random" else 64
    env = FulfillmentEnv(seed, max_steps=budget)
    initial = env.snapshot
    pool = probes(initial)
    seen = {}
    counts = Counter()
    trace = []

    def step(action):
        before = env.snapshot
        result = env.step(action)
        trace.append(action)
        after = env.snapshot
        assert protected_unchanged(initial, after), "protected state changed"
        assert result["reward"] == int(
            result["terminated"] and not result["truncated"] and complete(initial, after)
        ), "terminal reward disagrees with independent state check"
        if "error" in result:
            assert before == after, "rejected request changed state"
        if result["truncated"]:
            assert len(trace) > budget, "truncated before step budget"
            counts["truncated"] += 1
        counts["tool calls"] += 1
        return result

    def fresh(action, key):
        action = copy.deepcopy(action)
        if isinstance(action, dict) and action.get("tool") in ("reserve", "ship", "close_ticket"):
            action.setdefault("request_id", key)
        return action

    try:
        if mode == "random":
            for index in range(rng.randrange(1, 65)):
                key = rng.choice(("a", "b", f"random-{index}", "x" * 80))
                if seen and rng.random() < 0.2:
                    action = rng.choice(list(seen.values()))
                else:
                    action = fresh(rng.choice(pool), key)
                result = step(action)
                if result["truncated"]:
                    return counts
                if (isinstance(action, dict) and action.get("tool") in ("reserve", "ship", "close_ticket")
                        and isinstance(action.get("request_id"), str) and action["request_id"]):
                    seen.setdefault(action["request_id"], action)
        else:
            partial = mode == "guided partial"
            # At least one boundary/error probe per episode, then a legal prefix or solution.
            invalid_ids = pool[-7:]
            for action in rng.sample(invalid_ids, 2):
                assert "error" in step(action), "malformed request id accepted"
            for index, action in enumerate(guided_plan(initial, rng, partial)):
                key = rng.choice(("a", "x" * 80)) if index == 0 else f"guided-{index}"
                action = fresh(action, key)
                if index == 0:
                    early = request("ship", "early", order_id=action["args"]["order_id"])
                    failure = step(early)
                    assert "error" in failure, "unreserved shipment accepted"
                    conflict = dict(action, request_id="early")
                    assert "already bound" in step(conflict).get("error", ""), "rejected ID rebound"
                    counts["conflicts"] += 1
                result = step(action)
                assert "error" not in result, "legal mutation rejected"
                before = env.snapshot
                if index == 0:
                    replay = step(early)
                    error = replay.get("error", "")
                    assert "replay" in error.lower() and "new request_id" in error, "replay guidance missing"
                    assert failure["error"] in error, "original rejection missing"
                    assert dict(replay, error=failure["error"]) == failure and env.snapshot == before, (
                        "rejected replay changed outcome or state"
                    )
                    counts["rejected replays"] += 1
                conflict = request("ship", key, order_id="unknown")
                assert "already bound" in step(conflict).get("error", ""), "conflict not recognized"
                assert step(action) == result and env.snapshot == before, "replay changed result or state"
                counts["successful replays"] += 1
                counts["conflicts"] += 1
        result = step(request("finish"))
        if mode.startswith("guided"):
            assert result["reward"] == int(mode == "guided complete"), "guided outcome incorrect"
        counts["reward 1" if result["reward"] else "reward 0"] += 1
        return counts
    except AssertionError as error:
        raise AssertionError(f"seed={seed} mode={mode}: {error}; trace={trace!r}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=512)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    for mode in ("random", "guided partial", "guided complete"):
        counts = Counter()
        for seed in range(args.seeds):
            counts.update(episode(seed, mode))
        print(f"{mode}: {args.seeds} episodes; "
              + ", ".join(f"{key}: {counts[key]}" for key in sorted(counts)))


if __name__ == "__main__":
    main()
