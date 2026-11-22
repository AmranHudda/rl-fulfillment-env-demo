"""Independent final-state oracle and deterministic probes for task variants."""
import copy
import itertools
import random
import unittest
from collections import Counter

import audit_sequences
from environment import FulfillmentEnv
from solvers import (_solve, solve_base, solve_split, solve_duplicate, solve_stockout,
                     solve_cancellation, solve_refund)

SOLVERS = dict(base=solve_base, split=solve_split, duplicate=solve_duplicate,
               stockout=solve_stockout, cancellation=solve_cancellation, refund=solve_refund)


def complete(variant, initial, final):
    """Follow receipts and reconstruct allowed final records, without grade()."""
    if any(final[key].keys() != initial[key].keys() for key in ("orders", "tickets")):
        return False
    related = {t["order_id"] for t in initial["tickets"].values()}
    eligible = {oid for oid in related if not initial["orders"][oid]["needs_review"]}
    inventory = copy.deepcopy(initial["inventory"])
    shipments = copy.deepcopy(initial["shipments"])
    expected_ledgers = {name: {} for name in ("refunds", "cancellations") if name in initial}
    requested = set()
    short_units = 0
    for oid, original in initial["orders"].items():
        order = final["orders"][oid]
        tickets = {tid: t for tid, t in initial["tickets"].items() if t["order_id"] == oid}
        if oid not in eligible:
            if order != original or any(final["tickets"][tid] != t for tid, t in tickets.items()):
                return False
            continue
        is_refund = variant == "refund" and any(t["kind"] == "refund" for t in tickets.values())
        is_cancel = variant == "cancellation" and order.get("cancel_requested")
        receipt = order["receipt"]
        if is_refund or is_cancel:
            status, ledger, prefix = (("refunded", "refunds", "refund-") if is_refund
                                      else ("cancelled", "cancellations", "cancel-"))
            changes = dict(status=status, receipt=receipt)
            if is_cancel:
                requested.add(oid)
                changes.update(cancel_requested=True, warehouse=None)
            if (not isinstance(receipt, str) or not receipt.startswith(prefix)
                    or order != dict(original, **changes)):
                return False
            expected_ledgers[ledger][receipt] = oid
            for tid, ticket in tickets.items():
                changes = dict(status="closed", receipt=receipt)
                if is_cancel:
                    changes["cancel_requested"] = True
                if final["tickets"][tid] != dict(ticket, **changes):
                    return False
            continue
        if variant == "stockout" and original["sku"] == "limited-pin" and order == original:
            if any(final["tickets"][tid] != t for tid, t in tickets.items()):
                return False
            continue
        if order["status"] != "shipped" or receipt in shipments:
            return False
        shipment = final["shipments"].get(receipt)
        if not receipt or shipment is None:
            return False
        changes = dict(status="shipped", receipt=receipt, warehouse=order["warehouse"])
        expected = dict(order_id=oid, sku=original["sku"], qty=original["qty"], warehouse=order["warehouse"])
        if variant == "split":
            allocations = order["allocations"]
            if (not allocations or order["warehouse"] is not None
                    or any(type(q) is not int or q <= 0 for q in allocations.values())
                    or sum(allocations.values()) != original["qty"]):
                return False
            changes["allocations"] = allocations
            expected["allocations"] = allocations
        else:
            allocations = {order["warehouse"]: original["qty"]}
        if order != dict(original, **changes) or shipment != expected:
            return False
        for location, qty in allocations.items():
            if original["sku"] not in inventory.get(location, {}):
                return False
            inventory[location][original["sku"]] -= qty
        shipments[receipt] = expected
        if original["sku"] == "limited-pin":
            short_units += original["qty"]
        if any(final["tickets"][tid] != dict(t, status="closed", receipt=receipt)
               for tid, t in tickets.items()):
            return False
    if len(requested) > 1:
        return False
    if variant == "stockout":
        quantities = [initial["orders"][oid]["qty"] for oid in eligible
                      if initial["orders"][oid]["sku"] == "limited-pin"]
        warehouses = list(initial["inventory"])
        optimum = 0
        for allocation in itertools.product(range(-1, len(warehouses)), repeat=len(quantities)):
            if all(sum(q for q, w in zip(quantities, allocation) if w == i)
                   <= initial["inventory"][name]["limited-pin"] for i, name in enumerate(warehouses)):
                optimum = max(optimum, sum(q for q, w in zip(quantities, allocation) if w >= 0))
        if short_units != optimum:
            return False
    return (final["shipments"] == shipments and final["inventory"] == inventory
            and all(q >= 0 for stock in inventory.values() for q in stock.values())
            and all(final[name] == ledger for name, ledger in expected_ledgers.items()))


def episode(seed, mode, variant, difficulty):
    rng = random.Random(f"{variant}:{difficulty}:{mode}:{seed}")
    budget = rng.choice((0, 1, 63, 1024)) if mode == "random" else None
    env = FulfillmentEnv(seed, max_steps=budget, variant=variant, difficulty=difficulty)
    initial = env.snapshot
    counts = Counter()
    history = []
    step = env.step
    def checked(action):
        before = env.snapshot
        result = step(action)
        after = env.snapshot
        history.append(action)
        assert result["reward"] == int(result["terminated"] and not result["truncated"]
                                       and complete(variant, initial, after)), (seed, mode, history)
        if "error" in result and not result.get("events"):
            assert before == after, ("rejected request changed state", seed, history)
        counts["tool calls"] += 1
        counts["events"] += len(result.get("events", []))
        if result["truncated"]:
            counts["truncated"] += 1
        return result
    if mode == "random":
        pool = audit_sequences.probes(initial)
        for action in pool:
            if isinstance(action, dict) and action.get("tool") == "reserve" and variant == "split":
                if isinstance(action.get("args"), dict):
                    action["args"]["qty"] = rng.choice(("1", "0", "-1", "2", "bad"))
        for oid in initial["orders"]:
            pool += [audit_sequences.request("refund", order_id=oid),
                     audit_sequences.request("cancel_order", order_id=oid)]
        for index in range(rng.randint(1, 64)):
            action = copy.deepcopy(rng.choice(pool))
            if isinstance(action, dict):
                action.setdefault("request_id", rng.choice(("reused", "x" * 80, str(index))))
            result = checked(action)
            if result["terminated"] or result["truncated"]:
                break
        else:
            result = checked(audit_sequences.request("finish"))
    else:
        class PrefixEnd(Exception):
            pass
        limit = rng.randint(1, 5) if mode == "guided partial" else None
        class Actor:
            def step(self, action):
                if limit is not None and len(history) >= limit:
                    raise PrefixEnd
                return checked(action)
        try:
            result = SOLVERS[variant](Actor(), shuffle_seed=seed)
        except PrefixEnd:
            result = checked(audit_sequences.request("finish"))
        assert result["reward"] == int(mode == "guided complete"), (seed, variant, mode)
    counts["reward 1" if result["reward"] else "reward 0"] += 1
    return counts


def fixtures(variant, difficulty):
    """Valid final states, invalid known-field mutations, and truncation cases."""
    for seed in (7, 11):
        env = FulfillmentEnv(seed, variant=variant, difficulty=difficulty)
        SOLVERS[variant](env, finish=False)
        initial, final = env._initial, env.snapshot
        yield f"seed={seed} complete", env, final, False, True
        yield f"seed={seed} truncated", env, final, True, False
        yield f"seed={seed} unfinished", env, copy.deepcopy(initial), False, False
        def leaves(value, path=()):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from leaves(child, path + (key,))
            else:
                yield path, value
        for path, value in leaves(final):
            altered = copy.deepcopy(final)
            parent = altered
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = (not value if type(value) is bool else value + 1
                                if type(value) is int else "invalid")
            if not complete(variant, initial, altered):
                yield f"seed={seed} change {path}", env, altered, False, False
        for category in ("orders", "tickets", "shipments", "inventory"):
            for key in final[category]:
                altered = copy.deepcopy(final)
                del altered[category][key]
                yield f"seed={seed} delete {category}/{key}", env, altered, False, False
            altered = copy.deepcopy(final)
            altered[category]["extra"] = copy.deepcopy(next(iter(final[category].values())))
            yield f"seed={seed} extra {category}", env, altered, False, False
        def case(description, state, original=None):
            controller = copy.copy(env)
            controller._initial = copy.deepcopy(initial if original is None else original)
            return f"seed={seed} {description}", controller, state, False, False
        for receipt, shipment in final["shipments"].items():
            if receipt in initial["shipments"]:
                continue
            oid, sku = shipment["order_id"], shipment["sku"]
            altered = copy.deepcopy(final)
            for warehouse, qty in shipment.get("allocations", {shipment["warehouse"]: shipment["qty"]}).items():
                altered["inventory"][warehouse][sku] += qty
            del altered["shipments"][receipt]
            yield case("missing shipment with reconciled stock", altered)
            altered = copy.deepcopy(final)
            altered["orders"][oid]["receipt"] = "invented"
            for ticket in altered["tickets"].values():
                if ticket["order_id"] == oid:
                    ticket["receipt"] = "invented"
            yield case("matching invented ticket/order receipt", altered)
            if variant == "split":
                for invalid in ("sum", "zero", "float", "unknown", "sparse", "negative", "list", "sku", "stock"):
                    altered, original = copy.deepcopy(final), copy.deepcopy(initial)
                    allocations = copy.deepcopy(shipment["allocations"])
                    if invalid == "sum":
                        location = next(iter(allocations))
                        allocations[location] += 1
                    elif invalid == "float":
                        allocations = {w: float(q) for w, q in allocations.items()}
                    elif invalid in ("zero", "negative"):
                        allocations = {"north": 0 if invalid == "zero" else -1,
                                       "south": shipment["qty"] + (invalid == "negative")}
                    elif invalid == "unknown":
                        allocations = {"unknown": shipment["qty"]}
                    elif invalid == "list":
                        altered["shipments"][receipt]["allocations"] = [1]
                        altered["orders"][oid]["allocations"] = [1]
                        yield case("split non-dictionary allocations", altered)
                        continue
                    altered["shipments"][receipt]["allocations"] = allocations
                    altered["orders"][oid]["allocations"] = copy.deepcopy(allocations)
                    for w in ("north", "south"):
                        original["inventory"][w][sku] = shipment["qty"] + 3
                        altered["inventory"][w][sku] = original["inventory"][w][sku] - allocations.get(w, 0)
                    if invalid == "sparse":
                        location = next(iter(allocations))
                        del original["inventory"][location][sku]
                        del altered["inventory"][location][sku]
                    elif invalid == "sku":
                        altered["shipments"][receipt]["sku"] = "fixture-sku"
                        for location, qty in allocations.items():
                            original["inventory"][location]["fixture-sku"] = qty + 1
                            altered["inventory"][location]["fixture-sku"] = 1
                            altered["inventory"][location][sku] += qty
                    elif invalid == "stock":
                        location = next(iter(allocations))
                        original["inventory"][location][sku] = allocations[location] - 1
                        altered["inventory"][location][sku] = -1
                    yield case("split " + invalid, altered, original)
            else:
                location = shipment["warehouse"]
                for field in ("sku", "qty"):
                    altered, original = copy.deepcopy(final), copy.deepcopy(initial)
                    if field == "qty":
                        altered["shipments"][receipt]["qty"] += 1
                        original["inventory"][location][sku] += 2
                        altered["inventory"][location][sku] += 1
                    else:
                        new_sku = "fixture-sku"
                        original["inventory"][location][new_sku] = shipment["qty"] + 1
                        altered["inventory"][location][new_sku] = 1
                        altered["inventory"][location][sku] += shipment["qty"]
                        altered["shipments"][receipt]["sku"] = new_sku
                    yield case("wrong " + field + " with reconciled stock", altered, original)
                altered, original = copy.deepcopy(final), copy.deepcopy(initial)
                del original["inventory"][location][sku]
                del altered["inventory"][location][sku]
                yield case("warehouse without shipped sku", altered, original)
                altered, original = copy.deepcopy(final), copy.deepcopy(initial)
                other = "south" if location == "north" else "north"
                altered["shipments"][receipt]["warehouse"] = other
                altered["orders"][oid]["warehouse"] = other
                altered["inventory"][location][sku] += shipment["qty"]
                original["inventory"][other][sku] = 0
                altered["inventory"][other][sku] = -shipment["qty"]
                yield case("negative reconciled inventory", altered, original)
        for ledger, status in (("cancellations", "cancelled"), ("refunds", "refunded")):
            for receipt, oid in final.get(ledger, {}).items():
                for replacement in ("invented", 123):
                    altered = copy.deepcopy(final)
                    del altered[ledger][receipt]
                    altered[ledger][replacement] = oid
                    altered["orders"][oid]["receipt"] = replacement
                    for ticket in altered["tickets"].values():
                        if ticket["order_id"] == oid:
                            ticket["receipt"] = replacement
                    yield case("coherent fake " + status + " receipt", altered)
        if variant == "refund":
            altered = copy.deepcopy(final)
            oid = next(iter(altered["refunds"].values()))
            altered["refunds"]["refund-extra"] = oid
            yield case("duplicate refund ledger entry", altered)
        if variant == "stockout":
            greedy = FulfillmentEnv(seed, variant=variant, difficulty=difficulty)
            _solve(greedy, reverse=False, shuffle_seed=None, finish=False, stockout=True)
            yield "suboptimal first-fit allocation", greedy, greedy.snapshot, False, False
            for receipt, shipment in final["shipments"].items():
                if shipment["sku"] == "limited-pin":
                    continue
                altered = copy.deepcopy(final)
                oid = shipment["order_id"]
                del altered["shipments"][receipt]
                altered["orders"][oid] = copy.deepcopy(initial["orders"][oid])
                altered["inventory"][shipment["warehouse"]][shipment["sku"]] += shipment["qty"]
                for tid, ticket in initial["tickets"].items():
                    if ticket["order_id"] == oid:
                        altered["tickets"][tid] = copy.deepcopy(ticket)
                yield case("unfulfilled normal order with no other error", altered)
        if variant == "cancellation":
            no_event = FulfillmentEnv(seed, variant=variant, difficulty=difficulty)
            no_event._facts["cancel_step"] = 10 ** 6
            SOLVERS[variant](no_event, finish=False)
            yield "no cancellation", no_event, no_event.snapshot, False, True


class FinalStateCase(unittest.TestCase):
    def __init__(self, description, env, state, truncated, expected):
        super().__init__()
        self.description, self.env = description, env
        self.state, self.truncated, self.expected = state, truncated, expected

    def shortDescription(self):
        return self.description

    def runTest(self):
        self.env._state = copy.deepcopy(self.state)
        self.env._truncated = self.truncated
        self.assertEqual(self.env.grade(), self.expected, self.description)


def suite(variant, difficulty):
    return unittest.TestSuite(FinalStateCase(*case) for case in fixtures(variant, difficulty))
