"""Offline RL environment vertical slice. Fictional data; no network or models.

An actor has a small tool API, not shell access to the controller. This is an
in-memory prototype, not an OS sandbox or a production training platform.
"""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import random
import sys

DEV_SEEDS = range(0, 10000)
HELDOUT_SEEDS = range(1_000_000, 1_000_050)
DIFFICULTIES = {"easy": 4, "medium": 16, "hard": 32, "extreme": 64}
_KEEP = object()


def order_count(difficulty):
    if isinstance(difficulty, str) and difficulty in DIFFICULTIES:
        return DIFFICULTIES[difficulty]
    if type(difficulty) is int and 4 <= difficulty <= 64:
        return difficulty
    raise ValueError("difficulty must be easy, medium, hard, extreme or an integer 4..64")


VARIANTS = {}


class TaskError(Exception):
    pass


class FulfillmentEnv:
    MUTATIONS = {"reserve", "ship", "close_ticket"}
    ARGUMENTS = {
        "list_tickets": set(), "inventory": set(),
        "get_order": {"order_id"},
        "reserve": {"order_id", "warehouse"},
        "ship": {"order_id"},
        "close_ticket": {"ticket_id", "receipt"},
        "finish": set(),
    }

    def __init__(self, seed=0, max_steps=None, *, variant="base", difficulty="easy"):
        self._max_steps_override = max_steps
        self.variant, self.difficulty = variant, difficulty
        self.reset(seed)

    def _legacy_world(self, seed):
        rng = random.Random(seed)
        offset = rng.randrange(1000, 9000)
        ids = ["order-%s" % (offset + i) for i in range(4)]
        orders = {}
        for oid, sku, qty, review in zip(
            ids, ["tee", "jacket", "jacket", "tee"], [2, 1, 1, 1],
            [False, False, True, False],
        ):
            orders[oid] = dict(sku=sku, qty=qty, needs_review=review,
                               status="open", warehouse=None, receipt=None)
        tickets = {
            "ticket-%s" % (offset + i): dict(order_id=ids[i], status="open", receipt=None)
            for i in range(3)
        }
        self._state = dict(
            orders=orders, tickets=tickets, shipments={},
            inventory={"north": {"tee": 3 + rng.randrange(3), "jacket": 0},
                       "south": {"tee": 0, "jacket": 2 + rng.randrange(3)}},
        )
        return self._state

    def reset(self, seed=0, *, max_steps=_KEEP, variant=None, difficulty=None):
        variant = self.variant if variant is None else variant
        difficulty = self.difficulty if difficulty is None else difficulty
        count = order_count(difficulty)
        if variant not in VARIANTS:
            raise ValueError("Unknown variant: %s" % variant)
        self.variant, self.difficulty = variant, difficulty
        if max_steps is not _KEEP:
            self._max_steps_override = max_steps
        self._spec = VARIANTS[variant](self, seed, count)
        self._state = self._spec["state"]
        self._facts = self._spec["facts"]
        self._arguments = copy.deepcopy(self._spec["tools"])
        self._mutations = set(self._spec["mutations"])
        self._initial = copy.deepcopy(self._state)
        self._eligible = set(self._facts["eligible"])
        self._max_steps = (self._max_steps_override if self._max_steps_override is not None
                           else 64 if variant == "base" and count == 4 else 8 * count + 16)
        self._seen_requests = {}
        self._steps = 0
        self._done = self._truncated = False
        return {
            "task": self._spec["task"],
            "tools": {name: sorted(args) for name, args in self._arguments.items()},
            "max_steps": self.max_steps,
        }

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, value):
        self._max_steps = self._max_steps_override = value

    @property
    def snapshot(self):
        return copy.deepcopy(self._state)

    def _order(self, oid, mutate=False):
        if oid not in self._state["orders"]:
            raise TaskError("Unknown order")
        if mutate and self.variant == "base" and oid not in self._eligible:
            raise TaskError("Order is not authorized for automatic fulfillment")
        return self._state["orders"][oid]

    def _run(self, tool, args):
        if tool == "list_tickets":
            return copy.deepcopy(self._state["tickets"])
        if tool == "inventory":
            return copy.deepcopy(self._state["inventory"])
        if tool == "get_order":
            return copy.deepcopy(self._order(args["order_id"]))
        if tool == "reserve":
            order = self._order(args["order_id"], mutate=True)
            warehouse = self._state["inventory"].get(args["warehouse"])
            if warehouse is None:
                raise TaskError("Unknown warehouse")
            if order["status"] != "open":
                raise TaskError("Order is not open")
            if warehouse.get(order["sku"], 0) < order["qty"]:
                raise TaskError("Insufficient inventory")
            warehouse[order["sku"]] -= order["qty"]
            order.update(status="reserved", warehouse=args["warehouse"])
            return {"reserved": args["order_id"]}
        if tool == "ship":
            order = self._order(args["order_id"], mutate=True)
            if order["status"] != "reserved":
                raise TaskError("Order must be reserved before shipping")
            receipt = "shipment-%s" % (len(self._state["shipments"]) + 1)
            self._state["shipments"][receipt] = dict(
                order_id=args["order_id"], sku=order["sku"], qty=order["qty"],
                warehouse=order["warehouse"],
            )
            order.update(status="shipped", receipt=receipt)
            return {"receipt": receipt}
        if tool == "close_ticket":
            ticket = self._state["tickets"].get(args["ticket_id"])
            if ticket is None:
                raise TaskError("Unknown ticket")
            order = self._order(ticket["order_id"], mutate=True)
            shipment = self._state["shipments"].get(args["receipt"])
            if (order["status"] != "shipped" or shipment is None
                    or shipment["order_id"] != ticket["order_id"]
                    or order["receipt"] != args["receipt"]):
                raise TaskError("A matching real shipment receipt is required")
            if ticket["status"] != "open":
                raise TaskError("Ticket is already closed")
            ticket.update(status="closed", receipt=args["receipt"])
            return {"closed": args["ticket_id"]}
        raise TaskError("Unknown tool")

    def step(self, request):
        events = []
        if (not self._done and not self._truncated and self._steps < self.max_steps
                and self._spec.get("event")):
            events = self._spec["event"](self, self._steps + 1)
        result = self._step(request)
        if events:
            result["events"] = events
        return result

    def _step(self, request):
        if self._done or self._truncated:
            return {"error": "Episode has ended", "reward": 0, "terminated": self._done,
                    "truncated": self._truncated}
        self._steps += 1
        if self._steps > self.max_steps:
            self._truncated = True
            return {"error": "Step budget exceeded", "reward": 0,
                    "terminated": False, "truncated": True}
        record_request = False
        try:
            if not isinstance(request, dict) or set(request) - {"tool", "args", "request_id"}:
                raise TaskError("Invalid request envelope")
            tool, args = request.get("tool"), request.get("args", {})
            if not isinstance(tool, str) or tool not in self._arguments:
                raise TaskError("Unknown tool")
            if not isinstance(args, dict) or set(args) != self._arguments[tool]:
                raise TaskError("Invalid tool arguments")
            if any(not isinstance(value, str) for value in args.values()):
                raise TaskError("Arguments must be strings")
            if tool == "finish":
                reward = int(self.grade())
                self._done = True
                return {"observation": {"complete": bool(reward)}, "reward": reward,
                        "terminated": True, "truncated": False}
            key = request.get("request_id")
            fingerprint = hashlib.sha256(json.dumps({"tool": tool, "args": args},
                                                       sort_keys=True).encode()).hexdigest()
            if tool in self._mutations:
                if not isinstance(key, str) or not 1 <= len(key) <= 80:
                    raise TaskError("Mutation requires a request_id of 1 to 80 characters")
                if key in self._seen_requests:
                    previous, result = self._seen_requests[key]
                    if previous != fingerprint:
                        raise TaskError("request_id was already bound to another payload")
                    return copy.deepcopy(result)
                record_request = True
            result = {"observation": self._spec["run"](self, tool, args), "reward": 0,
                      "terminated": False, "truncated": False}
        except TaskError as error:
            result = {"error": str(error), "reward": 0,
                      "terminated": False, "truncated": False}
        if record_request:
            self._seen_requests[key] = fingerprint, copy.deepcopy(result)
        return result

    def grade(self):
        """Controller-side oracle: final state, never the actor's success claim."""
        if self._truncated:
            return False
        return self._spec["grade"](self)

    def _grade_base(self, state=None, initial=None, eligible=None):
        state = self._state if state is None else state
        initial = self._initial if initial is None else initial
        eligible = self._eligible if eligible is None else eligible
        expected_stock = copy.deepcopy(initial["inventory"])
        shipped = set()
        for receipt, shipment in state["shipments"].items():
            oid = shipment["order_id"]
            if oid not in eligible or oid in shipped:
                return False
            original = initial["orders"][oid]
            if shipment["sku"] != original["sku"] or shipment["qty"] != original["qty"]:
                return False
            final = state["orders"].get(oid)
            if final is None:
                return False
            if self._facts.get("split"):
                allocations = shipment.get("allocations")
                if (not isinstance(allocations, dict) or not allocations
                        or any(type(qty) is not int or qty <= 0 for qty in allocations.values())
                        or sum(allocations.values()) != original["qty"]
                        or final.get("allocations") != allocations
                        or final["warehouse"] is not None or shipment["warehouse"] is not None):
                    return False
            else:
                allocations = {shipment["warehouse"]: shipment["qty"]}
            for name, qty in allocations.items():
                warehouse = expected_stock.get(name)
                if warehouse is None or shipment["sku"] not in warehouse:
                    return False
                warehouse[shipment["sku"]] -= qty
                if warehouse[shipment["sku"]] < 0:
                    return False
            if (final["status"] != "shipped" or final["receipt"] != receipt
                    or final["warehouse"] != shipment["warehouse"]):
                return False
            shipped.add(oid)
        if shipped != eligible or state["inventory"] != expected_stock:
            return False
        for oid, original in initial["orders"].items():
            final = state["orders"].get(oid)
            if final is None:
                return False
            if oid not in eligible and final != original:
                return False
            if any(final[k] != original[k] for k in ("sku", "qty", "needs_review")):
                return False
        if set(state["orders"]) != set(initial["orders"]):
            return False
        if set(state["tickets"]) != set(initial["tickets"]):
            return False
        for tid, original in initial["tickets"].items():
            final = state["tickets"][tid]
            if original["order_id"] not in eligible:
                if final != original:
                    return False
            elif (final["order_id"] != original["order_id"] or final["status"] != "closed"
                  or final["receipt"] != state["orders"][original["order_id"]]["receipt"]):
                return False
        return True


def generate_base(env, seed, count):
    if count == 4:
        state = env._legacy_world(seed)
    else:
        rng = random.Random(seed)
        offset = rng.randrange(1000, 9000)
        skus = ["tee", "jacket", "scarf", "cap"]
        inventory = {"north": {sku: rng.randrange(1, 4) for sku in skus},
                     "south": {sku: 0 for sku in skus}}
        orders, tickets = {}, {}
        reviews = set(rng.sample(range(count - 2), count // 4))
        for i in range(count):
            oid = "order-%s" % (offset + i)
            sku, qty = rng.choice(skus), rng.randrange(1, 4)
            orders[oid] = dict(sku=sku, qty=qty, needs_review=i in reviews,
                               status="open", warehouse=None, receipt=None)
            if i < count - 2:
                tickets["ticket-%s" % (offset + i)] = dict(
                    order_id=oid, status="open", receipt=None)
                if i not in reviews:
                    inventory["north"][sku] += qty
        for sku in skus[1::2]:
            inventory["south"][sku], inventory["north"][sku] = inventory["north"][sku], 0
        state = dict(orders=orders, tickets=tickets, inventory=inventory, shipments={})
    eligible = {t["order_id"] for t in state["tickets"].values()
                if not state["orders"][t["order_id"]]["needs_review"]}
    if env.variant != "base":
        review = next(o for o in state["orders"].values() if o["needs_review"])
        review.update(sku="review-token", qty=1)
        state["inventory"]["north"]["review-token"] = 1
        state["inventory"]["south"]["review-token"] = 0
    return dict(state=state, facts={"eligible": eligible},
                task="Fulfill and close each ticket whose order does not need manual review. "
                     "Use available inventory. Leave review-only tickets and unrelated orders unchanged. "
                     "Finish only when the work is complete.",
                tools=FulfillmentEnv.ARGUMENTS, mutations=FulfillmentEnv.MUTATIONS,
                run=FulfillmentEnv._run, grade=FulfillmentEnv._grade_base)


VARIANTS["base"] = generate_base


def run_split(env, tool, args):
    if tool == "reserve":
        order = env._order(args["order_id"], mutate=True)
        raw = args["qty"]
        significant = raw.lstrip("0")
        if (not raw.isascii() or not raw.isdecimal() or not significant
                or len(significant) > len(str(order["qty"]))):
            raise TaskError("qty must be a positive decimal integer within remaining quantity")
        qty = int(significant)
        remaining = order["qty"] - sum(order["allocations"].values())
        warehouse = env._state["inventory"].get(args["warehouse"])
        if order["status"] not in {"open", "partially_reserved"}:
            raise TaskError("Order is not open for reservation")
        if qty > remaining:
            raise TaskError("Quantity exceeds remaining unreserved quantity")
        if warehouse is None:
            raise TaskError("Unknown warehouse")
        if warehouse.get(order["sku"], 0) < qty:
            raise TaskError("Insufficient inventory")
        warehouse[order["sku"]] -= qty
        allocations = order["allocations"]
        allocations[args["warehouse"]] = allocations.get(args["warehouse"], 0) + qty
        order["status"] = "reserved" if qty == remaining else "partially_reserved"
        return {"reserved": args["order_id"], "qty": qty}
    if tool == "ship":
        order = env._order(args["order_id"], mutate=True)
        if sum(order["allocations"].values()) != order["qty"]:
            raise TaskError("Order must be fully reserved before shipping")
        result = env._run(tool, args)
        env._state["shipments"][result["receipt"]]["allocations"] = copy.deepcopy(order["allocations"])
        return result
    return env._run(tool, args)


def generate_split(env, seed, count):
    spec = generate_base(env, seed, count)
    state = spec["state"]
    rng = random.Random(seed)
    inventory = {"north": {}, "south": {}}
    eligible = sorted(spec["facts"]["eligible"])
    split_orders = set(rng.sample(eligible, max(1, len(eligible) // 3)))
    for i, (oid, order) in enumerate(state["orders"].items()):
        order["allocations"] = {}
        sku = "parcel-%s" % i
        order["sku"] = sku
        if oid in split_orders:
            north, south = rng.randint(1, 3), rng.randint(1, 3)
            order["qty"] = north + south
        else:
            north, south = order["qty"] + rng.randint(1, 3), 0
            if i % 2:
                north, south = south, north
        inventory["north"][sku], inventory["south"][sku] = north, south
    state["inventory"] = inventory
    spec["facts"]["split"] = True
    spec["tools"] = copy.deepcopy(spec["tools"])
    spec["tools"]["reserve"] = {"order_id", "warehouse", "qty"}
    spec["run"] = run_split
    spec["task"] += (" Some orders require splitting across north and south. Reserve a positive "
                     "decimal-string qty per warehouse, up to the order's remaining quantity. "
                     "Ship only after allocations sum to the full order quantity.")
    return spec


def generate_duplicate(env, seed, count):
    spec = generate_base(env, seed, count)
    state = spec["state"]
    rng = random.Random(seed)
    eligible = sorted(spec["facts"]["eligible"])
    duplicated = rng.sample(eligible, max(1, len(eligible) // 3))
    review = next(t["order_id"] for t in state["tickets"].values()
                  if state["orders"][t["order_id"]]["needs_review"])
    for oid in duplicated + [review]:
        state["tickets"]["duplicate-" + oid] = dict(order_id=oid, status="open", receipt=None)
    spec["task"] += (" Multiple tickets may refer to the same order. Ship each eligible order "
                     "exactly once and close every associated ticket with that same real receipt. "
                     "Leave all review-only duplicates unchanged.")
    return spec


VARIANTS.update(split=generate_split, duplicate=generate_duplicate)


def generate_stockout(env, seed, count):
    spec = generate_base(env, seed, count)
    state = spec["state"]
    rng = random.Random(seed)
    ids = list(state["orders"])
    reviews = set(rng.sample(ids[3:], count // 4))
    unrelated = sorted(reviews)[-1]
    state["tickets"] = {"ticket-" + oid.removeprefix("order-"): dict(
        order_id=oid, status="open", receipt=None) for oid in ids if oid != unrelated}
    for oid, order in state["orders"].items():
        order["needs_review"] = oid in reviews
    eligible = set(ids) - reviews
    spec["facts"]["eligible"] = eligible
    scale = rng.randint(1, 5)
    for oid, qty in zip(ids[:3], (3 * scale, scale, 3 * scale)):
        state["orders"][oid].update(sku="limited-pin", qty=qty)
    inventory = {"north": {}, "south": {}}
    for i, (oid, order) in enumerate(state["orders"].items()):
        sku = order["sku"]
        for stock in inventory.values():
            stock.setdefault(sku, 0)
        if sku != "limited-pin":
            inventory["south" if i % 2 else "north"][sku] += order["qty"]
    for stock in inventory.values():
        stock["limited-pin"] = 3 * scale
    state["inventory"] = inventory
    spec["facts"]["short_sku"] = "limited-pin"
    quantities = [state["orders"][oid]["qty"] for oid in ids[:3]]
    def optimum(index, north, south):
        if index == len(quantities):
            return 0
        qty = quantities[index]
        choices = [optimum(index + 1, north, south)]
        if north >= qty:
            choices.append(qty + optimum(index + 1, north - qty, south))
        if south >= qty:
            choices.append(qty + optimum(index + 1, north, south - qty))
        return max(choices)
    spec["facts"]["optimum"] = optimum(0, 3 * scale, 3 * scale)
    spec["grade"] = grade_stockout
    spec["task"] = ("Ship as many units as possible of the limited-pin SKU; each order must ship "
                    "from one warehouse, never split. Fulfill all other non-review tickets. Leave "
                    "unfulfilled orders completely untouched and their tickets open. Never leave "
                    "reservations unshipped. Leave review and unrelated state unchanged.")
    return spec


def grade_stockout(env):
    shipped = {s["order_id"] for s in env._state["shipments"].values()}
    short = {oid for oid in env._eligible
             if env._initial["orders"][oid]["sku"] == env._facts["short_sku"]}
    if (not env._eligible - short <= shipped
            or sum(env._initial["orders"][oid]["qty"] for oid in short & shipped) != env._facts["optimum"]):
        return False
    return env._grade_base(eligible=env._eligible & shipped)


def close_special(env, args, status, ledger):
    ticket = env._state["tickets"].get(args["ticket_id"])
    if ticket is None:
        raise TaskError("Unknown ticket")
    order = env._order(ticket["order_id"])
    if (ticket["status"] != "open" or order["status"] != status
            or order["receipt"] != args["receipt"]
            or env._state[ledger].get(args["receipt"]) != ticket["order_id"]):
        raise TaskError("A matching real %s receipt is required" % status)
    ticket.update(status="closed", receipt=args["receipt"])
    return {"closed": args["ticket_id"]}


def generate_cancellation(env, seed, count):
    spec = generate_base(env, seed, count)
    rng = random.Random(seed)
    priority = sorted(spec["facts"]["eligible"])
    rng.shuffle(priority)
    solver_steps = 3 + len(spec["state"]["tickets"]) + 3 * len(priority)
    spec["facts"].update(cancel_step=rng.randint(4, max(5, solver_steps // 3)),
                         priority=priority, cancelled_target=None)
    for order in spec["state"]["orders"].values():
        order["cancel_requested"] = False
    for ticket in spec["state"]["tickets"].values():
        ticket["cancel_requested"] = False
    spec["state"]["cancellations"] = {}
    spec["tools"] = dict(spec["tools"], cancel_order={"order_id"})
    spec["mutations"] = spec["mutations"] | {"cancel_order"}
    spec.update(run=run_cancellation, grade=grade_cancellation, event=cancellation_event)
    spec["task"] += (" A cancellation may arrive in a response's events. Do not reserve or ship "
                     "cancel_requested orders. Use cancel_order to release any reservation and close "
                     "its ticket with the real cancellation receipt; fulfill the remaining orders.")
    return spec


def cancellation_event(env, step):
    if step != env._facts["cancel_step"]:
        return []
    oid = next((oid for oid in env._facts["priority"]
                if env._state["orders"][oid]["status"] != "shipped"), None)
    if oid is None:
        return []
    env._facts["cancelled_target"] = oid
    env._state["orders"][oid]["cancel_requested"] = True
    for ticket in env._state["tickets"].values():
        if ticket["order_id"] == oid:
            ticket["cancel_requested"] = True
    return [{"type": "cancellation", "order_id": oid}]


def run_cancellation(env, tool, args):
    if tool in {"reserve", "ship", "cancel_order"}:
        order = env._order(args["order_id"], mutate=True)
        if tool != "cancel_order" and order["cancel_requested"]:
            raise TaskError("Cancellation requested; cancel this order instead")
        if tool == "cancel_order":
            if not order["cancel_requested"] or order["status"] not in {"open", "reserved"}:
                raise TaskError("Order is not awaiting cancellation")
            if order["status"] == "reserved":
                env._state["inventory"][order["warehouse"]][order["sku"]] += order["qty"]
            receipt = "cancel-%s" % (len(env._state["cancellations"]) + 1)
            env._state["cancellations"][receipt] = args["order_id"]
            order.update(status="cancelled", warehouse=None, receipt=receipt)
            return {"receipt": receipt}
    if tool == "close_ticket":
        ticket = env._state["tickets"].get(args["ticket_id"])
        if ticket and ticket["cancel_requested"]:
            return close_special(env, args, "cancelled", "cancellations")
    return env._run(tool, args)


def grade_cancellation(env):
    state, initial = copy.deepcopy(env._state), copy.deepcopy(env._initial)
    if (set(state["orders"]) != set(initial["orders"])
            or set(state["tickets"]) != set(initial["tickets"])):
        return False
    target = env._facts["cancelled_target"]
    expected = {}
    if target is not None:
        order = state["orders"][target]
        if order != dict(initial["orders"][target], cancel_requested=True, status="cancelled",
                         receipt=order["receipt"]):
            return False
        receipt = order["receipt"]
        expected[receipt] = target
        if not isinstance(receipt, str) or not receipt.startswith("cancel-"):
            return False
        state["orders"][target] = copy.deepcopy(initial["orders"][target])
    if state["cancellations"] != expected:
        return False
    for tid, ticket in state["tickets"].items():
        original = initial["tickets"][tid]
        if original["order_id"] == target:
            if ticket != dict(original, cancel_requested=True, status="closed", receipt=receipt):
                return False
            state["tickets"][tid] = copy.deepcopy(original)
    for oid, order in state["orders"].items():
        if order["cancel_requested"] != initial["orders"][oid]["cancel_requested"]:
            return False
    for tid, ticket in state["tickets"].items():
        if ticket["cancel_requested"] != initial["tickets"][tid]["cancel_requested"]:
            return False
    return env._grade_base(state, initial, env._eligible - {target})


def generate_refund(env, seed, count):
    spec = generate_base(env, seed, count)
    state = spec["state"]
    rng = random.Random(seed)
    eligible = sorted(spec["facts"]["eligible"])
    refunds = set(rng.sample(eligible, max(1, len(eligible) // 3)))
    review = next(oid for oid, o in state["orders"].items() if o["needs_review"])
    related = {t["order_id"] for t in state["tickets"].values()}
    unrelated = next(oid for oid in state["orders"] if oid not in related)
    pre_shipped = refunds | {review, unrelated}
    for i, oid in enumerate(sorted(pre_shipped)):
        order = state["orders"][oid]
        receipt = "initial-shipment-%s" % i
        order.update(status="shipped", receipt=receipt, warehouse="north")
        state["shipments"][receipt] = dict(order_id=oid, sku=order["sku"], qty=order["qty"], warehouse="north")
    for ticket in state["tickets"].values():
        ticket["kind"] = "refund" if ticket["order_id"] in pre_shipped else "fulfill"
    state["refunds"] = {}
    spec["facts"]["refund_eligible"] = refunds
    spec["tools"] = dict(spec["tools"], refund={"order_id"})
    spec["mutations"] = spec["mutations"] | {"refund"}
    spec.update(run=run_refund, grade=grade_refund)
    spec["task"] = ("Process each non-review ticket according to kind: fulfill open orders normally; "
                    "refund already-shipped orders for refund tickets and close with the refund receipt. "
                    "Leave review-only tickets/orders and unrelated orders unchanged. Do not refund "
                    "anything else. Refunds do not restock inventory.")
    return spec


def run_refund(env, tool, args):
    if tool == "refund":
        order = env._order(args["order_id"])
        if order["status"] != "shipped":
            raise TaskError("Order must be shipped and not already refunded")
        receipt = "refund-%s" % (len(env._state["refunds"]) + 1)
        env._state["refunds"][receipt] = args["order_id"]
        order.update(status="refunded", receipt=receipt)
        return {"receipt": receipt}
    if tool == "close_ticket":
        ticket = env._state["tickets"].get(args["ticket_id"])
        if ticket and ticket["kind"] == "refund":
            return close_special(env, args, "refunded", "refunds")
    return env._run(tool, args)


def grade_refund(env):
    state, initial = copy.deepcopy(env._state), copy.deepcopy(env._initial)
    if (set(state["orders"]) != set(initial["orders"])
            or set(state["tickets"]) != set(initial["tickets"])):
        return False
    refunds = env._facts["refund_eligible"]
    if any(state["tickets"].get(tid, {}).get("kind") != ticket["kind"]
           for tid, ticket in initial["tickets"].items()):
        return False
    if (set(state["refunds"].values()) != refunds or len(state["refunds"]) != len(refunds)
            or any(not isinstance(r, str) or not r.startswith("refund-") for r in state["refunds"])):
        return False
    for receipt, oid in state["refunds"].items():
        if state["orders"][oid] != dict(initial["orders"][oid], status="refunded", receipt=receipt):
            return False
        for tid, original in initial["tickets"].items():
            if original["order_id"] == oid:
                if state["tickets"][tid] != dict(original, status="closed", receipt=receipt):
                    return False
                state["tickets"][tid] = copy.deepcopy(original)
        state["orders"][oid] = copy.deepcopy(initial["orders"][oid])
    for receipt, shipment in initial["shipments"].items():
        if state["shipments"].get(receipt) != shipment:
            return False
        del state["shipments"][receipt]
    return env._grade_base(state, initial, env._eligible - refunds)


VARIANTS.update(stockout=generate_stockout, cancellation=generate_cancellation, refund=generate_refund)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", nargs="?", type=int, default=0)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--difficulty", default="easy")
    options = parser.parse_args()
    difficulty = (int(options.difficulty) if options.difficulty.isdecimal()
                  else options.difficulty)
    env = FulfillmentEnv(options.seed, variant=options.variant, difficulty=difficulty)
    print(json.dumps(env.reset(options.seed)), flush=True)
    for line in sys.stdin:
        try:
            result = env.step(json.loads(line))
        except (ValueError, TypeError):
            result = {"error": "Invalid JSON", "reward": 0}
        print(json.dumps(result), flush=True)
        if result.get("terminated") or result.get("truncated"):
            break
