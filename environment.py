"""Offline RL environment vertical slice. Fictional data; no network or models.

An actor has a small tool API, not shell access to the controller. This is an
in-memory prototype, not an OS sandbox or a production training platform.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys


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

    def __init__(self, seed=0, max_steps=64):
        self.max_steps = max_steps
        self.reset(seed)

    def reset(self, seed=0):
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
        self._initial = copy.deepcopy(self._state)
        self._eligible = {t["order_id"] for t in tickets.values()
                          if not orders[t["order_id"]]["needs_review"]}
        self._seen_requests = {}
        self._steps = 0
        self._done = self._truncated = False
        return {
            "task": "Fulfill and close each ticket whose order does not need manual review. "
                    "Use available inventory. Leave review-only tickets and unrelated orders unchanged. "
                    "Finish only when the work is complete. "
                    "Mutations require a request_id of 1 to 80 characters. "
                    "Reusing it with the same tool and arguments replays the prior outcome without "
                    "running the mutation again; using it with a different payload is rejected. "
                    "Use a new request_id to retry a rejected mutation.",
            "tools": {name: sorted(args) for name, args in self.ARGUMENTS.items()},
            "max_steps": self.max_steps,
        }

    @property
    def snapshot(self):
        return copy.deepcopy(self._state)

    def _order(self, oid, mutate=False):
        if oid not in self._state["orders"]:
            raise TaskError("Unknown order")
        if mutate and oid not in self._eligible:
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
            if not isinstance(tool, str) or tool not in self.ARGUMENTS:
                raise TaskError("Unknown tool")
            if not isinstance(args, dict) or set(args) != self.ARGUMENTS[tool]:
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
            if tool in self.MUTATIONS:
                if not isinstance(key, str) or not 1 <= len(key) <= 80:
                    raise TaskError("Mutation requires a request_id of 1 to 80 characters")
                if key in self._seen_requests:
                    previous, result = self._seen_requests[key]
                    if previous != fingerprint:
                        raise TaskError("request_id was already bound to another payload")
                    if "error" in result:
                        raise TaskError("Replay of an earlier rejected mutation. "
                                        "Use a new request_id to retry. Original error: " + result["error"])
                    return copy.deepcopy(result)
                record_request = True
            result = {"observation": self._run(tool, args), "reward": 0,
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
        state, initial = self._state, self._initial
        expected_stock = copy.deepcopy(initial["inventory"])
        shipped = set()
        for receipt, shipment in state["shipments"].items():
            oid = shipment["order_id"]
            if oid not in self._eligible or oid in shipped:
                return False
            original = initial["orders"][oid]
            if shipment["sku"] != original["sku"] or shipment["qty"] != original["qty"]:
                return False
            warehouse = expected_stock.get(shipment["warehouse"])
            if warehouse is None or shipment["sku"] not in warehouse:
                return False
            warehouse[shipment["sku"]] -= shipment["qty"]
            if warehouse[shipment["sku"]] < 0:
                return False
            final = state["orders"].get(oid)
            if (final is None or final["status"] != "shipped" or final["receipt"] != receipt
                    or final["warehouse"] != shipment["warehouse"]):
                return False
            shipped.add(oid)
        if shipped != self._eligible or state["inventory"] != expected_stock:
            return False
        for oid, original in initial["orders"].items():
            final = state["orders"].get(oid)
            if final is None:
                return False
            if oid not in self._eligible and final != original:
                return False
            if any(final[k] != original[k] for k in ("sku", "qty", "needs_review")):
                return False
        if set(state["orders"]) != set(initial["orders"]):
            return False
        if set(state["tickets"]) != set(initial["tickets"]):
            return False
        for tid, original in initial["tickets"].items():
            final = state["tickets"][tid]
            if original["order_id"] not in self._eligible:
                if final != original:
                    return False
            elif (final["order_id"] != original["order_id"] or final["status"] != "closed"
                  or final["receipt"] != state["orders"][original["order_id"]]["receipt"]):
                return False
        return True


if __name__ == "__main__":
    env = FulfillmentEnv(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(json.dumps(env.reset(int(sys.argv[1]) if len(sys.argv) > 1 else 0)), flush=True)
    for line in sys.stdin:
        try:
            result = env.step(json.loads(line))
        except (ValueError, TypeError):
            result = {"error": "Invalid JSON", "reward": 0}
        print(json.dumps(result), flush=True)
        if result.get("terminated") or result.get("truncated"):
            break
