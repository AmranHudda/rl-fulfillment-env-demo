"""Scripted tool-only baselines, not trained policies."""
import random


def solve_base(env, *, reverse=False, shuffle_seed=None, finish=True):
    return _solve(env, reverse=reverse, shuffle_seed=shuffle_seed, finish=finish)


def solve_split(env, *, reverse=False, shuffle_seed=None, south_first=False, finish=True):
    return _solve(env, reverse=reverse, shuffle_seed=shuffle_seed, finish=finish,
                  split=True, south_first=south_first)


def solve_duplicate(env, *, reverse=False, shuffle_seed=None, finish=True):
    return _solve(env, reverse=reverse, shuffle_seed=shuffle_seed, finish=finish)


def solve_stockout(env, *, reverse=False, shuffle_seed=None, finish=True):
    return _solve(env, reverse=reverse, shuffle_seed=shuffle_seed, finish=finish, stockout=True)


def solve_refund(env, *, reverse=False, shuffle_seed=None, finish=True):
    return _solve(env, reverse=reverse, shuffle_seed=shuffle_seed, finish=finish, refund=True)


def _solve(env, *, reverse, shuffle_seed, finish, split=False, south_first=False,
           stockout=False, refund=False):
    def call(tool, args=None):
        request = dict(tool=tool, args=args or {})
        if tool in {"reserve", "ship", "close_ticket", "refund"}:
            request["request_id"] = "solver-%s" % call.count
        call.count += 1
        result = env.step(request)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    call.count = 0
    tickets = call("list_tickets")["observation"]
    inventory = call("inventory")["observation"]
    work = sorted(tickets.items(), reverse=reverse)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(work)
    for tid, ticket in work:
        oid = ticket["order_id"]
        order = call("get_order", {"order_id": oid})["observation"]
        if order["needs_review"]:
            continue
        if refund and ticket["kind"] == "refund":
            receipt = call("refund", dict(order_id=oid))["observation"]["receipt"]
        elif order["status"] == "shipped":
            receipt = order["receipt"]
        else:
            if split:
                remaining = order["qty"]
                for warehouse in sorted(inventory, reverse=south_first):
                    qty = min(remaining, inventory[warehouse].get(order["sku"], 0))
                    if qty:
                        call("reserve", dict(order_id=oid, warehouse=warehouse, qty=str(qty)))
                        inventory[warehouse][order["sku"]] -= qty
                        remaining -= qty
                if remaining:
                    raise RuntimeError("Insufficient inventory for split order")
            else:
                warehouse = next((w for w, stock in inventory.items()
                                  if stock.get(order["sku"], 0) >= order["qty"]), None)
                if warehouse is None and stockout:
                    continue
                if warehouse is None:
                    raise RuntimeError("Insufficient inventory")
                call("reserve", dict(order_id=oid, warehouse=warehouse))
                inventory[warehouse][order["sku"]] -= order["qty"]
            receipt = call("ship", dict(order_id=oid))["observation"]["receipt"]
        call("close_ticket", dict(ticket_id=tid, receipt=receipt))
    if finish:
        return call("finish")


def solve_cancellation(env, *, reverse=False, shuffle_seed=None, finish=True):
    cancelled = set()
    tickets = {}

    def call(name, args=None):
        args = args or {}
        request = dict(tool=name, args=args)
        if name in {"reserve", "ship", "close_ticket", "cancel_order"}:
            request["request_id"] = "adaptive-%s" % call.count
        call.count += 1
        result = env.step(request)
        for event in result.get("events", []):
            oid = event["order_id"]
            receipt = call("cancel_order", dict(order_id=oid))["observation"]["receipt"]
            cancelled.add(oid)
            known_tickets = result["observation"] if name == "list_tickets" else tickets
            for tid, ticket in known_tickets.items():
                if ticket["order_id"] == oid:
                    call("close_ticket", dict(ticket_id=tid, receipt=receipt))
        if "error" in result and args.get("order_id") not in cancelled:
            raise RuntimeError(result["error"])
        return result

    call.count = 0
    tickets = call("list_tickets")["observation"]
    work = sorted(tickets.items(), reverse=reverse)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(work)
    for tid, ticket in work:
        oid = ticket["order_id"]
        order = call("get_order", dict(order_id=oid))["observation"]
        if order["needs_review"] or oid in cancelled or ticket["status"] == "closed":
            continue
        if order["cancel_requested"]:
            receipt = call("cancel_order", dict(order_id=oid))["observation"]["receipt"]
            call("close_ticket", dict(ticket_id=tid, receipt=receipt))
            cancelled.add(oid)
            continue
        if order["status"] == "open":
            inventory = call("inventory")["observation"]
            if oid in cancelled:
                continue
            warehouse = next(w for w, stock in inventory.items()
                             if stock.get(order["sku"], 0) >= order["qty"])
            call("reserve", dict(order_id=oid, warehouse=warehouse))
        if oid in cancelled:
            continue
        if order["status"] == "shipped":
            receipt = order["receipt"]
        else:
            result = call("ship", dict(order_id=oid))
            if oid in cancelled:
                continue
            receipt = result["observation"]["receipt"]
        call("close_ticket", dict(ticket_id=tid, receipt=receipt))
    if finish:
        return call("finish")
