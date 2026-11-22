import unittest

from environment import FulfillmentEnv
from test_environment import call, solve


class GraderRegressionTests(unittest.TestCase):
    def test_unsuccessful_mutation_replays_even_after_state_changes(self):
        env = FulfillmentEnv(7)
        oid = next(iter(env.snapshot["orders"]))
        args = {"order_id": oid}
        failure = call(env, "ship", args, "early-ship")
        self.assertIn("error", failure)
        self.assertIn("observation", call(env, "reserve", dict(args, warehouse="north"), "reserve"))

        before = env.snapshot
        self.assertEqual(call(env, "ship", args, "early-ship"), failure)
        self.assertEqual(env.snapshot, before)
        self.assertIn("observation", call(env, "ship", args, "fresh-ship"))

    def test_unsuccessful_mutation_id_cannot_be_rebound(self):
        env = FulfillmentEnv(7)
        oid = next(iter(env.snapshot["orders"]))
        args = {"order_id": oid, "warehouse": "south"}
        self.assertIn("error", call(env, "reserve", args, "reserve"))

        before = env.snapshot
        result = call(env, "reserve", dict(args, warehouse="north"), "reserve")
        self.assertIn("already bound", result.get("error", ""))
        self.assertEqual(env.snapshot, before)

    def test_grader_returns_false_when_shipped_order_is_missing(self):
        env = FulfillmentEnv(7)
        self.assertEqual(solve(env)["reward"], 1)
        oid = next(iter(env.snapshot["shipments"].values()))["order_id"]
        del env._state["orders"][oid]
        self.assertFalse(env.grade())


class GraderOracleTests(unittest.TestCase):
    def setUp(self):
        self.env = FulfillmentEnv(7)
        self.initial = self.env.snapshot
        self.assertEqual(solve(self.env)["reward"], 1)
        self.state = self.env._state
        self.receipt, self.shipment = next(iter(self.state["shipments"].items()))
        self.oid = self.shipment["order_id"]
        self.order = self.state["orders"][self.oid]
        self.ticket = next(ticket for ticket in self.state["tickets"].values()
                           if ticket["order_id"] == self.oid)

    def test_truncation_overrides_otherwise_complete_state(self):
        self.env._truncated = True
        self.assertFalse(self.env.grade())

    def test_missing_shipment_rejected_even_when_remaining_stock_matches(self):
        self.state["inventory"][self.shipment["warehouse"]][self.shipment["sku"]] += self.shipment["qty"]
        del self.state["shipments"][self.receipt]
        self.assertFalse(self.env.grade())

    def test_unknown_order_shipment_rejected(self):
        self.shipment["order_id"] = "unknown-order"
        self.assertFalse(self.env.grade())

    def test_wrong_shipment_quantity_rejected_with_reconciled_stock(self):
        self.shipment["qty"] -= 1
        self.state["inventory"][self.shipment["warehouse"]][self.shipment["sku"]] += 1
        self.assertFalse(self.env.grade())

    def test_wrong_shipment_sku_rejected_with_reconciled_stock(self):
        jacket = next(shipment for shipment in self.state["shipments"].values()
                      if shipment["sku"] == "jacket")
        self.state["inventory"]["south"]["jacket"] += jacket["qty"]
        self.state["inventory"]["north"]["tee"] -= jacket["qty"]
        jacket.update(sku="tee", warehouse="north")
        self.state["orders"][jacket["order_id"]]["warehouse"] = "north"
        self.assertFalse(self.env.grade())

    def test_unknown_shipment_warehouse_rejected(self):
        self.shipment["warehouse"] = self.order["warehouse"] = "unknown-warehouse"
        self.assertFalse(self.env.grade())

    def test_warehouse_without_shipped_sku_rejected(self):
        # Reset stock is dense; this is a controller-side fixture with sparse stock.
        warehouse, sku = self.shipment["warehouse"], self.shipment["sku"]
        del self.env._initial["inventory"][warehouse][sku]
        del self.state["inventory"][warehouse][sku]
        self.assertFalse(self.env.grade())

    def test_reconciled_but_negative_stock_rejected(self):
        self.state["inventory"]["north"]["tee"] += self.shipment["qty"]
        self.state["inventory"]["south"]["tee"] -= self.shipment["qty"]
        self.shipment["warehouse"] = self.order["warehouse"] = "south"
        self.assertFalse(self.env.grade())

    def test_shipment_requires_matching_final_order_fields(self):
        for field, invalid in (("status", "reserved"), ("receipt", "invented"), ("warehouse", "south")):
            with self.subTest(field=field):
                valid = self.order[field]
                self.order[field] = invalid
                self.assertFalse(self.env.grade())
                self.order[field] = valid

    def test_matching_order_and_ticket_receipts_must_still_name_a_real_shipment(self):
        self.order["receipt"] = self.ticket["receipt"] = "invented"
        self.assertFalse(self.env.grade())

    def test_eligible_order_metadata_cannot_be_changed(self):
        for field, invalid in (("sku", "invented"), ("qty", 99), ("needs_review", True)):
            with self.subTest(field=field):
                valid = self.order[field]
                self.order[field] = invalid
                self.assertFalse(self.env.grade())
                self.order[field] = valid

    def test_extra_order_rejected(self):
        self.state["orders"]["extra"] = self.order.copy()
        self.assertFalse(self.env.grade())

    def test_extra_ticket_rejected(self):
        self.state["tickets"]["extra"] = self.ticket.copy()
        self.assertFalse(self.env.grade())

    def test_protected_order_status_cannot_be_changed(self):
        ticket_orders = {ticket["order_id"] for ticket in self.state["tickets"].values()}
        for oid, order in self.state["orders"].items():
            if order["needs_review"] or oid not in ticket_orders:
                with self.subTest(oid=oid):
                    order["status"] = "reserved"
                    self.assertFalse(self.env.grade())
                    order["status"] = self.initial["orders"][oid]["status"]

    def test_review_ticket_cannot_be_changed(self):
        review_ticket = next(ticket for ticket in self.state["tickets"].values()
                             if self.initial["orders"][ticket["order_id"]]["needs_review"])
        review_ticket["status"] = "closed"
        self.assertFalse(self.env.grade())

    def test_closed_ticket_requires_original_order_status_and_receipt(self):
        other_oid = next(oid for oid in self.state["orders"] if oid != self.oid)
        for field, invalid in (("order_id", other_oid), ("status", "open"), ("receipt", "invented")):
            with self.subTest(field=field):
                valid = self.ticket[field]
                self.ticket[field] = invalid
                self.assertFalse(self.env.grade())
                self.ticket[field] = valid
