import unittest

from environment import DEV_SEEDS, HELDOUT_SEEDS, DIFFICULTIES, VARIANTS, FulfillmentEnv
from solvers import (solve_base, solve_split, solve_duplicate, solve_stockout,
                     solve_cancellation, solve_refund)


class VariantTests(unittest.TestCase):
    def test_solutions_across_dev_seeds_and_action_orders(self):
        for difficulty in DIFFICULTIES:
            for seed in range(100):
                for ordering in ({}, {"reverse": True}, {"shuffle_seed": seed}):
                    with self.subTest(difficulty=difficulty, seed=seed, ordering=ordering):
                        env = FulfillmentEnv(seed, difficulty=difficulty)
                        self.assertEqual(solve_base(env, **ordering)["reward"], 1)
                        self.assertLess(env._steps, env.max_steps)

    def test_scaling_and_determinism(self):
        for difficulty, count in DIFFICULTIES.items():
            env = FulfillmentEnv(7, difficulty=difficulty)
            state = env.snapshot
            self.assertEqual(len(state["orders"]), count)
            self.assertEqual(state, FulfillmentEnv(7, difficulty=difficulty).snapshot)
            self.assertNotEqual(state, FulfillmentEnv(8, difficulty=difficulty).snapshot)
            self.assertEqual(state, FulfillmentEnv(7, difficulty=count).snapshot)
            self.assertEqual(len(state["tickets"]), count - (1 if count == 4 else 2))
            self.assertEqual(sum(o["needs_review"] for o in state["orders"].values()), count // 4)
        self.assertGreaterEqual(len(FulfillmentEnv(difficulty="extreme").snapshot["tickets"]), 60)

    def test_integer_difficulties_and_invalid_configs(self):
        for count in range(4, 65):
            env = FulfillmentEnv(9, difficulty=count)
            self.assertEqual(len(env.snapshot["orders"]), count)
            self.assertEqual(solve_base(env)["reward"], 1)
        for difficulty in (None, True, 3, 65, 4.0, "unknown", "16", [], {}):
            with self.subTest(difficulty=difficulty), self.assertRaises(ValueError):
                FulfillmentEnv(difficulty=difficulty)
        with self.assertRaises(ValueError):
            FulfillmentEnv(variant="unknown")

    def test_reset_keeps_and_can_change_configuration(self):
        env = FulfillmentEnv(7, difficulty="hard", max_steps=500)
        before = env.snapshot
        solve_base(env)
        observation = env.reset(7)
        self.assertEqual(env.snapshot, before)
        self.assertEqual(observation["max_steps"], 500)
        env.reset(8, difficulty="medium", max_steps=None, variant="base")
        self.assertEqual(len(env.snapshot["orders"]), 16)
        self.assertEqual(env.max_steps, 144)

    def test_seed_splits(self):
        self.assertEqual(len(DEV_SEEDS), 10000)
        self.assertEqual(len(HELDOUT_SEEDS), 50)
        self.assertEqual(HELDOUT_SEEDS.start, 1_000_000)
        self.assertFalse(set(DEV_SEEDS) & set(HELDOUT_SEEDS))

    def test_scaled_grader_failures(self):
        for failure in ("open_ticket", "review", "unrelated", "inventory", "budget"):
            with self.subTest(failure=failure):
                env = FulfillmentEnv(7, difficulty="hard")
                solve_base(env, finish=False)
                if failure == "open_ticket":
                    ticket = next(t for t in env._state["tickets"].values()
                                  if t["status"] == "closed")
                    ticket.update(status="open", receipt=None)
                elif failure == "review":
                    next(o for o in env._state["orders"].values()
                         if o["needs_review"])["status"] = "shipped"
                elif failure == "unrelated":
                    related = {t["order_id"] for t in env._state["tickets"].values()}
                    next(o for oid, o in env._state["orders"].items()
                         if oid not in related)["qty"] += 1
                elif failure == "inventory":
                    env._state["inventory"]["north"]["tee"] += 1
                else:
                    env.max_steps = env._steps
                self.assertEqual(env.step({"tool": "finish"})["reward"], 0)
                self.assertFalse(env.grade())

    def test_empty_completion_and_variant_hooks(self):
        self.assertEqual(FulfillmentEnv(difficulty="hard").step({"tool": "finish"})["reward"], 0)
        def generate(env, seed, count):
            return dict(state={"private_free": True}, facts={"eligible": set(), "secret": seed},
                        task="Test variant", tools={"finish": set(), "peek": set()},
                        mutations=set(), run=lambda env, tool, args: "visible",
                        grade=lambda env: env._facts["secret"] == 7)
        VARIANTS["test"] = generate
        try:
            env = FulfillmentEnv(7, variant="test")
            observation = env.reset(7)
            self.assertEqual(observation["tools"], {"finish": [], "peek": []})
            self.assertNotIn("secret", str(observation) + str(env.snapshot))
            self.assertIn("error", env.step({"tool": "inventory"}))
            self.assertEqual(env.step({"tool": "peek"})["observation"], "visible")
            self.assertEqual(env.step({"tool": "finish"})["reward"], 1)
        finally:
            del VARIANTS["test"]


def tool(env, name, args=None, key=None):
    request = dict(tool=name, args=args or {})
    if key is not None:
        request["request_id"] = key
    return env.step(request)


def check_variant_solutions(test, variant, solver):
    for difficulty in DIFFICULTIES:
        for seed in range(100):
            orderings = [{}, {"reverse": True}, {"shuffle_seed": seed}]
            if variant == "split":
                orderings += [{"south_first": True},
                              {"south_first": True, "reverse": True},
                              {"south_first": True, "shuffle_seed": seed}]
            for ordering in orderings:
                with test.subTest(difficulty=difficulty, seed=seed, ordering=ordering):
                    env = FulfillmentEnv(seed, variant=variant, difficulty=difficulty)
                    test.assertEqual(solver(env, **ordering)["reward"], 1)
                    test.assertLess(env._steps, env.max_steps)
                    if variant == "cancellation":
                        target = env._facts["cancelled_target"]
                        test.assertIsNotNone(target)
                        test.assertTrue(env.snapshot["orders"][target]["cancel_requested"])
                        test.assertEqual(env.snapshot["orders"][target]["status"], "cancelled")


class StepBudgetTests(unittest.TestCase):
    def test_assigned_budget_survives_reset(self):
        for variant in ("base", "cancellation"):
            with self.subTest(variant=variant):
                env = FulfillmentEnv(7, variant=variant, difficulty="medium")
                env.max_steps = 9
                self.assertEqual(env.reset(8)["max_steps"], 9)
                self.assertEqual(env.max_steps, 9)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.env = FulfillmentEnv(7, variant="split", difficulty="medium")
        state = self.env.snapshot
        self.oid = next(oid for oid in self.env._eligible
                        if state["orders"][oid]["qty"] > max(
                            stock[state["orders"][oid]["sku"]] for stock in state["inventory"].values()))
        self.order = state["orders"][self.oid]

    def test_solutions(self):
        check_variant_solutions(self, "split", solve_split)

    def test_generation_and_schema(self):
        for difficulty in DIFFICULTIES:
            env = FulfillmentEnv(7, variant="split", difficulty=difficulty)
            before = env.snapshot
            observation = env.reset(7)
            self.assertEqual(before, env.snapshot)
            self.assertNotEqual(before, FulfillmentEnv(8, variant="split", difficulty=difficulty).snapshot)
            self.assertEqual(observation["tools"]["reserve"], ["order_id", "qty", "warehouse"])
            self.assertNotIn("facts", observation)
            orders, inventory = before["orders"], before["inventory"]
            required = sum(orders[oid]["qty"] > max(stock[orders[oid]["sku"]]
                           for stock in inventory.values()) for oid in env._eligible)
            self.assertEqual(required, max(1, len(env._eligible) // 3))

    def test_partial_reservation_cannot_finish_or_ship(self):
        result = tool(self.env, "reserve", dict(order_id=self.oid, warehouse="north", qty="1"), "r")
        self.assertNotIn("error", result)
        self.assertEqual(self.env.snapshot["orders"][self.oid]["status"], "partially_reserved")
        before = self.env.snapshot
        self.assertIn("error", tool(self.env, "ship", dict(order_id=self.oid), "s"))
        self.assertEqual(before, self.env.snapshot)
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_full_quantity_in_one_warehouse_rejected(self):
        before = self.env.snapshot
        for warehouse in ("north", "south"):
            self.assertIn("error", tool(self.env, "reserve", dict(
                order_id=self.oid, warehouse=warehouse, qty=str(self.order["qty"])), warehouse))
            self.assertEqual(before, self.env.snapshot)
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_tampered_non_split_allocations_fail_grading(self):
        solve_split(self.env, finish=False)
        shipment = next(s for s in self.env._state["shipments"].values() if len(s["allocations"]) == 1)
        warehouse = next(iter(shipment["allocations"]))
        shipment["allocations"][warehouse] += 1
        self.env._state["inventory"][warehouse][shipment["sku"]] -= 1
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_replayed_reserve_does_not_double_allocate(self):
        args = dict(order_id=self.oid, warehouse="north", qty="1")
        first = tool(self.env, "reserve", args, "same")
        self.assertNotIn("error", first)
        before = self.env.snapshot
        self.assertEqual(first, tool(self.env, "reserve", args, "same"))
        self.assertEqual(before, self.env.snapshot)
        self.assertIn("error", tool(self.env, "reserve", dict(args, qty="2"), "same"))
        self.assertEqual(before, self.env.snapshot)

    def test_invalid_quantities_never_mutate(self):
        before = self.env.snapshot
        for qty in ("0", "-1", "+1", "1.0", "", " 1", "١", "9" * 5000,
                    str(self.order["qty"] + 1), 1):
            self.assertIn("error", tool(self.env, "reserve", dict(
                order_id=self.oid, warehouse="north", qty=qty), "bad"))
            self.assertEqual(before, self.env.snapshot)

    def test_review_and_unrelated_protected(self):
        solve_split(self.env, finish=False)
        for oid in self.env._state["orders"]:
            if oid not in self.env._eligible:
                before = self.env.snapshot
                self.assertIn("error", tool(self.env, "reserve", dict(
                    order_id=oid, warehouse="north", qty="1"), "protected"))
                self.assertEqual(before, self.env.snapshot)
        review = next(o for o in self.env._state["orders"].values() if o["needs_review"])
        review["allocations"]["north"] = 1
        self.assertEqual(tool(self.env, "finish")["reward"], 0)


class DuplicateTests(unittest.TestCase):
    def setUp(self):
        self.env = FulfillmentEnv(7, variant="duplicate", difficulty="medium")
        tickets = self.env.snapshot["tickets"]
        self.pair = next([tid for tid, t in tickets.items() if t["order_id"] == oid]
                         for oid in self.env._eligible
                         if sum(t["order_id"] == oid for t in tickets.values()) == 2)
        self.oid = tickets[self.pair[0]]["order_id"]

    def test_solutions(self):
        check_variant_solutions(self, "duplicate", solve_duplicate)

    def test_generation_and_receipts(self):
        for difficulty in DIFFICULTIES:
            env = FulfillmentEnv(7, variant="duplicate", difficulty=difficulty)
            before = env.snapshot
            self.assertEqual(before, FulfillmentEnv(7, variant="duplicate", difficulty=difficulty).snapshot)
            self.assertNotEqual(before, FulfillmentEnv(8, variant="duplicate", difficulty=difficulty).snapshot)
            self.assertEqual(env.reset(7)["tools"], FulfillmentEnv().reset()["tools"])
            self.assertGreater(len(before["tickets"]), len(env._eligible))
            solve_duplicate(env)
            self.assertEqual(len(env.snapshot["shipments"]), len(env._eligible))
            for oid in env._eligible:
                receipts = {t["receipt"] for t in env.snapshot["tickets"].values() if t["order_id"] == oid}
                self.assertEqual(receipts, {env.snapshot["orders"][oid]["receipt"]})

    def _ship_order(self):
        order = tool(self.env, "get_order", dict(order_id=self.oid))["observation"]
        stock = tool(self.env, "inventory")["observation"]
        warehouse = next(w for w in stock if stock[w].get(order["sku"], 0) >= order["qty"])
        self.assertNotIn("error", tool(self.env, "reserve", dict(order_id=self.oid, warehouse=warehouse), "r"))
        return tool(self.env, "ship", dict(order_id=self.oid), "s")["observation"]["receipt"]

    def test_only_one_ticket_closed_fails(self):
        tickets = tool(self.env, "list_tickets")["observation"]
        inventory = tool(self.env, "inventory")["observation"]
        for tid, ticket in tickets.items():
            if tid == self.pair[1]:
                continue
            oid = ticket["order_id"]
            order = tool(self.env, "get_order", dict(order_id=oid))["observation"]
            if order["needs_review"]:
                continue
            if order["status"] != "shipped":
                warehouse = next(w for w in inventory if inventory[w].get(order["sku"], 0) >= order["qty"])
                self.assertNotIn("error", tool(self.env, "reserve", dict(order_id=oid, warehouse=warehouse), tid + "r"))
                inventory[warehouse][order["sku"]] -= order["qty"]
                receipt = tool(self.env, "ship", dict(order_id=oid), tid + "s")["observation"]["receipt"]
            else:
                receipt = order["receipt"]
            self.assertNotIn("error", tool(self.env, "close_ticket", dict(ticket_id=tid, receipt=receipt), tid + "c"))
        self.assertEqual(self.env.snapshot["tickets"][self.pair[1]]["status"], "open")
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_fabricated_or_other_order_receipt_rejected(self):
        receipt = self._ship_order()
        self.assertNotIn("error", tool(self.env, "close_ticket", dict(ticket_id=self.pair[0], receipt=receipt), "c"))
        before = self.env.snapshot
        self.assertIn("error", tool(self.env, "close_ticket", dict(ticket_id=self.pair[1], receipt="fabricated"), "fake"))
        self.assertEqual(before, self.env.snapshot)
        other_tid = next(tid for tid, t in before["tickets"].items()
                         if t["order_id"] != self.oid and t["order_id"] in self.env._eligible)
        self.assertIn("error", tool(self.env, "close_ticket", dict(ticket_id=other_tid, receipt=receipt), "wrong"))
        self.assertEqual(before, self.env.snapshot)
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_review_duplicate_touched_fails(self):
        solve_duplicate(self.env, finish=False)
        tid = next(tid for tid, t in self.env.snapshot["tickets"].items()
                   if tid.startswith("duplicate-") and t["order_id"] not in self.env._eligible)
        before = self.env.snapshot
        self.assertIn("error", tool(self.env, "close_ticket", dict(ticket_id=tid, receipt="fake"), "review"))
        self.assertEqual(before, self.env.snapshot)
        self.env._state["tickets"][tid]["status"] = "closed"
        self.assertEqual(tool(self.env, "finish")["reward"], 0)


class StockoutTests(unittest.TestCase):
    def setUp(self):
        self.env = FulfillmentEnv(7, variant="stockout", difficulty="medium")

    def test_solutions(self):
        check_variant_solutions(self, "stockout", solve_stockout)

    def test_shortage_and_order_dependent_survivor(self):
        survivors = []
        for reverse in (False, True):
            env = FulfillmentEnv(7, variant="stockout", difficulty="medium")
            before = env.snapshot
            sku = env._facts["short_sku"]
            demand = sum(before["orders"][oid]["qty"] for oid in env._eligible
                         if before["orders"][oid]["sku"] == sku)
            self.assertEqual(sum(s[sku] for s in before["inventory"].values()), demand - 1)
            self.assertTrue(all(o["qty"] == 1 for o in before["orders"].values() if o["sku"] == sku))
            solve_stockout(env, reverse=reverse)
            remaining = [oid for oid in env._eligible if env.snapshot["orders"][oid]["status"] == "open"]
            self.assertEqual(len(remaining), 1)
            self.assertEqual(env.snapshot["orders"][remaining[0]], before["orders"][remaining[0]])
            survivors.append(remaining[0])
        self.assertNotEqual(*survivors)

    def test_too_many_orders_left_open(self):
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_dangling_reservation(self):
        oid = next(oid for oid in self.env._eligible
                   if self.env.snapshot["orders"][oid]["sku"] == self.env._facts["short_sku"])
        self.assertNotIn("error", tool(self.env, "reserve", dict(order_id=oid, warehouse="north"), "r"))
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_unshipped_ticket_cannot_be_closed(self):
        solve_stockout(self.env, finish=False)
        ticket_id = next(tid for tid, t in self.env.snapshot["tickets"].items()
                         if t["status"] == "open" and t["order_id"] in self.env._eligible)
        before = self.env.snapshot
        self.assertIn("error", tool(self.env, "close_ticket", dict(ticket_id=ticket_id, receipt="fake"), "fake"))
        self.assertEqual(before, self.env.snapshot)
        self.env._state["tickets"][ticket_id].update(status="closed", receipt="fake")
        self.assertEqual(tool(self.env, "finish")["reward"], 0)


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.env = FulfillmentEnv(7, variant="cancellation")

    def test_solutions(self):
        check_variant_solutions(self, "cancellation", solve_cancellation)

    def test_timing_range_scales_with_world_size(self):
        for difficulty in DIFFICULTIES:
            steps = []
            for seed in range(100):
                env = FulfillmentEnv(seed, variant="cancellation", difficulty=difficulty)
                solver_steps = 3 + len(env.snapshot["tickets"]) + 3 * len(env._eligible)
                step = env._facts["cancel_step"]
                self.assertGreaterEqual(step, 4)
                self.assertLessEqual(step, max(5, solver_steps // 3))
                steps.append(step)
            self.assertGreater(len(set(steps)), 1)
            if difficulty != "easy":
                self.assertGreater(max(steps), 5)

    def _arrive_after_reservations(self):
        before = self.env.snapshot
        for i, oid in enumerate(sorted(self.env._eligible)):
            order = before["orders"][oid]
            warehouse = next(w for w, stock in before["inventory"].items()
                             if stock.get(order["sku"], 0) >= order["qty"])
            self.assertNotIn("error", tool(self.env, "reserve", dict(order_id=oid, warehouse=warehouse), "pre-%s" % i))
        events = []
        for _ in range(4):
            events.extend(tool(self.env, "inventory").get("events", []))
        self.assertEqual(len(events), 1)
        oid = events[0]["order_id"]
        self.assertEqual(events[0]["type"], "cancellation")
        self.assertEqual(self.env.snapshot["orders"][oid]["status"], "reserved")
        tid = next(tid for tid, t in before["tickets"].items() if t["order_id"] == oid)
        self.assertTrue(self.env.snapshot["tickets"][tid]["cancel_requested"])
        return oid, tid

    def test_cancellation_after_reserve_handled_successfully(self):
        oid, tid = self._arrive_after_reservations()
        self.assertEqual(solve_cancellation(self.env)["reward"], 1)
        order = self.env.snapshot["orders"][oid]
        self.assertEqual(order["status"], "cancelled")
        self.assertEqual(self.env.snapshot["tickets"][tid]["receipt"], order["receipt"])
        self.assertNotIn(oid, [s["order_id"] for s in self.env.snapshot["shipments"].values()])

    def test_requested_order_cannot_ship_and_dangling_reservation_fails(self):
        oid, _ = self._arrive_after_reservations()
        before = self.env.snapshot
        self.assertIn("error", tool(self.env, "ship", dict(order_id=oid), "ship"))
        self.assertIn("error", tool(self.env, "reserve", dict(order_id=oid, warehouse="north"), "reserve"))
        self.assertEqual(before, self.env.snapshot)
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_cancelled_ticket_left_open_fails(self):
        oid, _ = self._arrive_after_reservations()
        self.assertNotIn("error", tool(self.env, "cancel_order", dict(order_id=oid), "cancel"))
        self._finish_other_reserved_orders(oid)
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_fabricated_cancellation_receipt_rejected(self):
        oid, tid = self._arrive_after_reservations()
        self.assertNotIn("error", tool(self.env, "cancel_order", dict(order_id=oid), "cancel"))
        before = self.env.snapshot
        self.assertIn("error", tool(self.env, "close_ticket", dict(ticket_id=tid, receipt="fake"), "fake"))
        self.assertEqual(before, self.env.snapshot)
        self._finish_other_reserved_orders(oid)
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def _finish_other_reserved_orders(self, cancelled):
        for tid, ticket in tool(self.env, "list_tickets")["observation"].items():
            oid = ticket["order_id"]
            order = tool(self.env, "get_order", dict(order_id=oid))["observation"]
            if oid == cancelled or order["needs_review"]:
                continue
            receipt = tool(self.env, "ship", dict(order_id=oid), tid + "s")["observation"]["receipt"]
            self.assertNotIn("error", tool(self.env, "close_ticket", dict(ticket_id=tid, receipt=receipt), tid + "c"))

    def test_ship_response_carries_event_and_solver_recovers(self):
        for seed in range(100):
            env = FulfillmentEnv(seed, variant="cancellation")
            if env._facts["cancel_step"] == 5:
                break
        else:
            self.fail("No step-5 seed found")
        tickets = tool(env, "list_tickets")["observation"]
        inventory = tool(env, "inventory")["observation"]
        oid = env._facts["priority"][0]
        order = tool(env, "get_order", dict(order_id=oid))["observation"]
        warehouse = next(w for w in inventory if inventory[w].get(order["sku"], 0) >= order["qty"])
        self.assertNotIn("error", tool(env, "reserve", dict(order_id=oid, warehouse=warehouse), "pre"))
        result = tool(env, "ship", dict(order_id=oid), "ship")
        self.assertIn("error", result)
        self.assertEqual(result["events"], [{"type": "cancellation", "order_id": oid}])
        self.assertEqual(env.snapshot["orders"][oid]["status"], "reserved")
        self.assertEqual(solve_cancellation(env)["reward"], 1)

    def test_no_cancellation_when_every_candidate_already_shipped(self):
        env = next(FulfillmentEnv(seed, variant="cancellation") for seed in range(100)
                   if FulfillmentEnv(seed, variant="cancellation")._facts["cancel_step"] == 5)
        state = env.snapshot
        for i, oid in enumerate(sorted(env._eligible)):
            order = state["orders"][oid]
            warehouse = next(w for w, stock in state["inventory"].items()
                             if stock.get(order["sku"], 0) >= order["qty"])
            self.assertNotIn("error", tool(env, "reserve", dict(order_id=oid, warehouse=warehouse), "r%s" % i))
            self.assertNotIn("error", tool(env, "ship", dict(order_id=oid), "s%s" % i))
        self.assertNotIn("events", tool(env, "inventory"))
        for tid, ticket in state["tickets"].items():
            order = env.snapshot["orders"][ticket["order_id"]]
            if not order["needs_review"]:
                self.assertNotIn("error", tool(env, "close_ticket", dict(ticket_id=tid, receipt=order["receipt"]), tid))
        self.assertEqual(tool(env, "finish")["reward"], 1)

    def test_non_requested_cancellation_rejected(self):
        oid = next(iter(self.env._eligible))
        before = self.env.snapshot
        self.assertIn("error", tool(self.env, "cancel_order", dict(order_id=oid), "invalid"))
        self.assertEqual(before, self.env.snapshot)
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_schedule_hidden_and_reset_deterministic(self):
        initial = self.env.snapshot
        observation = self.env.reset(7)
        self.assertNotIn("priority", str(observation) + str(initial))
        self.assertNotIn("cancel_step", str(observation) + str(initial))
        self.assertTrue(all(not o["cancel_requested"] for o in initial["orders"].values()))
        first = [tool(self.env, "inventory") for _ in range(6)]
        self.env.reset(7)
        self.assertEqual(first, [tool(self.env, "inventory") for _ in range(6)])


class CancellationResumeTests(unittest.TestCase):
    def test_solver_resumes_with_shipped_order_and_open_ticket(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                env = FulfillmentEnv(seed, variant="cancellation", difficulty="medium")
                tid, ticket = next(
                    (tid, t) for tid, t in sorted(env.snapshot["tickets"].items())
                    if t["order_id"] in env._eligible
                    and t["order_id"] not in env._facts["priority"][:1])
                env._facts["cancel_step"] = 10 ** 6
                oid = ticket["order_id"]
                order = env.snapshot["orders"][oid]
                warehouse = next(w for w, stock in env.snapshot["inventory"].items()
                                 if stock.get(order["sku"], 0) >= order["qty"])
                env.step({"tool": "reserve", "args": {"order_id": oid, "warehouse": warehouse},
                          "request_id": "pre-reserve"})
                shipped = env.step({"tool": "ship", "args": {"order_id": oid},
                                    "request_id": "pre-ship"})
                self.assertNotIn("error", shipped)
                self.assertEqual(solve_cancellation(env)["reward"], 1)


class RefundTests(unittest.TestCase):
    def setUp(self):
        self.env = FulfillmentEnv(7, variant="refund", difficulty="medium")

    def test_solutions(self):
        check_variant_solutions(self, "refund", solve_refund)

    def test_review_refund_tool_succeeds_but_grader_fails(self):
        solve_refund(self.env, finish=False)
        oid = next(oid for oid, o in self.env.snapshot["orders"].items()
                   if o["needs_review"] and o["status"] == "shipped")
        self.assertNotIn("error", tool(self.env, "refund", dict(order_id=oid), "review"))
        self.assertEqual(self.env.snapshot["orders"][oid]["status"], "refunded")
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_unrelated_shipped_refund_fails(self):
        solve_refund(self.env, finish=False)
        related = {t["order_id"] for t in self.env.snapshot["tickets"].values()}
        oid = next(oid for oid, o in self.env.snapshot["orders"].items()
                   if oid not in related and o["status"] == "shipped")
        self.assertNotIn("error", tool(self.env, "refund", dict(order_id=oid), "unrelated"))
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_shipment_receipt_rejected_for_refund_ticket(self):
        tid, ticket = next((tid, t) for tid, t in self.env.snapshot["tickets"].items()
                           if t["kind"] == "refund" and not self.env.snapshot["orders"][t["order_id"]]["needs_review"])
        order = self.env.snapshot["orders"][ticket["order_id"]]
        before = self.env.snapshot
        self.assertIn("error", tool(self.env, "close_ticket", dict(ticket_id=tid, receipt=order["receipt"]), "shipment"))
        self.assertEqual(before, self.env.snapshot)
        self.assertEqual(tool(self.env, "finish")["reward"], 0)

    def test_double_refund_rejected(self):
        oid = next(iter(self.env._facts["refund_eligible"]))
        result = tool(self.env, "refund", dict(order_id=oid), "refund")
        self.assertNotIn("error", result)
        before = self.env.snapshot
        self.assertEqual(result, tool(self.env, "refund", dict(order_id=oid), "refund"))
        self.assertIn("error", tool(self.env, "refund", dict(order_id=oid), "twice"))
        self.assertEqual(before, self.env.snapshot)

    def test_initial_shipments_preserved_and_inventory_reconciles(self):
        before = self.env.snapshot
        solve_refund(self.env, finish=False)
        for receipt, shipment in before["shipments"].items():
            self.assertEqual(self.env.snapshot["shipments"][receipt], shipment)
        self.assertTrue(self.env.grade())
        receipt = next(iter(before["shipments"]))
        self.env._state["shipments"][receipt]["qty"] += 1
        self.assertEqual(tool(self.env, "finish")["reward"], 0)
