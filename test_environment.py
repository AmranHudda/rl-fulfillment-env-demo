import copy
import unittest

from environment import FulfillmentEnv


def call(env, tool, args=None, key=None):
    request = {"tool": tool, "args": args or {}}
    if key is not None:
        request["request_id"] = key
    return env.step(request)


def solve(env, reverse=False):
    tickets = call(env, "list_tickets")["observation"]
    inventory = call(env, "inventory")["observation"]
    for tid, ticket in sorted(tickets.items(), reverse=reverse):
        oid = ticket["order_id"]
        order = call(env, "get_order", {"order_id": oid})["observation"]
        if order["needs_review"]:
            continue
        warehouse = next(w for w, stock in inventory.items()
                         if stock.get(order["sku"], 0) >= order["qty"])
        result = call(env, "reserve", {"order_id": oid, "warehouse": warehouse}, tid + "-reserve")
        assert "error" not in result, result
        inventory[warehouse][order["sku"]] -= order["qty"]
        receipt = call(env, "ship", {"order_id": oid}, tid + "-ship")["observation"]["receipt"]
        result = call(env, "close_ticket", {"ticket_id": tid, "receipt": receipt}, tid + "-close")
        assert "error" not in result, result
    return call(env, "finish")


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = FulfillmentEnv(seed=7)
        self.tickets = self.env.snapshot["tickets"]
        self.tid, self.ticket = next(iter(self.tickets.items()))
        self.oid = self.ticket["order_id"]

    def test_success_across_100_seeds(self):
        for seed in range(100):
            with self.subTest(seed=seed):
                self.assertEqual(solve(FulfillmentEnv(seed))["reward"], 1)

    def test_alternative_valid_action_order(self):
        self.assertEqual(solve(FulfillmentEnv(7), reverse=True)["reward"], 1)

    def test_success_claim_without_work_gets_zero(self):
        result = call(self.env, "finish")
        self.assertEqual(result["reward"], 0)
        self.assertTrue(result["terminated"])

    def test_close_with_fabricated_receipt_fails_without_mutation(self):
        before = self.env.snapshot
        result = call(self.env, "close_ticket", {"ticket_id": self.tid, "receipt": "pretend"}, "x")
        self.assertIn("error", result)
        self.assertEqual(before, self.env.snapshot)

    def test_ship_without_reservation_fails(self):
        before = self.env.snapshot
        self.assertIn("error", call(self.env, "ship", {"order_id": self.oid}, "x"))
        self.assertEqual(before, self.env.snapshot)

    def test_review_required_order_cannot_ship(self):
        oid = next(oid for oid, o in self.env.snapshot["orders"].items() if o["needs_review"])
        before = self.env.snapshot
        self.assertIn("error", call(self.env, "reserve", {"order_id": oid, "warehouse": "south"}, "x"))
        self.assertEqual(before, self.env.snapshot)

    def test_unrelated_order_cannot_mutate(self):
        all_ticket_orders = {t["order_id"] for t in self.tickets.values()}
        oid = next(oid for oid in self.env.snapshot["orders"] if oid not in all_ticket_orders)
        self.assertIn("error", call(self.env, "reserve", {"order_id": oid, "warehouse": "north"}, "x"))

    def test_wrong_warehouse_does_not_spend_inventory(self):
        before = self.env.snapshot
        self.assertIn("error", call(self.env, "reserve", {"order_id": self.oid, "warehouse": "south"}, "x"))
        self.assertEqual(before, self.env.snapshot)

    def test_duplicate_request_does_not_reserve_twice(self):
        args = {"order_id": self.oid, "warehouse": "north"}
        first = call(self.env, "reserve", args, "same-request")
        before = self.env.snapshot
        self.assertEqual(first, call(self.env, "reserve", args, "same-request"))
        self.assertEqual(before, self.env.snapshot)

    def test_same_request_id_with_changed_payload_rejected(self):
        call(self.env, "reserve", {"order_id": self.oid, "warehouse": "north"}, "same-request")
        before = self.env.snapshot
        result = call(self.env, "reserve", {"order_id": self.oid, "warehouse": "south"}, "same-request")
        self.assertIn("error", result)
        self.assertEqual(before, self.env.snapshot)

    def test_observation_mutation_cannot_change_world(self):
        tickets = call(self.env, "list_tickets")["observation"]
        tickets[self.tid]["status"] = "closed"
        self.assertEqual(self.env.snapshot["tickets"][self.tid]["status"], "open")
        self.assertFalse(self.env.grade())

    def test_unknown_reward_mutation_tool_rejected(self):
        before = self.env.snapshot
        result = call(self.env, "set_reward", {"reward": "1"})
        self.assertIn("error", result)
        self.assertEqual(before, self.env.snapshot)

    def test_malformed_envelopes_fail_without_mutation(self):
        before = self.env.snapshot
        for request in (None, [], {}, {"tool": []}, {"tool": "reserve", "args": []},
                        {"tool": "inventory", "args": {"extra": "value"}}):
            result = self.env.step(request)
            self.assertIn("error", result)
            self.assertEqual(result["reward"], 0)
            self.assertEqual(before, self.env.snapshot)

    def test_reset_restores_world_and_clears_episode_state(self):
        before = self.env.snapshot
        self.assertEqual(solve(self.env)["reward"], 1)
        self.env.reset(7)
        self.assertEqual(self.env.snapshot, before)
        self.assertFalse(self.env.grade())
        self.assertEqual(solve(self.env)["reward"], 1)

    def test_step_limit_cannot_be_ignored(self):
        env = FulfillmentEnv(7, max_steps=1)
        call(env, "inventory")
        result = call(env, "finish")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["reward"], 0)
        self.assertFalse(env.grade())

    def test_terminal_episode_cannot_farm_more_reward(self):
        self.assertEqual(solve(self.env)["reward"], 1)
        self.assertEqual(call(self.env, "finish")["reward"], 0)

    def test_grader_catches_inventory_tampering(self):
        solve(self.env)
        self.env._state["inventory"]["north"]["tee"] += 1
        self.assertFalse(self.env.grade())

    def test_grader_catches_fake_ticket_completion(self):
        for ticket in self.env._state["tickets"].values():
            ticket["status"] = "closed"
        self.assertFalse(self.env.grade())

    def test_transcript_preserves_observations_at_the_time_of_each_step(self):
        from demo import episode
        trace = episode()
        initial_stock = trace["transcript"][1]["response"]["observation"]
        self.assertEqual(initial_stock, FulfillmentEnv(7).snapshot["inventory"])
        self.assertEqual(trace["terminal_reward"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
