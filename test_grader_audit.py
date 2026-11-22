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
