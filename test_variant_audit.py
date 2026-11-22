import io
import unittest

from audit_variants import SOLVERS, complete, episode, suite
from environment import FulfillmentEnv


class VariantAuditTests(unittest.TestCase):
    def test_independent_oracle_accepts_solutions_and_rejects_unfinished_states(self):
        for variant, solver in SOLVERS.items():
            for difficulty in ("easy", "medium", "hard", "extreme"):
                with self.subTest(variant=variant, difficulty=difficulty):
                    env = FulfillmentEnv(11, variant=variant, difficulty=difficulty)
                    initial = env.snapshot
                    self.assertFalse(complete(variant, initial, initial))
                    self.assertEqual(solver(env)["reward"], 1)
                    self.assertTrue(complete(variant, initial, env.snapshot))

    def test_known_field_mutations_and_missing_records(self):
        for variant in SOLVERS:
            with self.subTest(variant=variant):
                output = io.StringIO()
                result = unittest.TextTestRunner(stream=output).run(suite(variant, "easy"))
                self.assertTrue(result.wasSuccessful(), output.getvalue())

    def test_sequence_strategies_at_every_difficulty(self):
        for variant in SOLVERS:
            for difficulty in ("easy", "medium", "hard", "extreme"):
                for mode in ("random", "guided partial", "guided complete"):
                    with self.subTest(variant=variant, difficulty=difficulty, mode=mode):
                        episode(11, mode, variant, difficulty)
