import contextlib
import io
import unittest
from unittest.mock import patch

import audit_grade
import audit_variants
import environment
from audit_variants import SOLVERS, reachable_suite
from environment import FulfillmentEnv


class MutationEquivalenceTests(unittest.TestCase):
    def test_reachable_solver_and_careless_prefixes_match_oracle(self):
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output).run(reachable_suite())
        self.assertTrue(result.wasSuccessful(), output.getvalue())

    def test_cross_variant_reachable_states_catch_locally_unreachable_mutants(self):
        selected = 0
        for description, mutant in audit_grade.mutants(FulfillmentEnv._grade_base):
            if not ('self._facts.get("split")' in description
                    or description.endswith("-> True") and any(fragment in description for fragment in
                        ("allocations", "qty <= 0", '"warehouse"] is not None'))):
                continue
            selected += 1
            with self.subTest(mutant=description), patch.object(FulfillmentEnv, "_grade_base", mutant):
                tests = unittest.TestSuite(audit_variants.ReachableStatesCase(variant, "easy")
                                           for variant in ("base", "split"))
                result = unittest.TextTestRunner(stream=io.StringIO()).run(tests)
                self.assertFalse(result.wasSuccessful(), description)
        self.assertEqual(selected, 11)

    def test_allowlist_is_exact_and_distinguishes_duplicate_predicates(self):
        seen = set()
        missing_order_reasons = []
        for owner, name in ((FulfillmentEnv, "_grade_base"), (environment, "grade_refund")):
            original = getattr(owner, name)
            for description, _ in audit_grade.mutants(original):
                reason = audit_grade.equivalent_reason(name, original, description)
                if description.endswith("final is None -> False"):
                    missing_order_reasons.append(reason)
                if reason is not None:
                    self.assertTrue(description.endswith("-> False"))
                    seen.add(reason)
            self.assertIsNone(audit_grade.equivalent_reason(name, original, "L0 unknown -> False"))
        self.assertEqual(seen, set(audit_grade.EQUIVALENT_MUTANTS.values()))
        self.assertEqual(len(seen), 7)
        self.assertIsNone(missing_order_reasons[0])
        self.assertIsNotNone(missing_order_reasons[1])

    def test_mutated_function_definitions_are_difficulty_independent(self):
        for variant in SOLVERS:
            expected = (getattr(environment, "grade_" + variant)
                        if variant in ("stockout", "cancellation", "refund")
                        else FulfillmentEnv._grade_base)
            for difficulty in ("easy", "medium", "hard", "extreme", 4, 24, 64):
                env = FulfillmentEnv(7, variant=variant, difficulty=difficulty)
                self.assertIs(env._spec["grade"], expected)
                self.assertIs(env._grade_base.__func__, FulfillmentEnv._grade_base)

    def run_mock_audit(self, expression):
        selected = next((description, mutant)
                        for description, mutant in audit_grade.mutants(FulfillmentEnv._grade_base)
                        if description.partition(" ")[2] == expression)
        def mutation_pool(original):
            return iter((selected,)) if original.__name__ == "_grade_base" else iter(())
        def empty_suite(*args, **kwargs):
            return unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        output = io.StringIO()
        with patch("sys.argv", ["audit_grade.py", "--variant", "base"]), \
                patch.object(audit_grade, "mutants", side_effect=mutation_pool), \
                patch.object(audit_grade, "suite", side_effect=empty_suite), \
                contextlib.redirect_stdout(output):
            status = audit_grade.main()
        return status, output.getvalue()

    def test_audit_exits_zero_for_documented_equivalent(self):
        status, output = self.run_mock_audit("oid in shipped -> False")
        self.assertFalse(status, output)
        self.assertIn("1 equivalent; 0 unexpected survivors", output)
        self.assertIn("ship requires reserved status", output)

    def test_audit_exits_nonzero_for_undocumented_survivor(self):
        status, output = self.run_mock_audit('self._facts.get("split") -> False')
        self.assertTrue(status, output)
        self.assertIn("0 equivalent; 1 unexpected survivors", output)
        self.assertIn("SURVIVED:", output)
