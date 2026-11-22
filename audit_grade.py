"""Run grade() condition mutants in memory, with no changes to source files."""
import argparse
import ast
import copy
import inspect
import io
import textwrap
import unittest
from unittest.mock import patch

import environment
from audit_variants import suite
from environment import FulfillmentEnv


EQUIVALENT_MUTANTS = {
    ("_grade_base", 1, "oid in shipped -> False"):
        "ship requires reserved status and changes it to shipped; no tool reopens an order, so shipment order IDs are unique",
    ("_grade_base", 2, "final is None -> False"):
        "tools never delete orders or replace their records with None",
    ("_grade_base", 1, "not allocations -> False"):
        "split ship requires allocations summing to the positive order quantity, so shipped allocations cannot be empty",
    ("_grade_base", 1, 'final["warehouse"] is not None -> False'):
        "split orders start with warehouse None and no split tool changes it",
    ("_grade_base", 1, 'shipment["warehouse"] is not None -> False'):
        "split ship copies the order warehouse, which is always None",
    ("grade_refund", 1, 'set(state["tickets"]) != set(initial["tickets"]) -> False'):
        "no tool creates or deletes tickets; ticket key sets never change",
    ("grade_refund", 1, 'len(state["refunds"]) != len(refunds) -> False'):
        "refund allows one entry per order; after the preceding set-equality check, ledger cardinality must equal eligible-refund cardinality",
}


def equivalent_reason(name, original, description):
    location, _, mutation = description.partition(" ")
    expression = mutation.rsplit(" -> ", 1)[0]
    source = textwrap.dedent(inspect.getsource(original))
    matches = [predicate for predicate in predicates(ast.parse(source))
               if " ".join(ast.get_source_segment(source, predicate).split()) == expression]
    occurrence = next((index for index, predicate in enumerate(matches, 1)
                       if f"L{predicate.lineno}" == location), None)
    return EQUIVALENT_MUTANTS.get((name, occurrence, mutation))


def predicates(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            yield node.test
            if isinstance(node.test, ast.BoolOp):
                yield from node.test.values
    for node in ast.walk(tree):
        if isinstance(node, ast.GeneratorExp):
            yield node.elt
    yield tree.body[0].body[-1].value  # final success return


def mutants(original):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    for index, predicate in enumerate(predicates(tree)):
        expression = " ".join(ast.get_source_segment(source, predicate).split())
        for value in (False, True):
            if isinstance(predicate, ast.Constant) and predicate.value is value:
                continue
            altered = copy.deepcopy(tree)
            target = list(predicates(altered))[index]

            class Replace(ast.NodeTransformer):
                def visit(self, node):
                    if node is target:
                        return ast.copy_location(ast.Constant(value=value), node)
                    return super().visit(node)

            altered = ast.fix_missing_locations(Replace().visit(altered))
            namespace = {}
            exec(compile(altered, "<grade mutant>", "exec"), original.__globals__, namespace)
            yield f"L{predicate.lineno} {expression} -> {value}", namespace[original.__name__]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-tests", action="store_true")
    parser.add_argument("--variant", choices=("base", "split", "duplicate", "stockout", "cancellation", "refund"))
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard", "extreme"), default="easy")
    args = parser.parse_args()

    def run():
        loader = unittest.TestLoader()
        if args.variant:
            tests = suite(args.variant, args.difficulty, cross_variant=True)
        else:
            tests = (loader.loadTestsFromName("test_environment") if args.original_tests
                     else loader.discover(".", pattern="test*.py"))
        return unittest.TextTestRunner(stream=io.StringIO()).run(tests)

    baseline = run()
    if not baseline.wasSuccessful():
        for test, details in baseline.failures + baseline.errors:
            print(f"{test}:\n{details}")
        return 1

    survivors = []
    equivalents = []
    total = 0
    targets = [(FulfillmentEnv, "grade")]
    if args.variant:
        targets.append((FulfillmentEnv, "_grade_base"))
        if args.variant in ("stockout", "cancellation", "refund"):
            targets.append((environment, "grade_" + args.variant))
    for owner, name in targets:
        original = getattr(owner, name)
        for description, mutated in mutants(original):
            total += 1
            with patch.object(owner, name, mutated):
                if run().wasSuccessful():
                    label = (name + ": " if args.variant else "") + description
                    reason = equivalent_reason(name, original, description) if args.variant else None
                    if reason is None:
                        survivors.append(label)
                    else:
                        equivalents.append((label, reason))
    for description, reason in equivalents:
        print(f"EQUIVALENT: {description}; {reason}")
    for description in survivors:
        print(f"SURVIVED: {description}")
    caught = total - len(survivors) - len(equivalents)
    if args.variant:
        print(f"{baseline.testsRun} tests; {caught}/{total} mutants caught; "
              f"{len(equivalents)} equivalent; {len(survivors)} unexpected survivors")
    else:
        print(f"{baseline.testsRun} tests; {caught}/{total} mutants caught; "
              f"{len(survivors)} survived")
    return bool(survivors)


if __name__ == "__main__":
    raise SystemExit(main())
