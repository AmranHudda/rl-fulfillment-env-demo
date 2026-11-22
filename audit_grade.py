"""Run grade() condition mutants in memory, with no changes to source files."""
import argparse
import ast
import copy
import inspect
import io
import textwrap
import unittest
from unittest.mock import patch

from environment import FulfillmentEnv


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
            yield f"L{predicate.lineno} {expression} -> {value}", namespace["grade"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-tests", action="store_true")
    args = parser.parse_args()

    def run():
        loader = unittest.TestLoader()
        suite = (loader.loadTestsFromName("test_environment") if args.original_tests
                 else loader.discover(".", pattern="test*.py"))
        return unittest.TextTestRunner(stream=io.StringIO()).run(suite)

    baseline = run()
    if not baseline.wasSuccessful():
        for test, details in baseline.failures + baseline.errors:
            print(f"{test}:\n{details}")
        return 1

    survivors = []
    total = 0
    for description, mutated in mutants(FulfillmentEnv.grade):
        total += 1
        with patch.object(FulfillmentEnv, "grade", mutated):
            if run().wasSuccessful():
                survivors.append(description)
    for description in survivors:
        print(f"SURVIVED: {description}")
    print(f"{baseline.testsRun} tests; {total - len(survivors)}/{total} mutants caught; "
          f"{len(survivors)} survived")
    return bool(survivors)


if __name__ == "__main__":
    raise SystemExit(main())
