import unittest

import numpy as np

from vera.rules import EqualityRule, IntervalRule
from vera.variables import (
    ContinuousVariable,
    DiscreteVariable,
    IndicatorVariable,
)


class TestSubset(unittest.TestCase):
    def setUp(self) -> None:
        self.idx = np.array([4, 0, 2])

    def test_continuous_variable(self):
        v = ContinuousVariable("cont1", np.array([5.0, 2.0, 3.0, 1.0, 5.0]))

        result = v.subset(self.idx)

        self.assertIsInstance(result, ContinuousVariable)
        self.assertEqual("cont1", result.name)
        np.testing.assert_equal(np.array([5.0, 5.0, 3.0]), result.values)

    def test_discrete_variable_retains_categories(self):
        v = DiscreteVariable(
            "disc1",
            np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
            categories=["low", "med", "high"],
            ordered=True,
        )

        result = v.subset(self.idx)

        self.assertIsInstance(result, DiscreteVariable)
        self.assertEqual(("low", "med", "high"), result.categories)
        self.assertTrue(result.ordered)
        np.testing.assert_equal(np.array([0.0, 0.0, 2.0]), result.values)

    def test_indicator_variable_retains_rule(self):
        base = ContinuousVariable("cont1", np.array([5.0, 2.0, 3.0, 1.0, 5.0]))
        rule = IntervalRule(2.5, np.inf, value_name="cont1")
        v = IndicatorVariable(base, rule, np.array([1.0, 0.0, 1.0, 0.0, 1.0]))

        result = v.subset(self.idx)

        self.assertIsInstance(result, IndicatorVariable)
        self.assertIs(rule, result.rule)
        np.testing.assert_equal(np.array([1.0, 1.0, 1.0]), result.values)

    def test_indicator_variable_subsets_its_base_variable(self):
        base = ContinuousVariable("cont1", np.array([5.0, 2.0, 3.0, 1.0, 5.0]))
        v = IndicatorVariable(
            base,
            EqualityRule(5.0, value_name="cont1"),
            np.array([1.0, 0.0, 0.0, 0.0, 1.0]),
        )

        result = v.subset(self.idx)

        self.assertIsInstance(result.base_variable, ContinuousVariable)
        self.assertEqual("cont1", result.base_variable.name)
        np.testing.assert_equal(
            np.array([5.0, 5.0, 3.0]), result.base_variable.values
        )

    def test_leaves_the_original_untouched(self):
        values = np.array([5.0, 2.0, 3.0, 1.0, 5.0])
        v = ContinuousVariable("cont1", values)

        v.subset(self.idx)

        np.testing.assert_equal(values, v.values)

    def test_with_a_boolean_mask(self):
        v = ContinuousVariable("cont1", np.array([5.0, 2.0, 3.0, 1.0, 5.0]))

        result = v.subset(np.array([True, False, True, False, True]))

        np.testing.assert_equal(np.array([5.0, 3.0, 5.0]), result.values)


if __name__ == "__main__":
    unittest.main()
