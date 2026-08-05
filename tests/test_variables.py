import unittest

import numpy as np

from vera.rules import EqualityRule, IntervalRule
from vera.variables import (
    ContinuousVariable,
    DiscreteVariable,
    IndicatorVariable,
    IndicatorVariableGroup,
    conjunction,
    disjunction,
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



class TestThreeValuedLogic(unittest.TestCase):
    """A NaN marks a sample the indicator has no measurement for, so it may
    still turn out to belong either way."""

    def test_conjunction(self):
        a = np.array([1.0, 1.0, 1.0, 0.0, 0.0, np.nan])
        b = np.array([1.0, 0.0, np.nan, 0.0, np.nan, np.nan])

        result = conjunction([a, b])

        np.testing.assert_equal(
            np.array([1.0, 0.0, np.nan, 0.0, 0.0, np.nan]), result
        )

    def test_disjunction(self):
        a = np.array([1.0, 1.0, 1.0, 0.0, 0.0, np.nan])
        b = np.array([1.0, 0.0, np.nan, 0.0, np.nan, np.nan])

        result = disjunction([a, b])

        np.testing.assert_equal(
            np.array([1.0, 1.0, 1.0, 0.0, np.nan, np.nan]), result
        )

    def test_a_known_exclusion_beats_a_missing_measurement(self):
        """A sample one indicator does not flag is out of the group, whether or
        not the others measured it."""
        known = np.array([0.0])
        unknown = np.array([np.nan])

        np.testing.assert_equal(np.array([0.0]), conjunction([known, unknown]))

    def test_group_values_use_the_conjunction(self):
        v1 = IndicatorVariable(
            ContinuousVariable("a", np.array([1.0, 1.0, 0.0])),
            EqualityRule(1, value_name="a"),
            np.array([1.0, 1.0, 0.0]),
        )
        v2 = IndicatorVariable(
            ContinuousVariable("b", np.array([1.0, np.nan, np.nan])),
            EqualityRule(1, value_name="b"),
            np.array([1.0, np.nan, np.nan]),
        )

        group = IndicatorVariableGroup([v1, v2])

        np.testing.assert_equal(np.array([1.0, np.nan, 0.0]), group.values)


if __name__ == "__main__":
    unittest.main()
