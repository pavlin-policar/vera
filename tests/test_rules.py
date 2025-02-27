import unittest

import vera.rules
from vera.preprocessing import discretize
from vera.variables import ContinuousVariable


class TestIntervalRule(unittest.TestCase):
    def test_can_merge_with_non_interval_rules(self):
        r1 = vera.rules.IntervalRule(0, 5)
        r2 = vera.rules.EqualityRule(5)

        self.assertFalse(r1.can_merge_with(r2))
        self.assertFalse(r2.can_merge_with(r1))

    def test_can_merge_with_other_discretized_interval_rules(self):
        r1 = vera.rules.IntervalRule(0, 5)
        r2 = vera.rules.IntervalRule(5, 10)
        r3 = vera.rules.IntervalRule(10, 15)

        self.assertTrue(r1.can_merge_with(r2))
        self.assertTrue(r2.can_merge_with(r1))
        self.assertTrue(r2.can_merge_with(r3))
        self.assertTrue(r3.can_merge_with(r2))
        self.assertFalse(r1.can_merge_with(r3))
        self.assertFalse(r3.can_merge_with(r1))

    def test_can_merge_with_overlapping_interval_rules(self):
        r1 = vera.rules.IntervalRule(0, 6)
        r2 = vera.rules.IntervalRule(4, 10)
        r3 = vera.rules.IntervalRule(9, 15)
        r4 = vera.rules.IntervalRule(1, 4)

        self.assertTrue(r1.can_merge_with(r2))
        self.assertTrue(r2.can_merge_with(r1))
        self.assertTrue(r2.can_merge_with(r3))
        self.assertTrue(r3.can_merge_with(r2))
        # One contained within the other
        self.assertTrue(r1.can_merge_with(r4))
        self.assertTrue(r4.can_merge_with(r1))
        # Disjoint
        self.assertFalse(r1.can_merge_with(r3))
        self.assertFalse(r3.can_merge_with(r1))

    def test_can_merge_with_open_intervals(self):
        r1 = vera.rules.IntervalRule(upper=6)
        r2 = vera.rules.IntervalRule(lower=5, upper=10)
        r3 = vera.rules.IntervalRule(lower=9)

        self.assertTrue(r1.can_merge_with(r1))
        self.assertTrue(r2.can_merge_with(r2))
        self.assertTrue(r3.can_merge_with(r3))

        self.assertTrue(r1.can_merge_with(r2))
        self.assertTrue(r2.can_merge_with(r1))
        self.assertTrue(r2.can_merge_with(r3))
        self.assertTrue(r3.can_merge_with(r2))
        self.assertFalse(r1.can_merge_with(r3))
        self.assertFalse(r3.can_merge_with(r1))

    def test_merge_with_non_interval_rule(self):
        r1 = vera.rules.IntervalRule(0, 5)
        r2 = vera.rules.EqualityRule(5)

        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r1.merge_with(r2)
        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r2.merge_with(r1)

    def test_merge_with_other_discretized_interval_rules(self):
        r1 = vera.rules.IntervalRule(0, 5)
        r2 = vera.rules.IntervalRule(5, 10)
        r3 = vera.rules.IntervalRule(10, 15)

        self.assertEqual(r1.merge_with(r1), r1)

        self.assertEqual(r1.merge_with(r2), vera.rules.IntervalRule(0, 10))
        self.assertEqual(r2.merge_with(r1), vera.rules.IntervalRule(0, 10))

        self.assertEqual(r2.merge_with(r3), vera.rules.IntervalRule(5, 15))
        self.assertEqual(r3.merge_with(r2), vera.rules.IntervalRule(5, 15))

        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r1.merge_with(r3)
        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r3.merge_with(r1)

    def test_merge_with_overlapping_interval_rules(self):
        r1 = vera.rules.IntervalRule(0, 6)
        r2 = vera.rules.IntervalRule(4, 10)
        r3 = vera.rules.IntervalRule(9, 15)
        r4 = vera.rules.IntervalRule(1, 4)

        self.assertEqual(r1.merge_with(r2), vera.rules.IntervalRule(0, 10))
        self.assertEqual(r2.merge_with(r1), vera.rules.IntervalRule(0, 10))

        self.assertEqual(r2.merge_with(r3), vera.rules.IntervalRule(4, 15))
        self.assertEqual(r3.merge_with(r2), vera.rules.IntervalRule(4, 15))

        # One contained within the other
        self.assertEqual(r1.merge_with(r4), vera.rules.IntervalRule(0, 6))
        self.assertEqual(r4.merge_with(r1), vera.rules.IntervalRule(0, 6))

        # Disjoint
        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r1.merge_with(r3)
        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r3.merge_with(r1)

    def test_merge_open_intervals(self):
        r1 = vera.rules.IntervalRule(upper=6)
        r2 = vera.rules.IntervalRule(lower=5, upper=10)
        r3 = vera.rules.IntervalRule(lower=9)

        self.assertEqual(r1.merge_with(r1), r1)
        self.assertEqual(r2.merge_with(r2), r2)
        self.assertEqual(r3.merge_with(r3), r3)

        self.assertEqual(r1.merge_with(r2), vera.rules.IntervalRule(upper=10))
        self.assertEqual(r2.merge_with(r1), vera.rules.IntervalRule(upper=10))

        self.assertEqual(r2.merge_with(r3), vera.rules.IntervalRule(lower=5))
        self.assertEqual(r3.merge_with(r2), vera.rules.IntervalRule(lower=5))

        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r1.merge_with(r3)
        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r3.merge_with(r1)

    def test_contains_non_interval_rule(self):
        r1 = vera.rules.IntervalRule(0, 5)
        r2 = vera.rules.EqualityRule(5)
        r3 = vera.rules.OneOfRule({1, 2, 3})

        self.assertFalse(r1.contains(r2))
        self.assertFalse(r1.contains(r3))

    def test_contains(self):
        r1 = vera.rules.IntervalRule(lower=5, upper=10)
        r2 = vera.rules.IntervalRule(lower=7, upper=8)
        self.assertTrue(r1.contains(r1))

        self.assertTrue(r1.contains(r2))
        self.assertFalse(r2.contains(r1))

        r1 = vera.rules.IntervalRule(lower=5, upper=10)
        r2 = vera.rules.IntervalRule(upper=6)
        r3 = vera.rules.IntervalRule(lower=9)
        self.assertFalse(r1.contains(r2))
        self.assertFalse(r1.contains(r3))
        self.assertFalse(r2.contains(r1))
        self.assertFalse(r3.contains(r1))

        r1 = vera.rules.IntervalRule(lower=5, upper=10)
        r2 = vera.rules.IntervalRule(lower=-50)
        r3 = vera.rules.IntervalRule(upper=50)
        self.assertTrue(r2.contains(r1))
        self.assertTrue(r3.contains(r1))
        self.assertFalse(r1.contains(r2))
        self.assertFalse(r1.contains(r3))


class TestEqualityRule(unittest.TestCase):
    def test_can_merge_with_other_rules(self):
        r1 = vera.rules.EqualityRule(5)
        r2 = vera.rules.IntervalRule(0, 5)
        r3 = vera.rules.OneOfRule({1, 2})
        r4 = vera.rules.EqualityRule(6)

        self.assertTrue(r1.can_merge_with(r1))

        self.assertFalse(r1.can_merge_with(r2))
        self.assertFalse(r2.can_merge_with(r1))

        self.assertTrue(r1.can_merge_with(r3))
        self.assertTrue(r3.can_merge_with(r1))

        self.assertTrue(r1.can_merge_with(r4))
        self.assertTrue(r4.can_merge_with(r1))

    def test_merge_with_incompatible_rules(self):
        r1 = vera.rules.EqualityRule(5)
        r2 = vera.rules.IntervalRule(0, 5)

        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r1.merge_with(r2)
        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r2.merge_with(r1)

    def test_merge_with_other_equality_rule(self):
        r1 = vera.rules.EqualityRule(3)
        r2 = vera.rules.EqualityRule(5)

        self.assertEqual(r1.merge_with(r2), vera.rules.OneOfRule({3, 5}))
        self.assertEqual(r2.merge_with(r1), vera.rules.OneOfRule({3, 5}))

    def test_contains_non_equality_rule(self):
        r1 = vera.rules.EqualityRule(5)
        r2 = vera.rules.IntervalRule(0, 5)
        r3 = vera.rules.OneOfRule({1, 2, 3})

        self.assertFalse(r1.contains(r2))
        self.assertFalse(r1.contains(r3))

    def test_contains(self):
        r1 = vera.rules.EqualityRule(5)
        r2 = vera.rules.EqualityRule(5)
        r3 = vera.rules.EqualityRule(2)

        self.assertTrue(r1.contains(r1))
        self.assertTrue(r1.contains(r2))
        self.assertTrue(r2.contains(r1))
        self.assertFalse(r1.contains(r3))
        self.assertFalse(r3.contains(r1))


class TestOneOfRule(unittest.TestCase):
    def test_can_merge_with_other_rules(self):
        r1 = vera.rules.OneOfRule({1, 2})
        r2 = vera.rules.IntervalRule(0, 5)
        r3 = vera.rules.EqualityRule(5)
        r4 = vera.rules.EqualityRule({2, 3})

        self.assertTrue(r1.can_merge_with(r1))

        self.assertFalse(r1.can_merge_with(r2))
        self.assertFalse(r2.can_merge_with(r1))

        self.assertTrue(r1.can_merge_with(r3))
        self.assertTrue(r3.can_merge_with(r1))

        self.assertTrue(r1.can_merge_with(r4))
        self.assertTrue(r4.can_merge_with(r1))

    def test_merge_with_incompatible_rules(self):
        r1 = vera.rules.OneOfRule({1, 2})
        r2 = vera.rules.IntervalRule(0, 5)

        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r1.merge_with(r2)
        with self.assertRaises(vera.rules.IncompatibleRuleError):
            r2.merge_with(r1)

    def test_merge_with_other_oneof_rule(self):
        r1 = vera.rules.OneOfRule({1, 2})
        r2 = vera.rules.OneOfRule({4, 5})

        self.assertEqual(r1.merge_with(r2), vera.rules.OneOfRule({1, 2, 4, 5}))
        self.assertEqual(r2.merge_with(r1), vera.rules.OneOfRule({1, 2, 4, 5}))

    def test_merge_with_equality_rule(self):
        r1 = vera.rules.OneOfRule({1, 2})
        r2 = vera.rules.EqualityRule(5)

        self.assertEqual(r1.merge_with(r2), vera.rules.OneOfRule({1, 2, 5}))
        self.assertEqual(r2.merge_with(r1), vera.rules.OneOfRule({1, 2, 5}))

    def test_merge_with_ovelapping_values(self):
        r1 = vera.rules.OneOfRule({1, 2, 3})
        r2 = vera.rules.OneOfRule({2, 3, 4})
        r3 = vera.rules.OneOfRule({2, 3})

        self.assertEqual(r1.merge_with(r2), vera.rules.OneOfRule({1, 2, 3, 4}))
        self.assertEqual(r2.merge_with(r1), vera.rules.OneOfRule({1, 2, 3, 4}))

        self.assertEqual(r1.merge_with(r3), vera.rules.OneOfRule({1, 2, 3}))
        self.assertEqual(r3.merge_with(r1), vera.rules.OneOfRule({1, 2, 3}))

    def test_contains_non_oneof_rule(self):
        r1 = vera.rules.OneOfRule({1, 2, 3})
        r2 = vera.rules.EqualityRule(2)
        r3 = vera.rules.IntervalRule(0, 5)

        self.assertTrue(r1.contains(r2))
        self.assertFalse(r1.contains(r3))

    def test_contains(self):
        r1 = vera.rules.OneOfRule({1, 2, 3})
        r2 = vera.rules.OneOfRule({1, 2})

        self.assertTrue(r1.contains(r1))
        self.assertTrue(r1.contains(r2))

        r1 = vera.rules.OneOfRule({1, 2, 3})
        r2 = vera.rules.OneOfRule({2, 3, 4})
        self.assertFalse(r1.contains(r2))

        r1 = vera.rules.OneOfRule({1, 2, 3})
        r2 = vera.rules.OneOfRule({7, 8, 9})
        self.assertFalse(r1.contains(r2))

        r1 = vera.rules.OneOfRule({1, 2, 3})
        r2 = vera.rules.EqualityRule(2)
        r3 = vera.rules.EqualityRule(10)
        self.assertTrue(r1.contains(r2))
        self.assertFalse(r1.contains(r3))


class TestIntervalRuleStringFormatting(unittest.TestCase):
    def test_decimals(self):
        r1 = vera.rules.IntervalRule(0, 1, precision=0)
        r2 = vera.rules.IntervalRule(1, 2, precision=1)
        r3 = vera.rules.IntervalRule(2, 4, precision=2)

        self.assertEqual(str(r1), "0 < x < 1")
        self.assertEqual(str(r2), "1.0 < x < 2.0")
        self.assertEqual(str(r3), "2.00 < x < 4.00")

    def test_none(self):
        r1 = vera.rules.IntervalRule(12123, 13023, precision=0)
        r2 = vera.rules.IntervalRule(13024, 13348, precision=0)
        r3 = vera.rules.IntervalRule(13349, 14012, precision=0)

        self.assertEqual(str(r1), "12123 < x < 13023")
        self.assertEqual(str(r2), "13024 < x < 13348")
        self.assertEqual(str(r3), "13349 < x < 14012")

    def test_rounding(self):
        r1 = vera.rules.IntervalRule(12123, 13023, precision=-3)
        r2 = vera.rules.IntervalRule(13024, 13348, precision=-3)
        r3 = vera.rules.IntervalRule(13349, 14012, precision=-3)

        self.assertEqual(str(r1), "12000 < x < 13000")
        self.assertEqual(str(r2), "13000 < x < 13000")
        self.assertEqual(str(r3), "13000 < x < 14000")
