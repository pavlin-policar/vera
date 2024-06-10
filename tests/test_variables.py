import unittest

import numpy as np
import pandas as pd

import vera
from vera import preprocessing as pp
from tests.utils import generate_clusters


class TestExplanatoryVariableSplit(unittest.TestCase):
    def test_split_variable_by_regions(self):
        """In this test, we generate two non-overlapping clusters with a single
        constant feature, resulting in a single ExplanatoryVariable with a region
        made of two parts. This test then splits the ExplanatoryVariable into
        two new ExplanatoryVariable, each containing their own region part."""
        np.random.seed(0)
        x, features = generate_clusters([1, -1], [0.25, 0.25], n_samples=50)
        # Create constant feature
        features = pd.DataFrame(0, index=features.index, columns=["constant"])

        feature_list = vera.an.generate_explanatory_features(
            features, embedding=x, scale_factor=0.5, filter_constant=False,
        )
        assert len(feature_list) == 1, "We should only have one feature"

        v = feature_list[0]
        assert len(v.region_annotations) == 1, \
            "The constant feature should only have one explanatory variable"

        expl_v = v.region_annotations[0]
        split_parts = expl_v.split_region()
        self.assertEqual(
            len(split_parts), 2, "Region was split into incorrect number of parts."
        )

    def test_split_variable_by_region_composite_explanatory_variables(self):
        """This test tests the same thing as the above, but instead of an
        ExplanatoryVariable, we here generate data that creates a
        CompositeExplanatoryVariable."""
        np.random.seed(0)
        x, features = generate_clusters([-1, 1], [0.25, 0.25], n_samples=100)
        # Shuffle the clusters so that both cluster ids appear in both clusters
        features = features.sample(frac=1)

        feature_list = vera.an.generate_explanatory_features(
            features, embedding=x, scale_factor=1, filter_constant=False,
        )
        assert len(feature_list) == 1, "We should only have one feature"

        v = feature_list[0]
        assert len(v.region_annotations) == 1
        expl_v = v.region_annotations[0]

        assert isinstance(expl_v, vera.explanation.CompositeRegionAnnotation)
        expl_v = v.region_annotations[0]
        split_parts = expl_v.split_region()
        self.assertEqual(
            len(split_parts), 2, "Region was split into incorrect number of parts."
        )

    def test_split_variables_equality_and_hash(self):
        """Same setup as the first test"""
        np.random.seed(0)
        x, features = generate_clusters([1, -1], [0.25, 0.25], n_samples=50)
        # Create constant feature
        features = pd.DataFrame(0, index=features.index, columns=["constant"])

        feature_list = vera.an.generate_explanatory_features(
            features, embedding=x, scale_factor=0.5, filter_constant=False,
        )
        assert len(feature_list) == 1, "We should only have one feature"

        v = feature_list[0]
        assert len(v.region_annotations) == 1, \
            "The constant feature should only have one explanatory variable"

        expl_v = v.region_annotations[0]
        split_parts = expl_v.split_region()
        assert len(split_parts) == 2
        part1, part2 = split_parts

        self.assertNotEquals(part1, part2, "__eq__ not working")
        self.assertEqual(len({part1, part2}), 2, "__hash__ not working")

    def test_split_variables_contained_samples(self):
        """Same setup as the first test"""
        np.random.seed(0)
        x, features = generate_clusters([1, -1], [0.25, 0.25], n_samples=50)
        # Create constant feature
        features = pd.DataFrame(0, index=features.index, columns=["constant"])

        feature_list = vera.an.generate_explanatory_features(
            features, embedding=x, scale_factor=0.5, filter_constant=False,
        )
        assert len(feature_list) == 1, "We should only have one feature"

        v = feature_list[0]
        assert len(v.region_annotations) == 1, \
            "The constant feature should only have one explanatory variable"

        expl_v = v.region_annotations[0]
        split_parts = expl_v.split_region()
        assert len(split_parts) == 2
        part1, part2 = split_parts

        # The shared samples should be empty
        shared_samples = part1.contained_samples & part2.contained_samples
        self.assertEqual(len(shared_samples), 0)

        # The values should also be updated
        # TODO: Not really, because the values refer to all the samples in the
        #  embedding with the rule, not just the samples in the region
        # print(part1.values & part2.values)

        # vera.pl.plot_regions([part1, part2], show=True, per_row=2, figwidth=6)


class TestExplanatoryVariable(unittest.TestCase):
    def test_can_merge_with_discretized_variables(self):
        x = np.random.normal(0, 1, size=50)
        z = np.random.normal(0, 1, size=(50, 2))
        df = pd.DataFrame(x, columns=["x"])
        variables = vera.an.generate_explanatory_features(df, z, scale_factor=0.5)
        expl = variables[0].region_annotations

        assert len(expl) >= 3, "Too few discretized bins to perform test"
        v1, v2, v3, *_ = expl

        self.assertTrue(v1.can_merge_with(v1))

        self.assertTrue(v1.can_merge_with(v2))
        self.assertTrue(v2.can_merge_with(v1))

        self.assertTrue(v2.can_merge_with(v3))
        self.assertTrue(v3.can_merge_with(v2))

        self.assertFalse(v1.can_merge_with(v3))
        self.assertFalse(v3.can_merge_with(v1))

    def test_merge_with_discretized_variables(self):
        x = np.random.normal(0, 1, size=50)
        z = np.random.normal(0, 1, size=(50, 2))
        df = pd.DataFrame(x, columns=["x"])
        df_discretized = pp.generate_region_annotations(df, z)

        variables = df_discretized.columns
        assert len(variables) >= 3, "Too few discretized bins to perform test"
        v1, v2, v3, *_ = variables

        v1v2 = v1.merge_with(v2)
        self.assertIsInstance(v1v2, vera.data.variable.IndicatorVariable)
        self.assertEqual(v1v2.rule.lower, v1.rule.lower)
        self.assertEqual(v1v2.rule.upper, v2.rule.upper)

        v2v1 = v2.merge_with(v1)
        self.assertEqual(v1v2, v2v1)

        # Test that we can merge a variable already contained into the interval
        v_combined = v1.merge_with(v2)
        v_combined_2 = v_combined.merge_with(v2)
        self.assertEqual(v_combined, v_combined_2)

        # Test that we can't merge non-neighboring bins
        with self.assertRaises(ValueError):
            v1.merge_with(v3)
        with self.assertRaises(ValueError):
            v3.merge_with(v1)

    def test_can_merge_with_one_hot_encoded_features(self):
        x = ["r", "g", "b"] * 5
        np.random.shuffle(x)
        z = np.random.normal(0, 1, size=(15, 2))

        df = pd.DataFrame(pd.Categorical(x), columns=["x"])
        df_encoded = pp.generate_region_annotations(df, z)

        variables = df_encoded.columns
        assert len(variables) == 3, "One hot encoding should produce exactly 3 variables"
        v1, v2, v3 = variables

        self.assertTrue(v1.can_merge_with(v2))
        self.assertTrue(v1.can_merge_with(v3))

        self.assertTrue(v2.can_merge_with(v1))
        self.assertTrue(v2.can_merge_with(v3))

        self.assertTrue(v3.can_merge_with(v1))
        self.assertTrue(v3.can_merge_with(v2))

    def test_merge_with_one_hot_encoded_features(self):
        x = ["r", "g", "b"] * 5
        np.random.shuffle(x)
        z = np.random.normal(0, 1, size=(15, 2))

        df = pd.DataFrame(pd.Categorical(x), columns=["x"])
        df_encoded = pp.generate_region_annotations(df, z)

        variables = df_encoded.columns
        assert len(variables) == 3, "One hot encoding should produce exactly 3 variables"
        v1, v2, v3 = variables

        v1v2 = v1.merge_with(v2)
        self.assertIsInstance(v1v2, vera.data.variable.IndicatorVariable)
        self.assertEqual(v1v2.rule, vera.rules.OneOfRule({v1.rule.value, v2.rule.value}))

        v2v1 = v2.merge_with(v1)
        self.assertEqual(v1v2, v2v1)

        v1v3 = v1.merge_with(v3)
        self.assertIsInstance(v1v3, vera.data.variable.IndicatorVariable)
        self.assertEqual(v1v3.rule, vera.rules.OneOfRule({v1.rule.value, v3.rule.value}))

        v3v1 = v3.merge_with(v1)
        self.assertEqual(v1v3, v3v1)
