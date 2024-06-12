import unittest
from itertools import product, combinations

import numpy as np
import pandas as pd

import vera
from vera import preprocessing as pp
from tests.utils import generate_clusters
from vera.region_annotation import RegionAnnotation
from vera.variables import IndicatorVariable, MergeError, IndicatorVariableGroup


class TestRegionAnnotationSplit(unittest.TestCase):
    def test_split_1(self):
        """In this test, we generate two non-overlapping clusters with a single
        constant feature, resulting in a single RegionAnnotation with a region
        made of two parts. This test then splits the RegionAnnotation into
        two new RegionAnnotation, each containing their own region part."""
        np.random.seed(0)
        x, features = generate_clusters([1, -1], [0.25, 0.25], n_samples=50)
        # Create constant feature
        features = pd.DataFrame(0, index=features.index, columns=["constant"])

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5, filter_constant=False, filter_uninformative=False,
        )
        assert len(region_annotations) == 1, "We should only have one variable"
        assert len(region_annotations[0]) == 1, "The variable should have only one region"

        ra: RegionAnnotation = region_annotations[0][0]
        assert len(ra.contained_region_annotations) == 1

        split_parts = ra.split()
        self.assertEqual(
            2, len(split_parts), "Region was split into incorrect number of parts."
        )

    def test_split_2(self):
        """This test is the same as above, but instead of a constant feature, we
        here use a feature that has random values, but should be merged into a
        single one during preprocessing."""
        np.random.seed(0)
        x, features = generate_clusters([-1, 1], [0.25, 0.25], n_samples=100)
        # Shuffle the clusters so that both cluster ids appear in both clusters
        features = features.sample(frac=1)

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=1, filter_uninformative=False,
        )
        assert len(region_annotations) == 1, "We should only have one variable"
        assert len(region_annotations[0]) == 1, "The variable should have only one region"

        ra: RegionAnnotation = region_annotations[0][0]
        # We should contain two subregion annotations, one for cluster 0 and one
        # for cluster 1
        assert len(ra.contained_region_annotations) == 2

        split_parts = ra.split()
        self.assertEqual(
            2, len(split_parts), "Region was split into incorrect number of parts."
        )

    def test_split_equality_and_hash(self):
        """Same setup as the first test"""
        np.random.seed(0)
        x, features = generate_clusters([1, -1], [0.25, 0.25], n_samples=50)
        # Create constant feature
        features = pd.DataFrame(0, index=features.index, columns=["constant"])

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5, filter_constant=False, filter_uninformative=False,
        )
        assert len(region_annotations) == 1, "We should only have one variable"
        assert len(region_annotations[0]) == 1, "The variable should have only one region"

        ra: RegionAnnotation = region_annotations[0][0]
        split_parts = ra.split()
        assert len(split_parts) == 2
        part1, part2 = split_parts

        self.assertNotEquals(part1, part2, "__eq__ not working")
        self.assertEqual(len({part1, part2}), 2, "__hash__ not working")

    def test_split_contained_samples(self):
        """Same setup as the first test"""
        np.random.seed(0)
        x, features = generate_clusters([1, -1], [0.25, 0.25], n_samples=50)
        # Create constant feature
        features = pd.DataFrame(0, index=features.index, columns=["constant"])

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5, filter_constant=False, filter_uninformative=False
        )
        assert len(region_annotations) == 1, "We should only have one feature"
        assert len(region_annotations[0]) == 1, "The variable should have only one region"

        ra = region_annotations[0][0]
        split_parts = ra.split()
        assert len(split_parts) == 2
        part1, part2 = split_parts

        # The shared samples should be empty
        shared_samples = part1.contained_samples & part2.contained_samples
        self.assertEqual(len(shared_samples), 0)


class TestIndicatorVariableMerge(unittest.TestCase):
    def test_merge_indicator_with_indicator_from_same_base_var_cont(self):
        np.random.seed(0)
        x = pd.Series(np.random.normal(0, 1, size=50), name="cont")

        cont_var = pp.ingest(x)
        cont_var_ind = pp.discretize(cont_var, n_bins=3)
        bin1, bin2, bin3 = cont_var_ind

        # These merges should be okay
        bin12 = bin1.merge_with(bin2)
        bin21 = bin2.merge_with(bin1)
        self.assertIsInstance(bin12, IndicatorVariable)
        self.assertIsInstance(bin21, IndicatorVariable)

        bin23 = bin2.merge_with(bin3)
        bin32 = bin3.merge_with(bin2)
        self.assertIsInstance(bin23, IndicatorVariable)
        self.assertIsInstance(bin32, IndicatorVariable)

        with self.assertRaises(MergeError):
            bin1.merge_with(bin3)
            bin3.merge_with(bin1)

    def test_merge_indicator_with_indicator_from_same_base_var_disc(self):
        np.random.seed(0)
        x = pd.Series(
            pd.Categorical.from_codes(
                np.random.random_integers(0, 2, size=50), categories=["r", "g", "b"]
            ),
            name="color",
        )

        disc_var = pp.ingest(x)
        disc_var_ind = pp.one_hot(disc_var)
        bin1, bin2, bin3 = disc_var_ind

        # These merges should be okay
        bin12 = bin1.merge_with(bin2)
        bin21 = bin2.merge_with(bin1)
        self.assertIsInstance(bin12, IndicatorVariable)
        self.assertIsInstance(bin21, IndicatorVariable)

        bin23 = bin2.merge_with(bin3)
        bin32 = bin3.merge_with(bin2)
        self.assertIsInstance(bin23, IndicatorVariable)
        self.assertIsInstance(bin32, IndicatorVariable)

        # For unordered, discrete variables, we can merge any bin
        bin13 = bin1.merge_with(bin3)
        bin31 = bin3.merge_with(bin1)
        self.assertIsInstance(bin13, IndicatorVariable)
        self.assertIsInstance(bin31, IndicatorVariable)

    # def test_merge_indicator_with_indicator_from_same_base_var_disc_ordered(self):
    #     # TODO: We don't yet have the infrastructure for ordered discrete variables
    #     np.random.seed(0)
    #     x = pd.Series(
    #         pd.Categorical.from_codes(
    #             np.random.random_integers(0, 2, size=50),
    #             categories=["low", "medium", "high"],
    #             ordered=True,
    #         ),
    #         name="value",
    #     )
    #
    #     disc_var = pp.ingest(x)
    #     disc_var_ind = pp.one_hot(disc_var)
    #     bin1, bin2, bin3 = disc_var_ind
    #
    #     # These merges should be okay
    #     bin12 = bin1.merge_with(bin2)
    #     bin21 = bin2.merge_with(bin1)
    #     self.assertIsInstance(bin12, IndicatorVariable)
    #     self.assertIsInstance(bin21, IndicatorVariable)
    #
    #     bin23 = bin2.merge_with(bin3)
    #     bin32 = bin3.merge_with(bin2)
    #     self.assertIsInstance(bin23, IndicatorVariable)
    #     self.assertIsInstance(bin32, IndicatorVariable)
    #
    #     # Ordered variables can't be merged willy-nilly
    #     with self.assertRaises(MergeError):
    #         bin1.merge_with(bin3)
    #         bin3.merge_with(bin1)

    def test_merge_indicator_with_indicator_from_different_base_vars(self):
        np.random.seed(0)
        c1 = pd.Series(np.random.normal(0, 1, size=50), name="cont1")
        c2 = pd.Series(np.random.normal(0, 1, size=50), name="cont2")
        d1 = pd.Series(
            pd.Categorical.from_codes(
                np.random.random_integers(0, 1, size=50), categories=["m", "f"]
            ),
            name="disc2",
        )
        d2 = pd.Series(
            pd.Categorical.from_codes(
                np.random.random_integers(0, 2, size=50), categories=["r", "g", "b"]
            ),
            name="disc2",
        )

        cont1_var_ind = pp.discretize(pp.ingest(c1), n_bins=3)
        cont2_var_ind = pp.discretize(pp.ingest(c2), n_bins=3)
        disc1_var_ind = pp.one_hot(pp.ingest(d1))
        disc2_var_ind = pp.one_hot(pp.ingest(d2))

        to_test = [cont1_var_ind, cont2_var_ind, disc1_var_ind, disc2_var_ind]
        for var_ind1, var_ind2 in combinations(to_test, 2):
            for ind1, ind2 in product(var_ind1, var_ind2):
                m1 = ind1.merge_with(ind2)
                m2 = ind2.merge_with(ind1)
                self.assertIsInstance(m1, IndicatorVariableGroup)
                self.assertIsInstance(m2, IndicatorVariableGroup)
                self.assertEqual(m1, m2)


class TestIndicatorVariableGroup(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        variables = {
            f"c{i + 1}": pd.Series(np.random.normal(0, 1, size=50), name=f"c{i + 1}")
            for i in range(5)
        }
        self.variable_parts = {
            k: pp.discretize(pp.ingest(v), n_bins=6) for k, v in variables.items()
        }

    def test_construction_1(self):
        """Test merging unrelated variables."""
        group = IndicatorVariableGroup([
            self.variable_parts["c1"][0],
            self.variable_parts["c2"][0],
            self.variable_parts["c3"][0],
        ])
        self.assertEqual(3, len(group.variables))

    def test_construction_2(self):
        """Test whether ordered, compatible rules can be merged."""
        group = IndicatorVariableGroup([
            self.variable_parts["c1"][0],
            self.variable_parts["c1"][1],
            self.variable_parts["c1"][2],
        ])
        self.assertEqual(1, len(group.variables))

    def test_construction_3(self):
        """Test whether unordered, but otherwise compatible rules can be merged."""
        group = IndicatorVariableGroup([
            self.variable_parts["c1"][2],
            self.variable_parts["c1"][0],
            self.variable_parts["c1"][1],
        ])
        self.assertEqual(1, len(group.variables))

    def test_construction_4(self):
        """Test whether uncompatible rules are properly handled."""
        group1 = IndicatorVariableGroup([
            self.variable_parts["c1"][0],
            self.variable_parts["c1"][2],
        ])
        group2 = IndicatorVariableGroup([
            self.variable_parts["c1"][2],
            self.variable_parts["c1"][0],
        ])
        self.assertEqual(group1, group2)
        self.assertEqual(2, len(group1.variables))

    def test_construction_5(self):
        """Ensure that merging [B1, B2, B4, B5] results in [B12, B45]."""
        group = IndicatorVariableGroup([
            self.variable_parts["c1"][0],
            self.variable_parts["c1"][1],
            self.variable_parts["c1"][3],
            self.variable_parts["c1"][4],
        ])
        print(group)
        self.assertEqual(2, len(group.variables))


class TestIndicatorVariableGroupMerge(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        variables = {
            f"c{i + 1}": pd.Series(np.random.normal(0, 1, size=50), name=f"c{i + 1}")
            for i in range(5)
        }
        self.variable_parts = {
            k: pp.discretize(pp.ingest(v), n_bins=3) for k, v in variables.items()
        }

    def test_merge_with_indicator_1(self):
        """For a variable group [C1_0, C2_0], add in C3_0. The result should be
        [C1_0, C2_0, C3_0]."""
        variable_group = IndicatorVariableGroup([
            self.variable_parts["c1"][0],
            self.variable_parts["c2"][0],
        ])

        # Merge [C1_0, C2_0] with C3_0
        result1 = variable_group.merge_with(self.variable_parts["c3"][0])
        self.assertIsInstance(result1, IndicatorVariableGroup)
        self.assertEqual(3, len(result1.variables))

        # Merge C3_0 with [C1_0, C2_0]
        result2 = self.variable_parts["c3"][0].merge_with(variable_group)
        self.assertIsInstance(result2, IndicatorVariableGroup)
        self.assertEqual(3, len(result2.variables))

        self.assertEqual(result1, result2)

    def test_merge_with_indicator_2(self):
        """For a variable group [C1_0, C2_0], add in C1_1. The result should be
        [C1_12, C2_0]."""
        variable_group = IndicatorVariableGroup([
            self.variable_parts["c1"][0],
            self.variable_parts["c2"][0],
        ])

        # Merge [C1_0, C2_0] with C1_1
        result1 = variable_group.merge_with(self.variable_parts["c1"][1])
        self.assertIsInstance(result1, IndicatorVariableGroup)
        self.assertEqual(2, len(result1.variables))

        # Merge C1_1 with [C1_0, C2_0]
        result2 = self.variable_parts["c1"][1].merge_with(variable_group)
        self.assertIsInstance(result2, IndicatorVariableGroup)
        self.assertEqual(2, len(result2.variables))

        self.assertEqual(result1, result2)
