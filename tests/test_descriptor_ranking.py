import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
from shapely import geometry as geom

from vera import metrics
from vera.embedding import Embedding
from vera.plotting import _format_descriptor, plot_annotation
from vera.region import Region
from vera.region_annotation import RegionAnnotation
from vera.rules import EqualityRule
from vera.variables import (
    ContinuousVariable,
    IndicatorVariable,
    IndicatorVariableGroup,
)

PRIOR_STRENGTH = 10


def make_indicator(name: str, values: np.ndarray) -> IndicatorVariable:
    values = np.asarray(values, dtype=float)
    base = ContinuousVariable(name, values)
    rule = EqualityRule("high", value_name=name)
    return IndicatorVariable(base, rule, values)


def make_values(n_samples: int, n_in_region: int, ones_inside: int, ones_outside: int):
    """Indicator values over a layout where the region contains the first
    `n_in_region` samples."""
    values = np.zeros(n_samples)
    values[:ones_inside] = 1
    values[n_in_region:n_in_region + ones_outside] = 1
    return values


def make_region_annotation(variables: list[IndicatorVariable], n_in_region: int):
    n_samples = len(variables[0].values)
    # Samples on a line; the region is a box around the first `n_in_region`
    embedding = Embedding(
        np.column_stack([np.arange(n_samples, dtype=float), np.zeros(n_samples)])
    )
    polygon = geom.box(-0.5, -1.0, n_in_region - 0.5, 1.0)
    region = Region(embedding, polygon)

    if len(variables) == 1:
        descriptor = variables[0]
    else:
        descriptor = IndicatorVariableGroup(variables)
    return RegionAnnotation(region, descriptor)


def expected_rates(v, ra, prior_strength=PRIOR_STRENGTH):
    S = sorted(ra.region.contained_samples)
    k = v.values[S].sum()
    q = v.values.mean()
    p = (k + prior_strength * q) / (len(S) + prior_strength)
    return p, q


class TestDescriptorScores(unittest.TestCase):
    def setUp(self):
        # Region contains samples 0-9 out of 40
        self.n_samples, self.n_in_region = 40, 10
        self.v_a = make_indicator(
            "a", make_values(self.n_samples, self.n_in_region, 8, 4)
        )
        self.v_b = make_indicator(
            "b", make_values(self.n_samples, self.n_in_region, 6, 0)
        )
        self.v_c = make_indicator(
            "c", make_values(self.n_samples, self.n_in_region, 10, 30)
        )
        self.ra = make_region_annotation(
            [self.v_a, self.v_b, self.v_c], self.n_in_region
        )

    def test_scores_match_hand_computed_values(self):
        for v in [self.v_a, self.v_b, self.v_c]:
            p, q = expected_rates(v, self.ra)
            expected = {
                "purity": p,
                "purity_gain": p - q,
                "lift": p / q,
                "purity_lift": p * np.log2(p / q),
            }
            for method, expected_score in expected.items():
                result = metrics.descriptor_scores(self.ra, method=method)
                self.assertAlmostEqual(
                    expected_score, result[v], msg=f"method={method}, v={v}"
                )

    def test_zero_prior_strength_yields_raw_rates(self):
        scores = metrics.descriptor_scores(
            self.ra, method="purity", prior_strength=0
        )
        self.assertAlmostEqual(0.8, scores[self.v_a])
        self.assertAlmostEqual(0.6, scores[self.v_b])
        self.assertAlmostEqual(1.0, scores[self.v_c])

    def test_shrinkage_pulls_small_regions_toward_the_background(self):
        # The same perfectly pure feature, measured in a small and in a large
        # region: the small region's estimate ends up closer to the background
        v_small = make_indicator("v", make_values(400, 5, 5, 35))
        ra_small = make_region_annotation([v_small], 5)
        v_large = make_indicator("v", make_values(400, 50, 50, 35))
        ra_large = make_region_annotation([v_large], 50)

        score_small = metrics.descriptor_scores(ra_small, method="purity")[v_small]
        score_large = metrics.descriptor_scores(ra_large, method="purity")[v_large]
        self.assertLess(score_small, score_large)

        q = v_small.values.mean()
        self.assertLess(abs(score_small - q), abs(score_small - 1.0))
        self.assertLess(abs(score_large - 1.0), abs(score_large - q))

    def test_ubiquitous_feature_scores_no_purity_gain(self):
        # v_c holds for every sample, so knowing the region adds nothing
        scores = metrics.descriptor_scores(self.ra, method="purity_gain")
        self.assertAlmostEqual(0, scores[self.v_c])
        self.assertGreater(scores[self.v_a], 0)
        self.assertGreater(scores[self.v_b], 0)

    def test_callable_method(self):
        scores = metrics.descriptor_scores(
            self.ra, method=lambda v, ra: -v.values.sum()
        )
        self.assertAlmostEqual(-12, scores[self.v_a])
        self.assertAlmostEqual(-6, scores[self.v_b])
        self.assertAlmostEqual(-40, scores[self.v_c])

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            metrics.descriptor_scores(self.ra, method="not_a_method")

    def test_negative_prior_strength_raises(self):
        with self.assertRaises(ValueError):
            metrics.descriptor_scores(self.ra, prior_strength=-1)

    def test_single_indicator_descriptor(self):
        v = make_indicator("a", make_values(40, 10, 8, 4))
        ra = make_region_annotation([v], 10)
        scores = metrics.descriptor_scores(ra)
        self.assertEqual([v], list(scores))


class TestRegionAnnotationRanking(unittest.TestCase):
    def setUp(self):
        self.n_samples, self.n_in_region = 40, 10
        # Alphabetical and characteristic order disagree: a is diluted by
        # out-of-region mass, b holds only inside the region, c is ubiquitous
        self.v_a = make_indicator(
            "a", make_values(self.n_samples, self.n_in_region, 8, 24)
        )
        self.v_b = make_indicator(
            "b", make_values(self.n_samples, self.n_in_region, 8, 0)
        )
        self.v_c = make_indicator(
            "c", make_values(self.n_samples, self.n_in_region, 10, 30)
        )
        self.variables = [self.v_a, self.v_b, self.v_c]

    def test_group_variables_are_ranked_most_characteristic_first(self):
        ra = make_region_annotation(self.variables, self.n_in_region)
        scores = metrics.descriptor_scores(ra)
        ranked = ra.descriptor.variables
        self.assertEqual([self.v_b, self.v_a, self.v_c], ranked)
        self.assertEqual(
            sorted(scores.values(), reverse=True), [scores[v] for v in ranked]
        )

    def test_single_variable_descriptor_passes_through_untouched(self):
        v = make_indicator("a", make_values(40, 10, 8, 4))
        ra = make_region_annotation([v], 10)
        self.assertIs(v, ra.descriptor)

    def test_shared_group_is_not_reordered_in_place(self):
        group = IndicatorVariableGroup(self.variables)
        original_order = list(group.variables)

        ra = make_region_annotation(self.variables, self.n_in_region)
        ra2 = RegionAnnotation(ra.region, group)

        self.assertEqual(original_order, group.variables)
        self.assertIsNot(group, ra2.descriptor)

    def test_ranked_group_keeps_the_descriptor_values(self):
        group = IndicatorVariableGroup(self.variables)
        ra = make_region_annotation(self.variables, self.n_in_region)
        np.testing.assert_allclose(group.values, ra.descriptor.values)
        self.assertEqual(
            frozenset(group.variables), frozenset(ra.descriptor.variables)
        )

    def test_contained_variables_is_independent_of_member_order(self):
        # v_a scores highest in a region over its own support, v_b in another;
        # identity must not depend on which region ranked the group
        ra_1 = make_region_annotation(self.variables, self.n_in_region)
        values = np.zeros(self.n_samples)
        values[self.n_in_region:self.n_in_region + 8] = 1
        v_b2 = make_indicator("b", values)
        ra_2 = RegionAnnotation(
            ra_1.region, IndicatorVariableGroup([self.v_a, v_b2, self.v_c])
        )
        self.assertEqual(
            ra_1.descriptor.contained_variables,
            tuple(sorted(ra_1.descriptor.contained_variables)),
        )
        self.assertEqual(
            [v.name for v in ra_1.descriptor.contained_variables],
            [v.name for v in ra_2.descriptor.contained_variables],
        )

    def test_split_parts_rank_against_their_own_regions(self):
        # Two boxes; u dominates the first part, w the second
        n_samples = 40
        embedding = Embedding(
            np.column_stack([np.arange(n_samples, dtype=float), np.zeros(n_samples)])
        )
        u_values = np.zeros(n_samples)
        u_values[0:10] = 1
        u_values[20:23] = 1
        w_values = np.zeros(n_samples)
        w_values[20:30] = 1
        w_values[0:3] = 1
        u, w = make_indicator("u", u_values), make_indicator("w", w_values)

        polygon = geom.MultiPolygon([
            geom.box(-0.5, -1.0, 9.5, 1.0),
            geom.box(19.5, -1.0, 29.5, 1.0),
        ])
        ra = RegionAnnotation(
            Region(embedding, polygon), IndicatorVariableGroup([u, w])
        )

        part_1, part_2 = ra.split()
        self.assertEqual([u, w], part_1.descriptor.variables)
        self.assertEqual([w, u], part_2.descriptor.variables)


class TestUnrankedDescriptors(unittest.TestCase):
    """Contrastive explanations keep group descriptors in alphabetical order
    so that a variable group reads identically in every region it annotates."""

    def _two_merged_ras(self):
        """Two rank_descriptor=False merges of the same two base variables,
        over regions that would rank them oppositely."""
        n = 40
        u_values, w_values = np.zeros(n), np.zeros(n)
        u_values[0:10] = 1
        w_values[0:10] = 1
        u_values[20:23] = 1
        w_values[20:30] = 1
        u, w = make_indicator("u", u_values), make_indicator("w", w_values)

        ras = []
        for lo, hi in [(0, 10), (20, 30)]:
            embedding = Embedding(
                np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
            )
            region = Region(embedding, geom.box(lo - 0.5, -1.0, hi - 0.5, 1.0))
            ras.append(RegionAnnotation.merge(
                [RegionAnnotation(region, u), RegionAnnotation(region, w)],
                rank_descriptor=False,
            ))
        return ras

    def test_merge_without_ranking_keeps_alphabetical_order(self):
        for ra in self._two_merged_ras():
            self.assertEqual(
                sorted(ra.descriptor.variables), list(ra.descriptor.variables)
            )

    def test_same_variable_set_lands_in_one_descriptor_group(self):
        from vera.utils import group_by_descriptor

        ra_1, ra_2 = self._two_merged_ras()
        self.assertEqual(1, len(group_by_descriptor([ra_1, ra_2])))


class TestFormatLabel(unittest.TestCase):
    def setUp(self):
        self.variables = [
            make_indicator(name, make_values(40, 10, 5 + i, i))
            for i, name in enumerate("abcde")
        ]
        self.group = IndicatorVariableGroup(self.variables)

    def test_no_cap_matches_str(self):
        self.assertEqual(str(self.group), self.group.format_label())
        self.assertEqual(
            str(self.group), self.group.format_label(max_descriptors=None)
        )

    def test_cap_at_least_group_size_matches_str(self):
        self.assertEqual(str(self.group), self.group.format_label(max_descriptors=5))
        self.assertEqual(str(self.group), self.group.format_label(max_descriptors=9))

    def test_cap_yields_k_variable_lines_plus_truncation_line(self):
        label = self.group.format_label(max_descriptors=2)
        lines = label.split("\n")
        self.assertEqual(3, len(lines))
        self.assertEqual(
            [str(v) for v in self.group.variables[:2]], lines[:2]
        )

    def test_truncation_marker_reports_remaining_count(self):
        label = self.group.format_label(max_descriptors=2)
        self.assertEqual("(+3 more)", label.split("\n")[-1])
        label = self.group.format_label(max_descriptors=4)
        self.assertEqual("(+1 more)", label.split("\n")[-1])

    def test_custom_truncation_template(self):
        label = self.group.format_label(
            max_descriptors=2, truncation_template="and {n} others"
        )
        self.assertEqual("and 3 others", label.split("\n")[-1])

    def test_single_indicator_ignores_the_cap(self):
        v = self.variables[0]
        self.assertEqual(str(v), v.format_label(max_descriptors=1))

    def test_non_positive_cap_raises(self):
        with self.assertRaises(ValueError):
            self.group.format_label(max_descriptors=0)
        with self.assertRaises(ValueError):
            self.group.format_label(max_descriptors=-1)


class TestPlottingLabelTruncation(unittest.TestCase):
    def setUp(self):
        self.variables = [
            make_indicator(name, make_values(80, 20, 10 + i, 2 * i))
            for i, name in enumerate("abcdefg")
        ]
        self.ra = make_region_annotation(self.variables, 20)

    def test_default_cap_is_five(self):
        label = _format_descriptor(self.ra.descriptor, max_descriptors=5)
        lines = label.split("\n")
        self.assertEqual(6, len(lines))
        self.assertEqual("(+2 more)", lines[-1])

    def test_no_cap_preserves_full_label(self):
        label = _format_descriptor(self.ra.descriptor)
        self.assertEqual(str(self.ra.descriptor), label)

    def test_plot_annotation_renders_truncated_label(self):
        handles, _, fig, ax = plot_annotation(
            [self.ra], optimize_labels=False, return_ax=True
        )
        texts = [h.get_text() for h in handles]
        self.assertEqual(1, len(texts))
        self.assertIn("(+2 more)", texts[0])
        self.assertEqual(6, len(texts[0].split("\n")))
