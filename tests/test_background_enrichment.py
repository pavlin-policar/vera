import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
from shapely import geometry as geom

from vera.embedding import Embedding
from vera.explain._descriptive import enrich_with_background
from vera.plotting import _format_descriptor, _format_descriptor_lines, plot_annotation
from vera.region import Region
from vera.region_annotation import RegionAnnotation
from vera.variables import IndicatorVariableGroup

from tests.test_descriptor_ranking import make_indicator, make_values


class BackgroundEnrichmentTestBase(unittest.TestCase):
    """40 samples on a line. The core region holds samples 0-9; background
    regions extend to the right of it."""

    def setUp(self):
        self.n = 40
        self.embedding = Embedding(
            np.column_stack([np.arange(self.n, dtype=float), np.zeros(self.n)])
        )
        self.v_a = make_indicator("a", make_values(self.n, 10, 10, 0))
        self.v_b = make_indicator("b", make_values(self.n, 10, 8, 0))
        # Background candidates; their regions, not their values, decide
        # whether they enrich the core. c scores the same as a
        self.v_c = make_indicator("c", make_values(self.n, 10, 10, 0))
        self.v_e = make_indicator("e", make_values(self.n, 10, 10, 25))
        self.v_f = make_indicator("f", make_values(self.n, 10, 10, 28))

    def region(self, start: int, stop: int) -> Region:
        """The region holding samples `start` to `stop - 1`."""
        return Region(self.embedding, geom.box(start - 0.5, -1.0, stop - 0.5, 1.0))

    def core(self, descriptor=None) -> RegionAnnotation:
        if descriptor is None:
            descriptor = IndicatorVariableGroup([self.v_a, self.v_b])
        return RegionAnnotation(self.region(0, 10), descriptor)


class TestBackgroundProvenance(BackgroundEnrichmentTestBase):
    def test_added_variables_are_marked_as_background(self):
        background = [RegionAnnotation(self.region(0, 30), self.v_c)]
        enriched = enrich_with_background(self.core(), background)
        self.assertEqual({self.v_a, self.v_b, self.v_c}, set(enriched.descriptor.variables))
        self.assertEqual(frozenset([self.v_c]), enriched.descriptor.background_variables)

    def test_unenriched_descriptors_have_no_background(self):
        # The background region covers only half of the core region
        background = [RegionAnnotation(self.region(5, 30), self.v_c)]
        enriched = enrich_with_background(self.core(), background)
        self.assertEqual(frozenset(), enriched.descriptor.background_variables)
        self.assertEqual(frozenset(), self.v_a.background_variables)

    def test_single_indicator_core_becomes_a_marked_group(self):
        background = [RegionAnnotation(self.region(0, 30), self.v_c)]
        enriched = enrich_with_background(self.core(self.v_a), background)
        self.assertIsInstance(enriched.descriptor, IndicatorVariableGroup)
        self.assertEqual(frozenset([self.v_c]), enriched.descriptor.background_variables)

    def test_background_members_take_part_in_membership(self):
        background = [RegionAnnotation(self.region(0, 30), self.v_c)]
        enriched = enrich_with_background(self.core(), background)
        expected = self.v_a.values * self.v_b.values * self.v_c.values
        np.testing.assert_array_equal(expected, enriched.descriptor.values)

    def test_repeated_enrichment_keeps_earlier_background(self):
        once = enrich_with_background(
            self.core(), [RegionAnnotation(self.region(0, 30), self.v_c)]
        )
        twice = enrich_with_background(
            once, [RegionAnnotation(self.region(0, 35), self.v_e)]
        )
        self.assertEqual(
            frozenset([self.v_c, self.v_e]), twice.descriptor.background_variables
        )


class TestBackgroundOrdering(BackgroundEnrichmentTestBase):
    def enriched(self):
        """The core group a, b enriched with c, which ties with a."""
        background = [RegionAnnotation(self.region(0, 30), self.v_c)]
        return enrich_with_background(self.core(), background).descriptor

    def test_ranking_is_by_score_alone(self):
        # c ties with a and sorts after it
        self.assertEqual([self.v_a, self.v_c, self.v_b], self.enriched().variables)

    def test_background_moves_after_the_core(self):
        descriptor = self.enriched()
        ordered = descriptor.with_background_last()
        self.assertEqual([self.v_a, self.v_b, self.v_c], ordered.variables)
        self.assertIsNot(descriptor, ordered)
        self.assertEqual([self.v_a, self.v_c, self.v_b], descriptor.variables)

    def test_ordered_group_keeps_its_marking_and_scores(self):
        descriptor = self.enriched()
        ordered = descriptor.with_background_last()
        self.assertEqual(descriptor.background_variables, ordered.background_variables)
        self.assertEqual(descriptor.scores, ordered.scores)

    def test_group_without_background_is_returned_unchanged(self):
        group = self.core().descriptor
        self.assertIs(group, group.with_background_last())

    def test_ties_are_not_reordered_across_the_background_boundary(self):
        # a and c tie; preferring c must not lift it above the core member
        descriptor = self.enriched().with_background_last()
        partitioned = descriptor.with_ties_ordered_by(lambda v: v is self.v_c)
        self.assertEqual([self.v_a, self.v_b, self.v_c], partitioned.variables)
        ordered = self.enriched().with_ties_ordered_by(lambda v: v is self.v_c)
        self.assertEqual([self.v_a, self.v_c, self.v_b], ordered.variables)


class TestBackgroundLabels(BackgroundEnrichmentTestBase):
    def enriched(self, *background_variables):
        background = [
            RegionAnnotation(self.region(0, 30), v) for v in background_variables
        ]
        return enrich_with_background(self.core(), background)

    def test_background_lines_follow_the_core_and_are_flagged(self):
        lines = _format_descriptor_lines(self.enriched(self.v_c).descriptor)
        self.assertEqual(
            [("a = high", False), ("b = high", False), ("c = high", True)], lines
        )

    def test_unenriched_label_is_unflagged_and_unchanged(self):
        descriptor = self.core().descriptor
        lines = _format_descriptor_lines(descriptor)
        self.assertEqual([(line, False) for line in str(descriptor).split("\n")], lines)

    def test_truncation_hides_background_first_behind_a_plain_marker(self):
        descriptor = self.enriched(self.v_c, self.v_e).descriptor
        lines = _format_descriptor_lines(descriptor, max_descriptors=2)
        self.assertEqual(
            [("a = high", False), ("b = high", False), ("(+2 more)", False)], lines
        )

    def test_background_starts_its_own_line_in_a_joined_label(self):
        descriptor = self.enriched(self.v_c, self.v_e).descriptor
        lines = _format_descriptor_lines(descriptor, sep=", ")
        self.assertEqual(
            [("a = high, b = high", False), ("c = high, e = high", True)], lines
        )

    def test_marker_sharing_a_background_line_is_flagged_with_it(self):
        descriptor = self.enriched(self.v_c, self.v_e, self.v_f).descriptor
        lines = _format_descriptor_lines(
            descriptor, max_descriptors=3, sep=", ", truncation_sep=" "
        )
        self.assertEqual(
            [("a = high, b = high", False), ("c = high (+2 more)", True)], lines
        )

    def test_wrapped_background_lines_stay_flagged(self):
        lines = _format_descriptor_lines(
            self.enriched(self.v_c, self.v_e).descriptor, sep=", ", max_width=10
        )
        self.assertEqual(
            [("a = high,", False), ("b = high", False), ("c = high,", True), ("e = high", True)],
            lines,
        )

    def test_plain_label_text_matches_the_lines(self):
        descriptor = self.enriched(self.v_c).descriptor
        self.assertEqual("a = high\nb = high\nc = high", _format_descriptor(descriptor))

    def test_plot_annotation_draws_background_in_italics(self):
        handles, _, fig, ax = plot_annotation(
            [self.enriched(self.v_c)], optimize_labels=False, return_ax=True
        )
        styles = [(t.get_text(), t.get_fontstyle()) for t in handles[0].texts]
        self.assertEqual(
            [("a = high", "normal"), ("b = high", "normal"), ("c = high", "italic")],
            styles,
        )

    def test_background_style_is_configurable(self):
        handles, _, fig, ax = plot_annotation(
            [self.enriched(self.v_c)],
            optimize_labels=False,
            return_ax=True,
            background_label_kwargs={},
        )
        self.assertEqual({"normal"}, {t.get_fontstyle() for t in handles[0].texts})


if __name__ == "__main__":
    unittest.main()
