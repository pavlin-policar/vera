import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
from shapely import geometry as geom

from vera.embedding import Embedding
from vera.explain._descriptive import enrich_with_background
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


if __name__ == "__main__":
    unittest.main()
