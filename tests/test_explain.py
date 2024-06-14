import sys
import unittest

import vera

sys.path.append("../experiments/")
import datasets


class ExplainTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.iris = datasets.Dataset.load("iris")
        cls.region_annotations = vera.an.generate_region_annotations(
            cls.iris.features,
            cls.iris.embedding,
            n_discretization_bins=10,
            scale_factor=1,
            sample_size=5000,
            contour_level=0.25,
            merge_min_sample_overlap=0.5,
            merge_min_purity_gain=0.5,
        )


class TestContrastiveRanking(ExplainTestBase):
    def test_1(self):
        layouts = vera.explain.contrastive(self.region_annotations, max_panels=2)

        self.assertEqual(2, len(layouts))


class TestDescriptiveRanking(ExplainTestBase):
    def test_1(self):
        layouts = vera.explain.descriptive(self.region_annotations, max_panels=2)

        self.assertEqual(2, len(layouts))
