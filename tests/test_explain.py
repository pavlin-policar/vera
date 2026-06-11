import os
import unittest

import pandas as pd

import vera

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "iris")


def load_iris():
    """Load the vendored iris fixture: features and a precomputed 2D embedding."""
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    embedding = pd.read_csv(
        os.path.join(DATA_DIR, "embedding.csv"), header=None
    ).values
    return features, embedding


class ExplainTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features, cls.embedding = load_iris()
        cls.region_annotations = vera.an.generate_region_annotations(
            cls.features,
            cls.embedding,
            n_discretization_bins=5,
            scale_factor=1,
            sample_size=5000,
            contour_level=0.25,
            merge_min_sample_overlap=0.5,
            random_state=0,
        )


class TestContrastiveRanking(ExplainTestBase):
    def test_1(self):
        layouts = vera.explain.contrastive(self.region_annotations, max_panels=2)

        self.assertEqual(2, len(layouts))


class TestDescriptiveRanking(ExplainTestBase):
    def test_1(self):
        layouts = vera.explain.descriptive(self.region_annotations, max_panels=2)

        self.assertEqual(2, len(layouts))
