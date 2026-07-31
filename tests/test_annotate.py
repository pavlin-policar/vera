import unittest

import numpy as np
import pandas as pd

import vera
from tests.test_explain import load_iris
from tests.utils import generate_clusters
from vera.rules import EqualityRule
from vera.variables import ContinuousVariable, IndicatorVariable


def _indicator_df(embedding_size: int, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a frame of pass-through indicator columns from boolean masks."""
    columns = {}
    for name, mask in masks.items():
        values = np.astype(mask, float)
        assert values.shape == (embedding_size,)
        base_variable = ContinuousVariable(name, values=values)
        rule = EqualityRule(True, value_name=name)
        columns[IndicatorVariable(base_variable, rule, values)] = values

    return pd.DataFrame(columns)


class TestFilterUninformative(unittest.TestCase):
    """`filter_uninformative` must judge indicator variables on whether their
    region says anything, since an indicator always forms a group of one."""

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(0)
        cls.embedding, _ = generate_clusters([-4, 0, 4], [0.4, 0.4, 0.4], n_samples=100)
        cls.n_samples = cls.embedding.shape[0]

        # One mask per cluster, each covering a compact third of the embedding
        cls.cluster_masks = {
            f"cluster_{i}": np.zeros(cls.n_samples, dtype=bool) for i in range(3)
        }
        for i, mask in enumerate(cls.cluster_masks.values()):
            mask[i * 100:(i + 1) * 100] = True

    def test_pass_through_indicators_survive_the_default_filter(self):
        features = _indicator_df(self.n_samples, self.cluster_masks)

        region_annotations = vera.an.generate_region_annotations(
            features, self.embedding, random_state=0
        )

        self.assertEqual(3, len(region_annotations))

    def test_degenerate_indicator_is_filtered(self):
        # True for 99% of the samples, so its region is the whole embedding
        mask = np.ones(self.n_samples, dtype=bool)
        mask[:3] = False
        features = _indicator_df(self.n_samples, {"almost_everything": mask})

        region_annotations = vera.an.generate_region_annotations(
            features, self.embedding, random_state=0
        )

        self.assertEqual(0, len(region_annotations))

    def test_localized_indicator_is_not_filtered(self):
        # True for 5% of the samples, all drawn from a single cluster
        mask = np.zeros(self.n_samples, dtype=bool)
        mask[:15] = True
        features = _indicator_df(self.n_samples, {"localized": mask})

        region_annotations = vera.an.generate_region_annotations(
            features, self.embedding, random_state=0
        )

        self.assertEqual(1, len(region_annotations))

    def test_indicator_spanning_the_embedding_is_filtered(self):
        # Matched by a third of the samples, but spread evenly enough that its
        # region takes in the whole embedding
        mask = np.zeros(self.n_samples, dtype=bool)
        mask[np.random.RandomState(0).choice(self.n_samples, 100, replace=False)] = True
        features = _indicator_df(self.n_samples, {"scattered": mask})

        region_annotations = vera.an.generate_region_annotations(
            features, self.embedding, contour_level=0.05, random_state=0
        )

        self.assertEqual(0, len(region_annotations))

    def test_coverage_threshold_is_configurable(self):
        mask = np.zeros(self.n_samples, dtype=bool)
        mask[:15] = True
        features = _indicator_df(self.n_samples, {"localized": mask})

        region_annotations = vera.an.generate_region_annotations(
            features,
            self.embedding,
            uninformative_max_sample_coverage=0.01,
            random_state=0,
        )

        self.assertEqual(0, len(region_annotations))


class TestFilterUninformativeRegression(unittest.TestCase):
    """Variables expanded into several indicators are unaffected by the
    indicator handling: they are dropped exactly when all their regions merged
    into one."""

    def test_continuous_variables_are_filtered_as_before(self):
        features, embedding = load_iris()
        kwargs = dict(
            n_discretization_bins=5,
            scale_factor=1,
            contour_level=0.25,
            merge_min_sample_overlap=0.5,
            random_state=0,
        )

        filtered = vera.an.generate_region_annotations(features, embedding, **kwargs)
        unfiltered = vera.an.generate_region_annotations(
            features, embedding, filter_uninformative=False, **kwargs
        )

        self.assertEqual([g for g in unfiltered if len(g) > 1], filtered)


if __name__ == "__main__":
    unittest.main()
