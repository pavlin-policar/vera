import unittest

import numpy as np
import pandas as pd

import vera
from vera.embedding import Embedding
from tests.utils import generate_clusters


class TestSampleWeights(unittest.TestCase):
    """Sample weights multiply the KDE weights and densities are contoured
    relative to the unweighted peak, so weights shrink or remove regions
    without changing which samples a region's rule describes."""

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(0)
        cls.embedding, clusters = generate_clusters(
            [-4, 0, 4], [0.4, 0.4, 0.4], n_samples=100
        )
        cls.cluster = clusters["cluster"].to_numpy().astype(int)
        cls.n_samples = cls.embedding.shape[0]
        # One indicator spans the two outer clusters, another the middle one
        cls.features = pd.DataFrame({
            "outer": cls.cluster != 1,
            "middle": cls.cluster == 1,
        })

    def _regions(self, **kwargs):
        ras = vera.an.generate_region_annotations(
            self.features,
            self.embedding,
            indicator_columns="all",
            filter_uninformative=False,
            random_state=0,
            **kwargs,
        )
        return {str(ra_group[0].descriptor): ra_group for ra_group in ras}

    def _covered_clusters(self, ra_group):
        members = set().union(*(ra.contained_samples for ra in ra_group))
        return set(np.unique(self.cluster[list(members)]))

    def test_unit_weights_leave_regions_unchanged(self):
        unweighted = self._regions()
        weighted = self._regions(sample_weights=np.ones(self.n_samples))
        self.assertEqual(unweighted.keys(), weighted.keys())
        for key in unweighted:
            for ra1, ra2 in zip(unweighted[key], weighted[key]):
                self.assertTrue(ra1.region.polygon.equals(ra2.region.polygon))

    def test_uniform_weights_shrink_regions(self):
        unweighted = self._regions()
        weighted = self._regions(sample_weights=np.full(self.n_samples, 0.5))
        self.assertEqual(unweighted.keys(), weighted.keys())
        for key in unweighted:
            p1 = unweighted[key][0].region.polygon
            p2 = weighted[key][0].region.polygon
            self.assertLess(p2.area, p1.area)
            self.assertTrue(p1.buffer(1e-9).contains(p2))

    def test_uniform_weights_below_contour_level_remove_regions(self):
        weighted = self._regions(
            sample_weights=np.full(self.n_samples, 0.2), contour_level=0.25
        )
        self.assertEqual(weighted, {})

    def test_poorly_weighted_variable_loses_its_region(self):
        weights = np.where(self.cluster == 1, 0.1, 1.0)
        weighted = self._regions(sample_weights=weights)
        self.assertNotIn("middle", weighted)
        self.assertIn("outer", weighted)

    def test_zero_weight_removes_a_cluster_from_the_region(self):
        self.assertEqual(self._covered_clusters(self._regions()["outer"]), {0, 2})

        weights = np.where(self.cluster == 2, 0.0, 1.0)
        weighted = self._regions(sample_weights=weights)
        self.assertEqual(self._covered_clusters(weighted["outer"]), {0})

    def test_variable_with_no_weight_yields_no_region(self):
        weights = np.where(self.cluster == 1, 0.0, 1.0)
        weighted = self._regions(sample_weights=weights)
        self.assertNotIn("middle", weighted)
        self.assertIn("outer", weighted)

    def test_weights_do_not_make_a_variable_informative(self):
        # Half of every cluster, so the region spans nearly the whole
        # embedding, about 90% of the samples
        rng = np.random.RandomState(0)
        features = self.features.assign(scattered=rng.rand(self.n_samples) < 0.5)
        weights = np.where(self.cluster == 2, 0.05, 1.0)

        def regions(**kwargs):
            ras = vera.an.generate_region_annotations(
                features, self.embedding, indicator_columns="all",
                uninformative_max_sample_coverage=0.85, random_state=0, **kwargs,
            )
            return {str(ra_group[0].descriptor) for ra_group in ras}

        self.assertNotIn("scattered", regions())
        self.assertNotIn("scattered", regions(sample_weights=weights))
        self.assertIn(
            "scattered", regions(sample_weights=weights, filter_uninformative=False)
        )

    def test_weights_follow_subsampling(self):
        weights = np.where(self.cluster == 2, 0.0, 1.0)
        weighted = self._regions(
            sample_weights=weights, sample_size=self.n_samples // 2
        )
        ra_group = weighted["outer"]
        embedding_x = ra_group[0].region.embedding.X[:, 0]
        members = set().union(*(ra.contained_samples for ra in ra_group))
        # Cluster 2 sits at x = 4 and carries no weight
        self.assertTrue(np.all(embedding_x[list(members)] < 2))

    def test_invalid_weights_are_rejected(self):
        for weights in [
            np.ones(self.n_samples - 1),
            -np.ones(self.n_samples),
            np.full(self.n_samples, np.nan),
            np.zeros(self.n_samples),
            np.full(self.n_samples, 1.5),
        ]:
            with self.assertRaises(ValueError):
                self._regions(sample_weights=weights)

    def test_rangeset_rejects_weights(self):
        with self.assertRaises(ValueError):
            self._regions(
                sample_weights=np.ones(self.n_samples), region_method="rangeset"
            )

    def test_embedding_rejects_weights_twice(self):
        embedding = Embedding(self.embedding, sample_weights=np.ones(self.n_samples))
        variables = vera.pp.expand_df(self.features, indicator_columns="all")
        with self.assertRaises(ValueError):
            vera.pp.extract_region_annotations(
                variables, embedding, sample_weights=np.ones(self.n_samples)
            )


if __name__ == "__main__":
    unittest.main()
