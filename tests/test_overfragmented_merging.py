import unittest

import numpy as np

import vera
from tests.utils import generate_clusters


class TestRegion(unittest.TestCase):
    def test_merge_1(self):
        """Two completely non-overlapping clusters."""
        np.random.seed(0)
        x, features = generate_clusters([1, -1], [0.25, 0.25], n_samples=50)

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5,
        )

        self.assertEqual(
            1,
            len(region_annotations),
            "Incorrect number of variables returned",
        )
        self.assertEqual(
            2,
            len(region_annotations[0]),
            "Incorrect number of region annotations returned",
        )

    def test_merge_2(self):
        """Four non-overlapping distributions."""
        np.random.seed(0)
        # Two separated Gaussians. These should not be merged
        x, features = generate_clusters(
            [[1, 1], [1, -1], [-1, -1], [-1, 1]], [0.25] * 4, n_samples=50
        )

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5,
        )

        # cluster_ras = ra_collection["cluster"]
        # vera.plotting.plot_regions_with_subregions(
        #     cluster.explanatory_variables, show=True, per_row=2, figwidth=8
        # )

        self.assertEqual(
            1,
            len(region_annotations),
            "Incorrect number of variables returned",
        )
        self.assertEqual(
            4,
            len(region_annotations[0]),
            "Incorrect number of region annotations returned",
        )

    def test_merge_3(self):
        """Four distributions: three overlapping, one separate."""
        np.random.seed(0)
        # Two separated Gaussians. These should not be merged
        x, features = generate_clusters([1, 1, 1, -1], [0.25] * 4, n_samples=50)

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5,
        )

        # cluster_ras = ra_collection["cluster"]
        # vera.plotting.plot_regions_with_subregions(
        #     cluster.explanatory_variables, show=True, per_row=2, figwidth=8
        # )

        self.assertEqual(
            1,
            len(region_annotations),
            "Incorrect number of variables returned",
        )
        self.assertEqual(
            2,
            len(region_annotations[0]),
            "Incorrect number of region annotations returned",
        )

    def test_merge_4(self):
        """Two semi-overlapping distributions"""
        np.random.seed(0)
        # Two slightly-overlapping Gaussians. These should not be merged
        x, features = generate_clusters([0.5, -0.5], [1] * 2, n_samples=100)

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5,
        )

        # cluster_ras = ra_collection["cluster"]
        # vera.plotting.plot_regions_with_subregions(
        #     cluster.explanatory_variables, show=True, per_row=2, figwidth=8
        # )

        self.assertEqual(
            1,
            len(region_annotations),
            "Incorrect number of variables returned",
        )
        self.assertEqual(
            2,
            len(region_annotations[0]),
            "Incorrect number of region annotations returned",
        )

    def test_merge_5(self):
        """Three semi-overlapping distributions"""
        np.random.seed(0)
        # Three separated, but partially-overlapping Gaussians. These should not
        # be merged
        x, features = generate_clusters([0.5, 0, -0.5], [0.25] * 3, n_samples=100)

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5,
        )

        # cluster_ras = ra_collection["cluster"]
        # vera.plotting.plot_regions_with_subregions(
        #     cluster.explanatory_variables, show=True, per_row=2, figwidth=8
        # )
        # vera.plotting.plot_annotation(
        #     cluster.explanatory_variables, show=True, figwidth=8
        # )
        self.assertEqual(
            1,
            len(region_annotations),
            "Incorrect number of variables returned",
        )
        self.assertEqual(
            3,
            len(region_annotations[0]),
            "Incorrect number of region annotations returned",
        )

    def test_merge_6(self):
        """Four semi-overlapping distributions, where we have to resolve a
        max-clique conflict."""
        np.random.seed(0)
        # Construct three partially-overlapping Gaussians. These should not be merged
        x, features = generate_clusters(
            [1, 0, -1], [1.3] * 3, n_samples=100
        )

        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5,
        )

        # cluster_ras = ra_collection["cluster"]
        # vera.plotting.plot_regions_with_subregions(cluster_ras, show=True, per_row=2, figwidth=8)
        # vera.plotting.plot_annotation(cluster_ras, show=True, figwidth=8)

        self.assertEqual(
            1,
            len(region_annotations),
            "Incorrect number of variables returned",
        )
        self.assertEqual(
            3,
            len(region_annotations[0]),
            "Incorrect number of region annotations returned",
        )
