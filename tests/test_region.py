import unittest

import numpy as np
import pandas as pd
import shapely

import vera
import vera.embedding
import vera.region


class TestIntersectWithOther(unittest.TestCase):
    def test_construct_region_with_single_hole(self):
        np.random.seed(0)

        x = np.random.uniform(-1, 1, size=(500, 2))

        features = pd.DataFrame()
        features["inner"] = np.linalg.norm(x, axis=1) < 0.5

        # The function returns the base variables, of which there should only be one
        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5, sample_size=None,
        )
        self.assertEqual(
            1,
            len(region_annotations),
            "`an.generate_explanatory_features` returned more than one variable"
        )
        self.assertEqual(
            2,
            len(region_annotations[0]),
            "The number of explanatory features was not 2!"
        )
        outer, inner = region_annotations[0]

        outer_region, inner_region = outer.region, inner.region
        self.assertEqual(
            1,
            len(inner_region.polygon.geoms),
            "Inner region should be comprised of a single polygon!"
        )
        self.assertEqual(
            1,
            len(outer_region.polygon.geoms),
            "Outer region should be comprised of a single polygon!"
        )
        self.assertEqual(len(inner_region.polygon.geoms[0].interiors), 0)
        self.assertEqual(len(outer_region.polygon.geoms[0].interiors), 1)

    def test_construct_region_with_multiple_holes(self):
        np.random.seed(0)

        x = np.random.uniform(-2, 2, size=(500, 2))

        features = pd.DataFrame()
        hole1 = np.linalg.norm(x - [1, 1], axis=1) < 0.5
        hole2 = np.linalg.norm(x - [-1, 1], axis=1) < 0.5
        hole3 = np.linalg.norm(x - [1, -1], axis=1) < 0.5
        hole4 = np.linalg.norm(x - [-1, -1], axis=1) < 0.5
        features["holes"] = hole1 | hole2 | hole3 | hole4
        features["holes"] = features["holes"]

        # The function returns the base variables, of which there should only be one
        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5, sample_size=None,
        )
        self.assertEqual(
            1,
            len(region_annotations),
            "`an.generate_explanatory_features` returned more than one variable"
        )
        self.assertEqual(
            2,
            len(region_annotations[0]),
            "The number of explanatory features was not 2!"
        )
        outer, inner = region_annotations[0]

        outer_region, inner_region = outer.region, inner.region
        self.assertEqual(
            4,
            len(inner_region.polygon.geoms),
            "Inner region should be comprised of four polygons!"
        )
        self.assertEqual(
            1,
            len(outer_region.polygon.geoms),
            "Outer region should be comprised of a single polygon!"
        )
        self.assertEqual(0, len(inner_region.polygon.geoms[0].interiors))
        self.assertEqual(4, len(outer_region.polygon.geoms[0].interiors))


class TestFromTriangulation(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.x = np.random.uniform(-1, 1, size=(500, 2))
        self.embedding = vera.embedding.Embedding(self.x, scale_factor=0.5)

    def test_members_without_triangles_get_disks(self):
        for n_members in (1, 2):
            mask = np.zeros(len(self.x))
            mask[:n_members] = 1
            region = vera.region.Region.from_triangulation(self.embedding, mask)
            self.assertEqual(n_members, region.num_parts)
            for i in range(n_members):
                disk = shapely.geometry.Point(self.x[i]).buffer(self.embedding.scale)
                self.assertTrue(region.polygon.covers(shapely.geometry.Point(self.x[i])))
                self.assertAlmostEqual(disk.area, region.polygon.geoms[i].area)

    def test_pruned_triangles_give_disks(self):
        # Three members too far apart to share a triangle edge
        x = np.vstack([self.x, [[-10, -10], [10, -10], [0, 10]]])
        embedding = vera.embedding.Embedding(x, scale_factor=0.5)
        mask = np.zeros(len(x))
        mask[-3:] = 1
        region = vera.region.Region.from_triangulation(embedding, mask)
        self.assertEqual(3, region.num_parts)
        self.assertEqual({500, 501, 502}, region.contained_samples)

    def test_no_members_gives_empty_region(self):
        region = vera.region.Region.from_triangulation(self.embedding, np.zeros(len(self.x)))
        self.assertTrue(region.polygon.is_empty)

    def test_single_member_bin_is_merged_into_surrounding_bin(self):
        # One sample with a value of its own, embedded among the samples of
        # the adjacent bin
        x = self.x
        values = np.where(np.arange(len(x)) == 0, 100.0, np.random.uniform(0, 1, len(x)))
        x[0] = [0, 0]
        features = pd.DataFrame({"value": values})
        region_annotations = vera.an.generate_region_annotations(
            features, embedding=x, scale_factor=0.5, sample_size=None,
            region_method="rangeset", n_discretization_bins=2,
            filter_uninformative=False,
        )
        self.assertEqual(1, len(region_annotations[0]))
        self.assertIn(0, region_annotations[0][0].all_members)
