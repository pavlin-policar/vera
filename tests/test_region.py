import unittest

import numpy as np
import pandas as pd

import veca.preprocessing
import veca.annotate
import matplotlib.pyplot as plt


# from veca.preprocessing import estimate_feature_densities, find_regions


def plot(x, y=None, show=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    c = "k"
    if y is not None:
        c = y

    ax.scatter(x[:, 0], x[:, 1], c=c)

    if show:
        plt.show()


class TestRegion(unittest.TestCase):
    def test_construct_region_with_single_hole(self):
        np.random.seed(0)

        x = np.random.uniform(-1, 1, size=(500, 2))

        features = pd.DataFrame()
        features["inner"] = np.linalg.norm(x, axis=1) < 0.5

        # The function returns the base variables, of which there should only be one
        feature_list = veca.an.generate_explanatory_features(
            features, embedding=x, scale_factor=0.5,
        )
        plt.show()
        self.assertEqual(
            len(feature_list), 1,
            "`an.generate_explanatory_features` returned more than one variable"
        )
        self.assertEqual(
            len(feature_list[0].explanatory_variables), 2,
            "The number of explanatory features was not 2!"
        )
        outer, inner = feature_list[0].explanatory_variables

        outer_region, inner_region = outer.region, inner.region
        self.assertEqual(
            len(inner_region.polygon.geoms), 1,
            "Inner region should be comprised of a single polygon!"
        )
        self.assertEqual(
            len(outer_region.polygon.geoms), 1,
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
        features["holes"] = features["holes"].astype("category")

        # The function returns the base variables, of which there should only be one
        feature_list = veca.an.generate_explanatory_features(
            features, embedding=x, scale_factor=0.5,
        )
        plt.show()
        self.assertEqual(
            len(feature_list), 1,
            "`an.generate_explanatory_features` returned more than one variable"
        )
        self.assertEqual(
            len(feature_list[0].explanatory_variables), 2,
            "The number of explanatory features was not 2!"
        )
        outer, inner = feature_list[0].explanatory_variables

        outer_region, inner_region = outer.region, inner.region
        self.assertEqual(
            len(inner_region.polygon.geoms), 4,
            "Inner region should be comprised of four polygons!"
        )
        self.assertEqual(
            len(outer_region.polygon.geoms), 1,
            "Outer region should be comprised of a single polygon!"
        )
        self.assertEqual(len(inner_region.polygon.geoms[0].interiors), 0)
        self.assertEqual(len(outer_region.polygon.geoms[0].interiors), 4)
