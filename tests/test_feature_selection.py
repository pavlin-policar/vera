import unittest

import openTSNE
import numpy as np
import pandas as pd
from sklearn import datasets

import vera.preprocessing
from vera import data, pl
from vera.preprocessing import estimate_feature_densities, find_regions
from vera.embedding import kth_neighbor_distance
from vera.feature_selection import feature_merge


class TestMergeFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        iris = datasets.load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        cls.iris = df

        tsne = openTSNE.TSNE(metric="cosine", perplexity=30)
        cls.embedding = tsne.fit(df.values)

    def test_iris_sepal_length_5_clusters(self):
        k_neighbors = int(np.floor(np.sqrt(self.iris.shape[0])))
        scale = kth_neighbor_distance(self.embedding, k_neighbors)

        features = self.iris[["petal length (cm)"]]
        features = vera.preprocessing._discretize(features, n_bins=10)

        merged_features = feature_merge(features, self.embedding, scale)

        feature_densities = estimate_feature_densities(
            merged_features.columns.tolist(),
            merged_features,
            self.embedding,
            bw=scale,
        )
        regions = find_regions(feature_densities)

        import matplotlib.pyplot as plt
        pl.plot_regions(regions.keys(), regions, self.embedding, per_row=1, figwidth=4)
        plt.tight_layout()
        plt.show()
