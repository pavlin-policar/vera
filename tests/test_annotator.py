import unittest

import numpy as np

import vera as annotate
import vera.embedding
import vera.explain._contrastive
import vera.explain._descriptive
import vera.preprocessing

DATA_DIR = "data"


import sys
sys.path.append("../experiments/")
import datasets


class TestAnnotator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.iris = datasets.Dataset.load("bike_sharing")

    def test_iris(self):
        x, embedding = self.iris.features, self.iris.embedding
        x, embedding = x[:5000], embedding[:5000]
        features = annotate.an.generate_explanatory_features(x, embedding)

        merged_features = annotate.pp.merge_overfragmented(features, min_sample_overlap=0.5)
        pass

    def test_bla(self):
        embedding, x = self.iris.embedding, self.iris.features

        # Desired API
        features = vera.an.generate_explanatory_features()
        features = vera.an.filter_explanatory_features(features)

        feature_groups = vera.an.group_similar_features(features)
        feature_groups = vera.an.filter_explanatory_features(feature_groups)

        layouts = vera.an.find_layouts(feature_groups)

        # Find contrastive features
        vera.explain.contrastive.contrastive(features)
        # find other
        vera.explain.descriptive.descriptive(features)


        layouts = vera.explain.contrastive.contrastive(layouts)
        vera.pl.layouts(layouts)

        # END

        features = vera.pp.generate_explanatory_features(x, embedding)

        layouts = vasari.explain(features, embedding)
        layouts = vera.explain.contrastive.contrastive(layouts)
        vasari.plot.layouts(layouts, embedding)

        # Or, simlpy
        layouts = vasari.explain(features, embedding)
        layouts = vera.explain.contrastive.contrastive(layouts)
        vasari.plot.layouts(layouts, embedding)

        # Current implementation
        features = vera.preprocessing.generate_explanatory_features(x)

        k_neighbors = int(np.floor(np.sqrt(embedding.shape[0])))
        scale = vera.embedding.kth_neighbor_distance(embedding, k_neighbors)

        candidates = annotate.fs.morans_i(embedding, features, scale=scale)

        feature_densities = vera.preprocessing.estimate_feature_densities(
            candidates["feature"].tolist(),
            features,
            embedding,
            bw=scale,
        )
        regions = vera.preprocessing.find_regions(feature_densities, level=0.25)

        merged_regions = annotate.an.stage_1_merge_regions(regions, overlap_threshold=0.75)

        clusters, cluster_densities = annotate.an.group_similar_variables(
            merged_regions.tolist(),
            merged_regions,
            threshold=0.85,
            method="connected-components",
        )

        # annotate.pl.plot_feature_densities(
        #     list(clusters),
        #     cluster_densities,
        #     embedding=embedding,
        #     levels=4,
        #     skip_first=False,
        #     per_row=1,
        #     figwidth=4,
        # )
        # import matplotlib.pyplot as plt
        #
        # plt.tight_layout()
        # plt.show()
