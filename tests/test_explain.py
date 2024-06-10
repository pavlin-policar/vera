import sys
import unittest
from collections import defaultdict

import vera.explain._descriptive

sys.path.append("../experiments/")
import datasets
import numpy as np
import vera
import scipy.stats as stats


def mean_variable_occurence(layout):
    """Count the number of times each base variable appears in a layout."""
    base_variable_count = defaultdict(int)
    for variable_group in layout:
        for variable in variable_group.variable_dict:
            base_variable_count[variable.base_variable] += 1

    return np.mean(list(base_variable_count.values()))


class TestDescriptiveRanking(unittest.TestCase):
    def test_1(self):
        data = datasets.Dataset.load("titanic")
        x, embedding = data.features, data.embedding

        variables = vera.an.generate_explanatory_features(
            x,
            embedding,
            n_discretization_bins=10,
            scale_factor=1,
            sample_size=5000,
            contour_level=0.25,
            merge_min_sample_overlap=0.5,
            merge_min_purity_gain=0.5,
            merge_min_geary_gain=0  # -1,
        )

        # Extract actual explanatory variables
        explanatory_features = [ex for v in variables for ex in v.region_annotations]

        # Split explanatory variables into subgroups
        all_features = []
        for e in explanatory_features:
            all_features.extend(e.split_region())

        clusters = vera.explain.descriptive.group_similar_variables(all_features, threshold=0.8)
        filtered_clusters = vera.explain.descriptive.filter_explanatory_features(clusters)

        # Begin selection
        selected_layouts = []
        while len(filtered_clusters):
            layouts = vera.explain.descriptive.find_layouts(filtered_clusters, max_overlap=0)

            # Find the best layout
            # Filter layouts, so they can contain at most 5 regions
            layouts = [l for l in layouts if len(l) <= 5]

            mean_variable_occurences = np.array([
                mean_variable_occurence(layout) for layout in layouts
            ])
            ranks = stats.rankdata(mean_variable_occurences, method="average")

            best_layout_idx = np.argmax(ranks)
            # END

            selected_layout = layouts[best_layout_idx]
            selected_layouts.append(selected_layout)

            for cluster in selected_layout:
                filtered_clusters.remove(cluster)

        vera.pl.plot_annotations(selected_layouts, show=True, per_row=3, figwidth=6)

    def test_2_iris(self):
        data = datasets.Dataset.load("iris")
        x, embedding = data.features, data.embedding

        variables = vera.an.generate_explanatory_features(
            x,
            embedding,
            n_discretization_bins=10,
            scale_factor=1,
            sample_size=5000,
            contour_level=0.25,
            merge_min_sample_overlap=0.5,
            merge_min_purity_gain=0.5,
            merge_min_geary_gain=0  # -1,
        )

        # Extract actual explanatory variables
        explanatory_features = [ex for v in variables for ex in v.region_annotations]

        # Split explanatory variables into subgroups
        all_features = []
        for e in explanatory_features:
            all_features.extend(e.split_region())

        filtered_features = vera.explain.descriptive.filter_explanatory_features(all_features)

        clusters = vera.explain.descriptive.group_similar_variables(filtered_features, threshold=0.8)
        filtered_clusters = vera.explain.descriptive.filter_explanatory_features(clusters)

        for cluster in clusters:
            print("\nCluster")
            for v in cluster.variable_dict:
                print(v.rule)
