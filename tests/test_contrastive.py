import sys
import unittest

sys.path.append("../experiments/")
import datasets
import vera


class TestDescriptiveRanking(unittest.TestCase):
    def test_1(self):
        data = datasets.Dataset.load("iris")
        x, embedding = data.features, data.embedding

        region_annotations = vera.an.generate_region_annotations(
            x,
            embedding,
            n_discretization_bins=10,
            scale_factor=1,
            sample_size=5000,
            contour_level=0.25,
            merge_min_sample_overlap=0.5,
            merge_min_purity_gain=0.5,
        )

        layouts = vera.explain.contrastive(region_annotations, max_panels=2)

        self.assertEqual(2, len(layouts))

        # vera.pl.plot_annotations(layouts, show=True, per_row=1, figwidth=6)
