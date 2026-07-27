import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import vera
from vera.plotting import get_cmap_colors, get_cmap_hues
from tests.test_explain import load_iris

# The three colormap shapes get_cmap_colors must handle. Note that viridis is
# a ListedColormap with 256 entries; coolwarm is the segmented one.
LISTED_CMAP = "tab10"
LISTED_CONTINUOUS_CMAP = "viridis"
SEGMENTED_CMAP = "coolwarm"
ALL_CMAPS = [LISTED_CMAP, LISTED_CONTINUOUS_CMAP, SEGMENTED_CMAP]


class TestGetCmapColors(unittest.TestCase):
    def test_listed_colormap(self):
        colors = get_cmap_colors(LISTED_CMAP)

        self.assertEqual(10, len(colors))
        self.assertEqual(
            list(matplotlib.colormaps[LISTED_CMAP].colors), list(colors)
        )

    def test_segmented_colormap_is_sampled(self):
        cmap_obj = matplotlib.colormaps[SEGMENTED_CMAP]
        self.assertFalse(hasattr(cmap_obj, "colors"))

        colors = get_cmap_colors(SEGMENTED_CMAP)

        self.assertEqual(min(cmap_obj.N, 256), len(colors))
        # Samples span the full colormap
        self.assertEqual(tuple(cmap_obj(0.0)), tuple(colors[0]))
        self.assertEqual(tuple(cmap_obj(1.0)), tuple(colors[-1]))

    def test_colors_are_valid_rgb(self):
        for cmap in ALL_CMAPS:
            with self.subTest(cmap=cmap):
                colors = get_cmap_colors(cmap)

                self.assertGreater(len(colors), 0)
                self.assertLessEqual(len(colors), 256)
                for c in colors:
                    self.assertIn(len(c), (3, 4))
                    mcolors.to_rgb(c)  # raises on anything malformed

    def test_hues(self):
        for cmap in ALL_CMAPS:
            with self.subTest(cmap=cmap):
                hues = get_cmap_hues(cmap)

                self.assertEqual(len(get_cmap_colors(cmap)), len(hues))
                self.assertTrue(np.all((hues >= 0) & (hues <= 1)))


class TestPlotAnnotationsSmoke(unittest.TestCase):
    """End-to-end render check: raw features through to a drawn figure."""

    @classmethod
    def setUpClass(cls) -> None:
        features, embedding = load_iris()
        region_annotations = vera.an.generate_region_annotations(
            features,
            embedding,
            n_discretization_bins=5,
            scale_factor=1,
            sample_size=5000,
            contour_level=0.25,
            merge_min_sample_overlap=0.5,
            random_state=0,
        )
        cls.layout = vera.explain.descriptive(region_annotations, max_panels=2)

    def tearDown(self) -> None:
        plt.close("all")

    def test_plot_annotations(self):
        for cmap in ALL_CMAPS:
            with self.subTest(cmap=cmap):
                fig, ax = vera.pl.plot_annotations(
                    self.layout, cmap=cmap, per_row=2, return_ax=True
                )

                fig.canvas.draw()
                self.assertEqual(len(self.layout), len(ax))
