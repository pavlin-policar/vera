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


class _StubAnnotation:
    """A region annotation reduced to what `layout_variable_colors` reads.

    The hash is set explicitly so a test can reproduce what differing string
    hash salts do to real region annotations across processes: the same
    layout, its objects hashing differently.
    """

    def __init__(self, descriptor, hash_value):
        self.descriptor = descriptor
        self.hash_value = hash_value

    def __hash__(self):
        return self.hash_value


def _stub_layout(panels, hash_order):
    """Build a layout of stubs from descriptor names, one hash per stub."""
    hashes = iter(hash_order)
    return [[_StubAnnotation(d, next(hashes)) for d in panel] for panel in panels]


def _colors_by_position(layout, colors):
    return [[colors[ra] for ra in panel] for panel in layout]


class TestLayoutVariableColors(unittest.TestCase):
    PANELS = [["c", "a"], ["b", "a", "d"]]

    def test_colors_independent_of_hashes(self):
        n = sum(len(panel) for panel in self.PANELS)
        ascending = _stub_layout(self.PANELS, range(n))
        descending = _stub_layout(self.PANELS, reversed(range(n)))

        self.assertEqual(
            _colors_by_position(ascending, vera.pl.layout_variable_colors(ascending)),
            _colors_by_position(descending, vera.pl.layout_variable_colors(descending)),
        )

    def test_descriptors_colored_in_order_of_first_appearance(self):
        layout = _stub_layout(self.PANELS, range(5))
        colors = vera.pl.layout_variable_colors(layout, cmap="tab10")

        palette = [mcolors.to_rgb(c) for c in get_cmap_colors("tab10")]
        by_descriptor = {ra.descriptor: colors[ra] for panel in layout for ra in panel}
        # Repeats of "a" neither take a second color nor shift later descriptors
        self.assertEqual(
            {"c": palette[0], "a": palette[1], "b": palette[2], "d": palette[3]},
            by_descriptor,
        )

    def test_glasbey_extension_is_stable(self):
        panels = [[f"d{i}" for i in range(7)], [f"d{i}" for i in range(7, 14)]]
        ascending = _stub_layout(panels, range(14))
        descending = _stub_layout(panels, reversed(range(14)))

        colors = vera.pl.layout_variable_colors(ascending, cmap="tab10")
        self.assertEqual(14, len(set(colors.values())))
        self.assertEqual(
            _colors_by_position(ascending, colors),
            _colors_by_position(
                descending, vera.pl.layout_variable_colors(descending, cmap="tab10")
            ),
        )


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
