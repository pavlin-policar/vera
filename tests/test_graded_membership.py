import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from vera.plotting import (
    MEMBERSHIP_SHADING_METHODS,
    get_cmap_colors,
    oklab_to_rgb,
    plot_annotation,
    rgb_to_oklab,
)
from tests.test_descriptor_ranking import make_indicator, make_region_annotation

BACKGROUND = np.array(mcolors.to_rgb("#aaaaaa"))


def scatter_colors(region_annotations, **kwargs) -> np.ndarray:
    """Render a panel and return the RGB color drawn for each sample."""
    fig, ax = plt.subplots()
    plot_annotation(
        region_annotations,
        ax=ax,
        indicate_membership=True,
        draw_labels=False,
        **kwargs,
    )
    scatter = next(
        c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)
    )
    colors = scatter.get_facecolors()[:, :3]
    plt.close(fig)
    return colors


class TestPartialValues(unittest.TestCase):
    def test_single_indicator_matches_values(self):
        v = make_indicator("a", [1, 0, 1, 0])

        np.testing.assert_array_equal(v.values, v.partial_values)

    def test_group_averages_member_values(self):
        ra = make_region_annotation(
            [
                make_indicator("a", [1, 1, 1, 0]),
                make_indicator("b", [1, 1, 0, 0]),
                make_indicator("c", [1, 0, 0, 0]),
            ],
            n_in_region=4,
        )

        np.testing.assert_allclose(
            [1, 2 / 3, 1 / 3, 0], ra.descriptor.partial_values
        )

    def test_group_is_one_exactly_where_values_are(self):
        ra = make_region_annotation(
            [
                make_indicator("a", [1, 1, 1, 0]),
                make_indicator("b", [1, 1, 0, 0]),
            ],
            n_in_region=4,
        )

        np.testing.assert_array_equal(
            ra.descriptor.values == 1, ra.descriptor.partial_values == 1
        )


class TestMemberFractions(unittest.TestCase):
    def setUp(self):
        # The region contains samples 0-2; sample 4 lies outside it
        self.ra = make_region_annotation(
            [
                make_indicator("a", [1, 1, 1, 0, 1]),
                make_indicator("b", [1, 1, 0, 0, 1]),
            ],
            n_in_region=3,
        )

    def test_all_member_fractions(self):
        np.testing.assert_allclose(
            [1, 1, 0.5, 0, 1], self.ra.all_member_fractions
        )

    def test_contained_member_fractions_zero_outside_region(self):
        np.testing.assert_allclose(
            [1, 1, 0.5, 0, 0], self.ra.contained_member_fractions
        )


class TestGradedMembershipColors(unittest.TestCase):
    """Point colors interpolate between the background and the region color."""

    def setUp(self):
        # Samples 0-3 lie inside the region and satisfy 3, 2, 1 and 0 of the
        # three variables respectively; sample 4 lies outside it
        self.ra = make_region_annotation(
            [
                make_indicator("a", [1, 1, 1, 0, 1]),
                make_indicator("b", [1, 1, 0, 0, 1]),
                make_indicator("c", [1, 0, 0, 0, 1]),
            ],
            n_in_region=4,
        )

    def test_endpoints_match_binary_shading(self):
        graded = scatter_colors([self.ra])
        binary = scatter_colors([self.ra], graded_membership=False)

        # Full and zero membership are shaded identically either way
        np.testing.assert_allclose(binary[0], graded[0])
        np.testing.assert_allclose(binary[3], graded[3])
        # Binary shading knows only the two endpoint colors
        np.testing.assert_allclose(binary[1], binary[3])
        np.testing.assert_allclose(binary[2], binary[3])

    def test_partial_membership_lies_between(self):
        colors = scatter_colors([self.ra])
        full, background = colors[0], colors[3]

        np.testing.assert_allclose(BACKGROUND, background)
        for sample in (1, 2):
            between = (colors[sample] - background) / (full - background)
            self.assertTrue(np.all((between > 0) & (between < 1)))

    def test_shading_increases_with_membership(self):
        colors = scatter_colors([self.ra])

        distances = [np.linalg.norm(c - BACKGROUND) for c in colors[:4]]
        self.assertEqual(distances, sorted(distances, reverse=True))

    def test_points_outside_region_stay_gray(self):
        colors = scatter_colors([self.ra])

        np.testing.assert_allclose(BACKGROUND, colors[4])

    def test_points_outside_region_shaded_when_requested(self):
        colors = scatter_colors([self.ra], only_color_inside_members=False)

        np.testing.assert_allclose(colors[0], colors[4])


class TestOklab(unittest.TestCase):
    def test_roundtrip_preserves_colors(self):
        colors = np.array([mcolors.to_rgb(c) for c in get_cmap_colors("tab10")])

        np.testing.assert_allclose(
            colors, oklab_to_rgb(rgb_to_oklab(colors)), atol=1e-6
        )

    def test_lightness_ordering(self):
        # Oklab's first coordinate is perceived lightness
        lightness = rgb_to_oklab(np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]))[:, 0]

        self.assertEqual(list(lightness), sorted(lightness))

    def test_midpoint_is_perceptually_centered(self):
        """Perceived color, not channel value, is what a reader compares. The
        sRGB midpoint sits past halfway towards the color, so a half-fulfilled
        descriptor looks more complete than it is; the Oklab midpoint does not.
        """
        color = np.array([mcolors.to_rgb("tab:blue")])
        half = np.array([0.5])

        srgb_mid = MEMBERSHIP_SHADING_METHODS["linear"](color, half, BACKGROUND)
        oklab_mid = MEMBERSHIP_SHADING_METHODS["perceptual"](color, half, BACKGROUND)

        def perceived_progress(shaded):
            """How far the shaded color has travelled from gray to the color."""
            lab, lab_color = rgb_to_oklab(shaded), rgb_to_oklab(color)
            lab_background = rgb_to_oklab(BACKGROUND)
            span = np.linalg.norm(lab_color - lab_background)
            return np.linalg.norm(lab - lab_background) / span

        self.assertAlmostEqual(0.5, float(perceived_progress(oklab_mid)), places=6)
        self.assertGreater(perceived_progress(srgb_mid), 0.5)


class TestShadingMethods(unittest.TestCase):
    """Every method has to agree on the two endpoints, whatever it does between."""

    def setUp(self):
        self.color = np.tile(mcolors.to_rgb("tab:blue"), (3, 1))
        self.weights = np.array([0.0, 0.5, 1.0])

    def test_endpoints_are_fixed(self):
        for name, method in MEMBERSHIP_SHADING_METHODS.items():
            with self.subTest(method=name):
                shaded = method(self.color, self.weights, BACKGROUND)

                np.testing.assert_allclose(BACKGROUND, shaded[0], atol=1e-6)
                np.testing.assert_allclose(self.color[2], shaded[2], atol=1e-6)

    def test_shading_is_monotone(self):
        weights = np.linspace(0, 1, 21)
        colors = np.tile(mcolors.to_rgb("tab:blue"), (len(weights), 1))

        for name, method in MEMBERSHIP_SHADING_METHODS.items():
            with self.subTest(method=name):
                shaded = method(colors, weights, BACKGROUND)

                distances = np.linalg.norm(shaded - BACKGROUND, axis=1)
                self.assertTrue(np.all(np.diff(distances) >= -1e-9))

    def test_unknown_method_is_rejected(self):
        ra = make_region_annotation([make_indicator("a", [1, 0])], n_in_region=2)

        with self.assertRaises(ValueError):
            scatter_colors([ra], membership_shading="nonexistent")


class TestOverlappingAnnotations(unittest.TestCase):
    def test_strongest_membership_wins(self):
        # Both regions cover every sample, so their descriptors compete for it
        weak = make_region_annotation(
            [
                make_indicator("a", [1, 1]),
                make_indicator("b", [1, 0]),
            ],
            n_in_region=2,
        )
        strong = make_region_annotation(
            [
                make_indicator("c", [1, 1]),
                make_indicator("d", [0, 1]),
            ],
            n_in_region=2,
        )
        ra_colors = {
            weak: mcolors.to_rgb("tab:blue"),
            strong: mcolors.to_rgb("tab:orange"),
        }

        colors = scatter_colors([weak, strong], ra_colors=ra_colors)
        reversed_colors = scatter_colors([strong, weak], ra_colors=ra_colors)

        np.testing.assert_allclose(colors, reversed_colors)


if __name__ == "__main__":
    unittest.main()
