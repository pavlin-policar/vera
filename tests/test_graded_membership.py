import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from vera.plotting import plot_annotation
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
