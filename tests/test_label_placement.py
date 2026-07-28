import logging
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.ops

import vera.label_placement as label_placement
from vera.label_placement import (
    convert_ax_to_data,
    count_crossings,
    leader_attachment_boundary,
    leader_endpoints,
    uncross_points,
    uncross_boxes,
    evaluate_label_pos_quality,
    apply_force_directed_layout,
    optimize_label_positions,
)


def box(cx, cy, w=1.0, h=1.0):
    return shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def centroids(polygons):
    return np.array([p.centroid.coords[0] for p in polygons])


class TestConvertAxToData(unittest.TestCase):
    """The axes are deliberately given different x and y scales, so the two
    reductions have to disagree."""

    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)

    def tearDown(self):
        plt.close("all")

    def test_max_and_min_differ(self):
        self.assertGreater(
            convert_ax_to_data(self.ax, 0.1, reduction="max"),
            convert_ax_to_data(self.ax, 0.1, reduction="min"),
        )

    def test_reductions_match_the_axis_extents(self):
        # A tenth of each axis, in data units
        self.assertAlmostEqual(10.0, convert_ax_to_data(self.ax, 0.1, "max"))
        self.assertAlmostEqual(0.1, convert_ax_to_data(self.ax, 0.1, "min"))

    def test_unknown_reduction(self):
        with self.assertRaises(ValueError):
            convert_ax_to_data(self.ax, 0.1, reduction="median")


class TestUncrossPoints(unittest.TestCase):
    def test_uncrosses(self):
        # Text positions are swapped relative to their targets, so the two
        # leader lines cross
        text_locations = np.array([[0.0, 1.0], [1.0, 1.0]])
        label_locations = np.array([[1.0, 0.0], [0.0, 0.0]])

        uncross_points(text_locations, label_locations)

        np.testing.assert_allclose([[1.0, 1.0], [0.0, 1.0]], text_locations)

    def test_swap_is_not_undone_within_a_pass(self):
        # Each unordered pair must be swapped at most once per pass; visiting
        # a pair from both directions returns it to where it started
        text_locations = np.array([[0.0, 1.0], [1.0, 1.0]])
        label_locations = np.array([[1.0, 0.0], [0.0, 0.0]])

        uncross_points(text_locations, label_locations, n_iter=1)

        np.testing.assert_allclose([[1.0, 1.0], [0.0, 1.0]], text_locations)

    def test_leaves_uncrossed_input_alone(self):
        text_locations = np.array([[0.0, 1.0], [1.0, 1.0]])
        label_locations = np.array([[0.0, 0.0], [1.0, 0.0]])

        uncross_points(text_locations, label_locations)

        np.testing.assert_allclose([[0.0, 1.0], [1.0, 1.0]], text_locations)


class OptimizerTestBase(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.embedding_region = shapely.box(-10, -10, 10, 10)

    def tearDown(self):
        plt.close("all")


class TestApplyForceDirectedLayout(OptimizerTestBase):
    def test_step_never_exceeds_max_step_norm(self):
        """The bound applies to the displacement actually applied, which
        momentum accumulates well beyond the size of a single force."""
        labels = [box(-1, 0), box(1, 0)]
        targets = [box(-6, 0), box(6, 0)]
        max_step_norm = 0.1

        _, history = apply_force_directed_layout(
            labels, targets, self.embedding_region, self.ax,
            max_step_norm=max_step_norm, lr=1, momentum=0.9, max_iter=40,
            eps=0, return_history=True,
        )

        steps = np.diff(np.array([centroids(h) for h in history]), axis=0)
        step_norms = np.linalg.norm(steps, axis=2)
        self.assertLessEqual(
            float(np.max(step_norms)), max_step_norm + 1e-6
        )

    def test_learning_rate_does_not_compound_through_momentum(self):
        """Under a constant force the displacement is a geometric series in
        momentum alone, converging to lr * force / (1 - momentum). The
        learning rate scaling the velocity buffer as well would make the ratio
        momentum * lr, which grows without bound for lr > 1."""
        force = 0.1
        lr, momentum, n_epochs = 2.0, 0.9, 30
        constant_step = np.array([[force, 0.0]])

        with mock.patch.object(
            label_placement,
            "_optimize_label_positions_update_step",
            side_effect=lambda *a, **kw: constant_step.copy(),
        ):
            _, history = apply_force_directed_layout(
                [box(0, 0)], [box(0, -5)], self.embedding_region, self.ax,
                max_step_norm=None, lr=lr, momentum=momentum,
                max_iter=n_epochs, eps=0, return_history=True,
            )

        steps = np.diff(np.array([centroids(h) for h in history]), axis=0)
        step_norms = np.linalg.norm(steps, axis=2)[:, 0]

        # Partial sums of lr * force * sum(momentum ** k)
        epochs = np.arange(1, len(step_norms) + 1)
        expected = lr * force * (1 - momentum**epochs) / (1 - momentum)
        np.testing.assert_allclose(expected, step_norms, rtol=1e-9)
        self.assertLess(float(np.max(step_norms)), lr * force / (1 - momentum))


class TestLogging(unittest.TestCase):
    """Log calls must interpolate their values rather than pass them as
    surplus arguments, which logging drops while reporting a formatting
    error."""

    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

    def tearDown(self):
        plt.close("all")

    def test_records_format(self):
        labels = [box(-1, 0), box(1, 0)]
        targets = [box(-6, 0), box(6, 0)]

        with self.assertLogs("VERA", level=logging.DEBUG) as captured:
            apply_force_directed_layout(
                labels, targets, shapely.box(-10, -10, 10, 10), self.ax,
                max_iter=3,
            )

        for record in captured.records:
            record.getMessage()  # raises if the arguments do not interpolate


class TestCountCrossings(unittest.TestCase):
    def test_counts_a_crossing_pair(self):
        text = [(-3.0, 4.0), (3.0, 4.0)]
        targets = [(3.0, -4.0), (-3.0, -4.0)]

        self.assertEqual(1, count_crossings(text, targets))

    def test_uncrossed_pair_scores_zero(self):
        text = [(-3.0, 4.0), (3.0, 4.0)]
        targets = [(-3.0, -4.0), (3.0, -4.0)]

        self.assertEqual(0, count_crossings(text, targets))

    def test_counts_each_pair_once(self):
        # Three mutually crossing leaders
        text = [(-3.0, 4.0), (3.0, 4.0), (0.0, 5.0)]
        targets = [(3.0, -4.0), (-3.0, -4.0), (0.0, -5.0)]

        self.assertLessEqual(count_crossings(text, targets), 3)
        self.assertGreaterEqual(count_crossings(text, targets), 1)


class TestScoreCrossings(unittest.TestCase):
    """The objective ignored its target regions entirely, so it could not
    distinguish a crossed layout from an uncrossed one."""

    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.targets = [box(3, -4), box(-3, -4)]
        self.crossed = [box(-3, 4), box(3, 4)]
        self.uncrossed = [box(3, 4), box(-3, 4)]

    def tearDown(self):
        plt.close("all")

    def quality(self, labels, **kwargs):
        return evaluate_label_pos_quality(
            labels, self.targets, self.targets, self.ax, **kwargs
        )

    def test_absent_by_default(self):
        self.assertNotIn("crossings", self.quality(self.crossed))

    def test_distinguishes_crossed_from_uncrossed(self):
        crossed = self.quality(self.crossed, score_crossings=True)
        uncrossed = self.quality(self.uncrossed, score_crossings=True)

        self.assertEqual(1.0, crossed["crossings"])
        self.assertEqual(0.0, uncrossed["crossings"])

    def test_other_entries_are_unchanged(self):
        without = self.quality(self.crossed)
        with_ = self.quality(self.crossed, score_crossings=True)

        self.assertEqual(without, {k: v for k, v in with_.items() if k != "crossings"})


class TestUncrossBoxes(unittest.TestCase):
    def test_boxes_trade_positions(self):
        labels = [box(-3, 4), box(3, 4)]
        targets = [box(3, -4), box(-3, -4)]

        swaps = uncross_boxes(labels, targets)

        self.assertEqual([(0, 1)], swaps)
        np.testing.assert_allclose([[3.0, 4.0], [-3.0, 4.0]], centroids(labels))

    def test_leaves_uncrossed_input_alone(self):
        labels = [box(-3, 4), box(3, 4)]
        targets = [box(-3, -4), box(3, -4)]

        swaps = uncross_boxes(labels, targets)

        self.assertEqual([], swaps)
        np.testing.assert_allclose([[-3.0, 4.0], [3.0, 4.0]], centroids(labels))

    def test_box_sizes_are_preserved(self):
        labels = [box(-3, 4, w=4, h=1), box(3, 4, w=0.5, h=0.5)]
        targets = [box(3, -4), box(-3, -4)]
        areas = sorted(l.area for l in labels)

        uncross_boxes(labels, targets)

        self.assertEqual(areas, sorted(l.area for l in labels))


class TestOptimizeLabelPositions(OptimizerTestBase):
    def identity_layout_pass(self):
        """Hold the layout still, so uncrossing alone drives the rounds."""
        return mock.patch.object(
            label_placement,
            "apply_force_directed_layout",
            side_effect=lambda labels, *a, **kw: (labels, [list(labels)]),
        )

    def record_events(self, events, uncross):
        """Patch both stages to log the order they run in."""
        def log_layout(labels, *a, **kw):
            events.append("layout")
            return labels, [list(labels)]

        def log_uncross(labels, targets, **kw):
            swaps = uncross(labels, targets)
            events.append(f"uncross({len(swaps)})")
            return swaps

        return (
            mock.patch.object(
                label_placement, "apply_force_directed_layout",
                side_effect=log_layout,
            ),
            mock.patch.object(
                label_placement, "uncross_boxes", side_effect=log_uncross
            ),
        )

    def test_no_swap_is_applied_after_the_final_layout(self):
        """The labels returned must have been settled by a layout pass. A
        trailing uncross that swaps nothing leaves them settled; one that
        swaps would not."""
        events = []
        # Always reports a swap, so the rounds run to the cap
        always_swap = lambda ls, ts, **kw: [(0, 1)]
        layout, uncross = self.record_events(events, always_swap)

        with layout, uncross:
            optimize_label_positions(
                [box(-3, 4), box(3, 4)], [box(3, -4), box(-3, -4)],
                self.embedding_region, self.ax,
                score_fn=lambda ls: 1.0, n_rounds=3,
            )

        self.assertEqual("layout", events[-1])
        self.assertEqual(3, events.count("layout"))

    def test_alternates_layout_and_uncrossing(self):
        events = []
        real = label_placement.uncross_boxes
        layout, uncross = self.record_events(events, real)

        with layout, uncross:
            optimize_label_positions(
                [box(-3, 4), box(3, 4)], [box(3, -4), box(-3, -4)],
                self.embedding_region, self.ax,
                score_fn=lambda ls: 1.0, n_rounds=3,
            )

        # Each round uncrosses then lays out; the second uncross finds
        # nothing, which ends the loop before another layout pass
        self.assertEqual(["uncross(1)", "layout", "uncross(0)"], events)

    def test_returns_best_scoring_round_not_the_last(self):
        labels = [box(-3, 4), box(3, 4)]
        targets = [box(3, -4), box(-3, -4)]
        seen = []

        def score_fn(current):
            seen.append(list(current))
            return [1.0, 9.0, 9.0][len(seen) - 1]

        # Always reports a swap, so every round runs and the best scoring one
        # is not the last
        with self.identity_layout_pass(), mock.patch.object(
            label_placement, "uncross_boxes", side_effect=lambda *a, **kw: [(0, 1)]
        ):
            best, _ = optimize_label_positions(
                labels, targets, self.embedding_region, self.ax,
                score_fn=score_fn, n_rounds=3,
            )

        self.assertEqual(3, len(seen))
        np.testing.assert_allclose(centroids(seen[0]), centroids(best))

    def test_returns_a_layout_when_every_score_is_infinite(self):
        with self.identity_layout_pass():
            best, _ = optimize_label_positions(
                [box(-3, 4)], [box(-3, -4)], self.embedding_region, self.ax,
                score_fn=lambda current: np.inf, n_rounds=3,
            )

        self.assertIsNotNone(best)

    def test_does_not_mutate_the_callers_list(self):
        labels = [box(-3, 4), box(3, 4)]
        targets = [box(3, -4), box(-3, -4)]
        before = centroids(labels)

        with self.identity_layout_pass():
            optimize_label_positions(
                labels, targets, self.embedding_region, self.ax,
                score_fn=lambda ls: 1.0, n_rounds=3,
            )

        np.testing.assert_allclose(before, centroids(labels))

    def test_rejects_zero_rounds(self):
        with self.assertRaises(AssertionError):
            optimize_label_positions(
                [box(-3, 4)], [box(-3, -4)], self.embedding_region, self.ax,
                score_fn=lambda ls: 1.0, n_rounds=0,
            )


class TestLeaderGeometry(unittest.TestCase):
    """Leaders are drawn, and crossings detected, on the nearest-point line
    the layout pulls each label along, with the attachment kept off the
    corners of the label's box."""

    def setUp(self):
        self.label = shapely.box(0, 0, 4, 1)

    def attach(self, target_x, target_y):
        point, _ = shapely.ops.nearest_points(
            leader_attachment_boundary(self.label),
            shapely.Point(target_x, target_y),
        )
        return point.coords[0]

    def test_diagonal_target_does_not_attach_at_a_corner(self):
        self.assertNotIn(self.attach(6, 3), set(self.label.exterior.coords))

    def test_attachment_stays_within_the_middle_of_an_edge(self):
        """One coordinate sits on the edge itself; the other, which runs along
        the edge, must fall inside its middle 60%."""
        for target in [(6, 3), (-4, 4), (2, -5), (6, 0.5), (-2, -3)]:
            with self.subTest(target=target):
                x, y = self.attach(*target)

                on_side = np.isclose(x, 0) or np.isclose(x, 4)
                along = y if on_side else x
                lo, hi = (0.2, 0.8) if on_side else (0.8, 3.2)
                self.assertTrue(
                    lo <= along <= hi,
                    f"attachment {(x, y)} runs to {along}, outside [{lo}, {hi}]",
                )

    def test_endpoints_lie_on_label_and_region(self):
        region = shapely.Point(8, 5).buffer(1.5)
        starts, ends = leader_endpoints([self.label], [region])

        self.assertAlmostEqual(
            0.0, self.label.boundary.distance(shapely.Point(starts[0]))
        )
        self.assertAlmostEqual(
            0.0, region.boundary.distance(shapely.Point(ends[0]))
        )

    def test_detection_uses_the_drawn_endpoints(self):
        """Two labels whose boxes do not cross by centroid, but whose drawn
        leaders do, must be seen as crossing."""
        labels = [box(-3, 4), box(3, 4)]
        targets = [shapely.Point(3, -4).buffer(0.5), shapely.Point(-3, -4).buffer(0.5)]
        starts, ends = leader_endpoints(labels, targets)

        self.assertEqual(1, count_crossings(starts, ends))
