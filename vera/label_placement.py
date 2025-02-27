import io
import logging

import matplotlib
import numpy as np
import shapely
import shapely.affinity
import shapely.ops
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances

try:
    from matplotlib.backend_bases import _get_renderer as matplot_get_renderer
except ImportError:
    matplot_get_renderer = None


logger = logging.getLogger("VERA")

# Define useful types
BoundingBox = tuple[float, float, float, float]  # (x0, y0, x1, y1)


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def intersect(x0, y0, x1, y1):
    return ccw(x0, y0, y1) != ccw(x1, y0, y1) and ccw(x0, x1, y0) != ccw(x0, x1, y1)


def fix_crossings(text_locations, label_locations, n_iter=3):
    """Find crossing lines and swap labels; repeat as required"""
    for n in range(n_iter):
        for i in range(text_locations.shape[0]):
            for j in range(text_locations.shape[0]):
                if intersect(
                    text_locations[i],
                    text_locations[j],
                    label_locations[i],
                    label_locations[j],
                ):
                    swap = text_locations[i].copy()
                    text_locations[i] = text_locations[j]
                    text_locations[j] = swap


# From adjustText (https://github.com/Phlya/adjustText)
def get_renderer(fig):
    # If the backend support get_renderer() or renderer, use that.
    if hasattr(fig.canvas, "get_renderer"):
        return fig.canvas.get_renderer()

    if hasattr(fig.canvas, "renderer"):
        return fig.canvas.renderer

    # Otherwise, if we have the matplotlib function available, use that.
    if matplot_get_renderer:
        return matplot_get_renderer(fig)

    # No dice, try and guess.
    # Write the figure to a temp location, and then retrieve whichever
    # render was used (doesn't work in all matplotlib versions).
    fig.canvas.print_figure(io.BytesIO())
    try:
        return fig._cachedRenderer

    except AttributeError:
        # No luck.
        # We're out of options.
        raise ValueError("Unable to determine renderer") from None


# Adapted from adjustText (https://github.com/Phlya/adjustText)
def get_artist_bounding_box(obj, ax, expand=(1, 1)) -> BoundingBox:
    """Get a mpl's artists bounding box in data units."""
    r = get_renderer(ax.get_figure())
    bbox = obj.get_window_extent(r).expanded(*expand)

    # Convert display coords to data coordinate space
    display_to_data = ax.transData.inverted()
    bl = display_to_data.transform([bbox.xmin, bbox.ymin])
    tr = display_to_data.transform([bbox.xmax, bbox.ymax])

    return np.hstack([bl, tr])


def get_artist_bounding_boxes(objs, ax, expand=(1, 1)) -> list[BoundingBox]:
    """Get a list of  mpl's artists bounding boxes in data units."""
    return np.vstack([get_artist_bounding_box(obj, ax, expand) for obj in objs])


# Adapted from datamapplot (https://github.com/TutteInstitute/datamapplot)
def initial_text_location_placement(
    embedding, label_targets, label_radius=None, radius_factor=0.25,
):
    """Find an initial placement for labels.

    Parameters
    ----------
    embedding : np.ndarray
    label_targets : np.ndarray
        Initial label positions, where the labels are pointing at.
    label_radius : float, optional
        If provided, all the labels will be at this distance from the middle of
        the embedding. If this parameter is set, `radius_factor` will be ignored
    radius_factor : float, optional
        If `label_radius` is not provided, the labels will be placed at
        (1 + radius_factor) * max(distance from center) around the plot.

    Returns
    -------
    np.ndarray

    """
    # Center the labels
    embedding_center_point = (
        np.min(embedding, axis=0) + np.max(embedding, axis=0)
    ) / 2
    label_targets = label_targets - embedding_center_point

    if label_radius is None:
        centered_embedding = embedding - embedding_center_point
        dists_from_origin = np.linalg.norm(centered_embedding, axis=1)
        label_radius = np.max(dists_from_origin) * (1 + radius_factor)

    # Determine the angles of the label positions
    label_thetas = np.arctan2(label_targets.T[0], label_targets.T[1])

    # Construct a ring of possible label placements around the embedding, we
    # refer to these as spokes
    xs = np.linspace(0, 1, max(len(label_thetas) + 1, 8), endpoint=False)
    spoke_thetas = xs * 2 * np.pi

    # Rotate the spokes little by little and see how well it matches the label
    # locations, and select the rotation which achieves the best matching
    best_rotation = 0
    min_score = np.inf
    best_label_spoke_pairing = None

    for rotation in np.linspace(
        -np.pi / int(len(label_thetas) + 5), np.pi / int(len(label_thetas) + 5), 32
    ):
        # We can use unit vectors since we're calcualting cosine similarity
        test_spoke_label_locations = np.vstack([
            np.cos(spoke_thetas + rotation), np.sin(spoke_thetas + rotation),
        ]).T

        # Determine an initial ordering on how to match labels to their spokes
        # By default, we will match the closest ones first
        distances = pairwise_distances(
            label_targets, test_spoke_label_locations, metric="cosine"
)
        # Find and sort the labels by their minimum distance to the nearest spoke
        best_dists_to_labels = np.min(distances, axis=1)
        label_order = np.argsort(best_dists_to_labels)

        spokes_taken = set()
        label_spoke_pairing = {}
        for label_idx in label_order:
            # Compute the distance from the current label to the remaining available
            # spoke locations
            candidates = list(set(range(test_spoke_label_locations.shape[0])) - spokes_taken)
            candidate_distances = pairwise_distances(
                [label_targets[label_idx]],
                test_spoke_label_locations[candidates],
                metric="cosine",
            )
            closest_spoke = candidates[np.argmin(candidate_distances[0])]
            label_spoke_pairing[label_idx] = closest_spoke
            spokes_taken.add(closest_spoke)

        # Having assigned each label its closest available spoke, calculate the
        # average distance for each pair
        cosine_dists = np.array([
            distance.cosine(label_targets[k], test_spoke_label_locations[v])
            for k, v in label_spoke_pairing.items()
        ])
        score = np.mean(cosine_dists)

        # The score is the sum of the distances to the nearest spoke
        if score < min_score:
            min_score = score
            best_rotation = rotation
            best_label_spoke_pairing = label_spoke_pairing

    # Convert the spoke locations to cartesian coordinates
    spoke_label_locations = np.vstack([
        label_radius * np.cos(spoke_thetas + best_rotation),
        label_radius * np.sin(spoke_thetas + best_rotation),
    ]).T

    label_targets = np.asarray([
        spoke_label_locations[best_label_spoke_pairing[i]]
        for i in sorted(best_label_spoke_pairing.keys())
    ])

    # Un-center label positions
    return label_targets + embedding_center_point


def get_ax_bounding_box(ax: matplotlib.axes.Axes) -> BoundingBox:
    ax_x_min, ax_x_max = ax.get_xlim()
    ax_y_min, ax_y_max = ax.get_ylim()
    return ax_x_min, ax_y_min, ax_x_max, ax_y_max


def set_ax_bounding_box(ax: matplotlib.axes.Axes, bbox: BoundingBox) -> None:
    x_min, y_min, x_max, y_max = bbox
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def set_bbox_square_aspect(bbox: BoundingBox) -> BoundingBox:
    """Convert given bounding box coords to square aspect ratio, preserving the
    longer side"""
    x_min, y_min, x_max, y_max = bbox

    x_span = x_max - x_min
    y_span = y_max - y_min
    longer_span = max(x_span, y_span)

    # How much do we need to add to the shorter span to match the longer one
    x_span_diff = longer_span - x_span
    y_span_diff = longer_span - y_span

    new_x_min = x_min - x_span_diff / 2
    new_x_max = x_max + x_span_diff / 2
    new_y_min = y_min - y_span_diff / 2
    new_y_max = y_max + y_span_diff / 2

    return new_x_min, new_y_min, new_x_max, new_y_max


def add_bbox_padding(bbox: BoundingBox, padding=(1, 1)) -> BoundingBox:
    x_min, y_min, x_max, y_max = bbox
    x_padding, y_padding = padding

    if x_padding > 0:
        x_span = x_max - x_min
        x_min -= x_padding * x_span
        x_max += x_padding * x_span

    if y_padding > 0:
        y_span = y_max - y_min
        y_min -= y_padding * y_span
        y_max += y_padding * y_span

    return x_min, y_min, x_max, y_max


def center_bbox_on_element(
    bbox: BoundingBox, center_bbox: BoundingBox = None, center_point=None
) -> BoundingBox:
    """Center the bounding box on a given element, ensuring the element is
    completely contained within the bounding box, and extending the bounding box
    as needed."""
    bb_x_min, bb_y_min, bb_x_max, bb_y_max = bbox

    if center_bbox is not None:
        el_x_min, el_y_min, el_x_max, el_y_max = center_bbox

    # If a center point is given, use that, otherwise use the center of the
    # bounding box
    if center_point is None:
        if center_bbox is None:
            raise ValueError(
                "Either `center_point` or `center_bbox` must be specified!"
            )
        x_center = el_x_min + (el_x_max - el_x_min) / 2
        y_center = el_y_min + (el_y_max - el_y_min) / 2
    else:
        x_center, y_center = center_point

    # Ensure the bounding box fits the entire center element
    if center_bbox is not None:
        bb_x_min = min(el_x_min, bb_x_min)
        bb_y_min = min(el_y_min, bb_y_min)
        bb_x_max = max(el_x_max, bb_x_max)
        bb_y_max = max(el_y_max, bb_y_max)

    # Determine how much space we need in both x/y directions from the center
    centered_x_dist = max(
        abs(x_center - bb_x_min), abs(x_center - bb_x_max)
    )
    centered_y_dist = max(
        abs(y_center - bb_y_min), abs(y_center - bb_y_max)
    )

    bb_x_min = x_center - centered_x_dist
    bb_x_max = x_center + centered_x_dist
    bb_y_min = y_center - centered_y_dist
    bb_y_max = y_center + centered_y_dist

    return bb_x_min, bb_y_min, bb_x_max, bb_y_max


def enforce_scatterplot_size(
    bbox: BoundingBox, scatter_bbox: BoundingBox, min_scatter_size: float
) -> BoundingBox:
    """For a given bounding box and a scatter plot bounding box, ensure that the
    scatter plot takes up at least `min_scatter_size` proportion of the bounding
    box."""
    x_min, y_min, x_max, y_max = bbox
    sc_x_min, sc_y_min, sc_x_max, sc_y_max = scatter_bbox

    sc_x_span = sc_x_max - sc_x_min
    sc_y_span = sc_y_max - sc_y_min
    max_x_axis_span = sc_x_span / min_scatter_size
    max_y_axis_span = sc_y_span / min_scatter_size

    x_span = x_max - x_min
    y_span = y_max - y_min

    if (x_axis_diff := max_x_axis_span - x_span) < 0:
        diff_each_side = abs(x_axis_diff) / 2
        x_max -= diff_each_side
        x_min += diff_each_side

    if (y_axis_diff := max_y_axis_span - y_span) < 0:
        diff_each_side = abs(y_axis_diff) / 2
        y_max -= diff_each_side
        y_min += diff_each_side

    return x_min, y_min, x_max, y_max


def fit_elements_onto_axis(
    ax: matplotlib.axes.Axes,
    label_objs: list[matplotlib.text.Text],
    max_iter: int = 10,
    padding: tuple[float, float] = (0, 0),
    eps: float = 1,
    scatter_obj: matplotlib.collections.LineCollection = None,
    center_on_scatter: bool = True,
    min_scatter_size: float = 0.25,
):
    """Ensure that the labels fit onto the plot canvas. Because the label
    fontsize is kept constant and rescaling the axes changes the size and
    positions of the labels, this has to be iterated until the label bounding
    boxes stop changing."""

    # Determine scatter plot bounds if available
    if scatter_obj is not None:
        scatter_positions = scatter_obj.get_offsets()
        sc_x_min, sc_y_min = np.min(scatter_positions, axis=0)
        sc_x_max, sc_y_max = np.max(scatter_positions, axis=0)

        if center_on_scatter:
            sc_x_center, sc_y_center = np.mean(scatter_positions, axis=0)

    coords = get_artist_bounding_boxes(label_objs, ax)

    for i in range(max_iter):
        bb_x_min, bb_y_min = np.min(coords[:, [0, 1]], axis=0)
        bb_x_max, bb_y_max = np.max(coords[:, [2, 3]], axis=0)

        # If we have a scatter plot element, ensure that the limits include that
        if scatter_obj is not None:
            bb_x_min = min(sc_x_min, bb_x_min)
            bb_y_min = min(sc_y_min, bb_y_min)
            bb_x_max = max(sc_x_max, bb_x_max)
            bb_y_max = max(sc_y_max, bb_y_max)

        # If the scatter plot is to be centered, extend bounds as needed
        if scatter_obj is not None and center_on_scatter:
            centered_x_dist = max(
                abs(sc_x_center - bb_x_min), abs(sc_x_center - bb_x_max)
            )
            centered_y_dist = max(
                abs(sc_y_center - bb_y_min), abs(sc_y_center - bb_y_max)
            )
            bb_x_min = sc_x_center - centered_x_dist
            bb_x_max = sc_x_center + centered_x_dist
            bb_y_min = sc_y_center - centered_y_dist
            bb_y_max = sc_y_center + centered_y_dist

        # Ensure the scatter element isn't too small
        if scatter_obj and min_scatter_size > 0:
            bb_x_min, bb_y_min, bb_x_max, bb_y_max = enforce_scatterplot_size(
                (bb_x_min, bb_y_min, bb_x_max, bb_y_max),
                (sc_x_min, sc_y_min, sc_x_max, sc_y_max),
                min_scatter_size=min_scatter_size,
            )

        # Apply padding
        bb_x_min, bb_y_min, bb_x_max, bb_y_max = add_bbox_padding(
            (bb_x_min, bb_y_min, bb_x_max, bb_y_max), padding=padding
        )

        # Force square aspect ratio
        bb_x_min, bb_y_min, bb_x_max, bb_y_max = set_bbox_square_aspect(
            (bb_x_min, bb_y_min, bb_x_max, bb_y_max)
        )

        ax.set_xlim(bb_x_min, bb_x_max)
        ax.set_ylim(bb_y_min, bb_y_max)

        # Get new coordinates after rescaling
        new_coords = get_artist_bounding_boxes(label_objs, ax)
        if np.allclose(coords, new_coords, atol=eps):
            logger.debug(f"Stopped after {i} iterations")
            break
        coords = new_coords


def bbox_to_polygon(bbox: BoundingBox) -> shapely.Polygon:
    x0, y0, x1, y1 = bbox
    return shapely.Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])


def convert_ax_to_data(ax, fraction: float, reduction="max") -> float:
    """Convert a number in the axis units to data units."""
    axis_to_data_transform = (ax.transAxes + ax.transData.inverted()).transform
    p0, p1 = axis_to_data_transform([[0, 0], [fraction, fraction]])
    diff = np.abs(p1 - p0)
    match reduction:
        case "max":
            return np.max(diff)
        case "min":
            return np.max(diff)
        case _:
            raise ValueError(f"Unrecognized reduction `{reduction}`")


def get_vector_between(
    geom1: shapely.Polygon,
    geom2: shapely.Polygon,
    between: str = "auto",
) -> tuple[np.ndarray, float]:
    """Given two polygons, determine their direction of attraction/repulsion.

    Parameters
    ----------
    geom1: shapely.Polygon
    geom2: shapely.Polygon
    between: str
        Specifies how the vector and distance between polygons is calculated.
        Can be onen of "boundaries", "centroids", and "auto".
        The behavior for "auto" is as follows:
        - For intersecting polygons, the direction is determined by the
          differences in their centroids. The distance for interseting polygons
          is always 0.
        - For disjoint polygons, the direction is determined along the line
          connecting the two closest points in both polygons. The distance is
          determined by the length of this line.

    Returns
    -------
    np.ndarray
        The unit-scaled vector corresponding to the direction of
        attraction/repulsion.
    float
        The distance between two polygons.

    """
    def _between_centroids(geom1, geom2):
        g1_centroid = np.array(geom1.centroid.xy).ravel()
        g2_centroid = np.array(geom2.centroid.xy).ravel()

        vector = g1_centroid - g2_centroid
        vector /= np.linalg.norm(vector) + 1e-8
        dist = shapely.distance(geom1, geom2)

        return vector, dist

    def _between_boundaries(geom1, geom2):
        # Get the coordinates of the two points that lie closest to one another
        p_g1, p_g2 = shapely.ops.nearest_points(geom1.boundary, geom2.boundary)
        p_g1, p_g2 = np.array(p_g1.coords[0]), np.array(p_g2.coords[0])

        # Compute the vector of attraction
        vector = p_g1 - p_g2
        dist = np.linalg.norm(vector)
        vector /= dist + 1e-8

        return vector, dist

    match between:
        case "boundaries":
            return _between_boundaries(geom1, geom2)
        case "centroids":
            return _between_centroids(geom1, geom2)
        case "auto":
            if geom1.intersects(geom2):
                return _between_centroids(geom1, geom2)
            else:
                return _between_boundaries(geom1, geom2)


def _optimize_label_positions_update_step(
    labels: list[shapely.Polygon],
    label_target_regions: list[shapely.Polygon],
    embedding_region: shapely.Polygon,
    ax: matplotlib.axes.Axes,
    label_region_margin: float,
    label_label_margin: float,
    bounds_margin: float,
    bounds_factor: float = 10,
    region_attraction_factor: float = 0.01,
    region_repulsion_factor: float = 1,
    label_repulsion_factor: float = 1,

):
    updates = np.zeros(shape=(len(labels), 2), dtype=float)

    # Ensure labels remain within the axes limits
    F_bounds = np.zeros_like(updates)
    ax_x_min, ax_x_max = ax.get_xlim()
    ax_y_min, ax_y_max = ax.get_xlim()

    # Add margin to bounds
    ax_x_min += bounds_margin
    ax_y_min += bounds_margin
    ax_x_max -= bounds_margin
    ax_y_max -= bounds_margin

    # Margin repulsion
    for i, label_i in enumerate(labels):
        bb_x_min, bb_y_min, bb_x_max, bb_y_max = label_i.bounds
        x_min_diff = min(0, bb_x_min - ax_x_min)
        x_max_diff = max(0, bb_x_max - ax_x_max)
        y_min_diff = min(0, bb_y_min - ax_y_min)
        y_max_diff = max(0, bb_y_max - ax_y_max)

        F_bounds[i] -= np.array([x_min_diff, y_min_diff])
        F_bounds[i] -= np.array([x_max_diff, y_max_diff])

    # Label-region attraction
    F_attr = np.zeros_like(updates)
    for i, (label_i, region_i) in enumerate(zip(labels, label_target_regions)):
        vec, dist = get_vector_between(label_i, region_i, between="boundaries")
        # Beyond the margin, we don't really care how far apart the labels are
        weight = max(0, dist - label_region_margin)
        F_attr[i] -= weight * vec

    # Label-label repulsion
    F_label_rep = np.zeros_like(updates)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            vec, dist = get_vector_between(labels[i], labels[j])
            # Beyond the margin, we don't really care how far apart the labels are
            weight = max(0, -dist + label_label_margin)
            F_label_rep[i] += weight * vec
            F_label_rep[j] -= weight * vec

    # Label-region repulsion
    F_region_rep = np.zeros_like(updates)
    for i, label_i in enumerate(labels):
        vec, dist = get_vector_between(label_i, embedding_region)
        # Beyond the margin, we don't really care how far apart the labels are
        weight = max(0, -dist + label_region_margin)
        F_region_rep[i] += weight * vec

    logger.debug(
        f"F_bounds norm: {np.linalg.norm(F_bounds):.2f} - "
        f"F_attr norm: {np.linalg.norm(F_attr):.2f} - "
        f"F_region_rep norm: {np.linalg.norm(F_region_rep):.2f} - "
        f"F_label_rep norm: {np.linalg.norm(F_label_rep):.2f} - "
    )

    return (
        bounds_factor * F_bounds +
        region_attraction_factor * F_attr +
        region_repulsion_factor * F_region_rep +
        label_repulsion_factor * F_label_rep
    )


def optimize_label_positions(
    labels: list[shapely.Polygon],
    label_target_regions: list[shapely.Polygon],
    embedding_region: shapely.Polygon,
    ax: matplotlib.axes.Axes,
    eps: float = 0.01,
    lr: float = 5,
    momentum: float = 0.8,
    max_iter: int = 50,
    max_step_norm: float = 5,
    label_region_margin: float = 0.03,
    label_label_margin: float = 0.02,
    bounds_margin: float = 0.03,
    return_history=False,
):
    assert len(labels) == len(label_target_regions), \
        "Each label needs an associated target region!"

    label_region_margin = convert_ax_to_data(ax, label_region_margin)
    label_label_margin = convert_ax_to_data(ax, label_label_margin)
    bounds_margin = convert_ax_to_data(ax, bounds_margin)

    if return_history:
        label_pos_history = [labels.copy()]

    # Gradually increase the bounding box repulsion factor
    bounds_factors = np.linspace(0.01, 10, num=max_iter, endpoint=True)

    updates = None
    for epoch in range(max_iter):
        step = _optimize_label_positions_update_step(
            labels,
            label_target_regions,
            embedding_region,
            ax,
            label_region_margin=label_region_margin,
            label_label_margin=label_label_margin,
            bounds_margin=bounds_margin,
            bounds_factor=bounds_factors[epoch],
            region_attraction_factor=0.01,
            region_repulsion_factor=1,
            label_repulsion_factor=1,
        )
        step_norms = np.linalg.norm(step, axis=1)
        if max_step_norm is not None:
            step_rescale = (
                np.minimum(step_norms, max_step_norm) / (step_norms + 1e-8)
            )
            step *= step_rescale[:, None]

        if updates is not None:
            updates *= momentum
            updates += step
        else:
            updates = step

        updates *= lr

        for i in range(len(labels)):
            labels[i] = shapely.affinity.translate(labels[i], *updates[i])

        if return_history:
            label_pos_history.append(labels.copy())

        logger.debug("update norm", np.linalg.norm(updates))
        # Check if stopping criteria met
        if np.max(np.linalg.norm(updates, axis=1)) < eps:
            logger.info("early stopping", epoch, step_norms)
            break

    if return_history:
        return labels, label_pos_history
    return labels


def get_label_bounding_boxes_on_ax(
    ax: matplotlib.axes.Axes,
    label_strs: list[str],
    label_positions: list[tuple[float, float]],
    label_kwargs: dict = None,
) -> list[shapely.Polygon]:
    """Determine label bounding boxes after rendered on the given axis.

    This function renders the label on the axis, checks its bounding box, and
    removes it from the axis.

    This is useful since labels can have very different sizes based on the axis
    size, dpi, and various font settings, e.g., font family, font size, etc.
    """
    if label_kwargs is None:
        label_kwargs = {}

    label_bboxes = []
    for label, label_pos in zip(label_strs, label_positions):
        # Render the label on the current axis, get its bounding box,
        # convert that to a polygon, then remove the rendered label
        handle = ax.text(*label_pos, label, **label_kwargs)
        label_bbox = get_artist_bounding_box(handle, ax)
        label_bboxes.append(bbox_to_polygon(label_bbox))
        handle.remove()

    return label_bboxes


def evaluate_label_pos_quality(
    labels: list[shapely.Polygon],
    label_target_regions: list[shapely.Polygon],
    all_regions: list[shapely.Polygon],
    ax: matplotlib.axes.Axes,
    label_region_margin: float = 0.03,
    label_label_margin: float = 0.02,
    bounds_margin: float = 0.03,
):
    ax_bbox = np.array(get_ax_bounding_box(ax))

    label_region_margin = convert_ax_to_data(ax, label_region_margin)
    label_label_margin = convert_ax_to_data(ax, label_label_margin)
    bounds_margin = convert_ax_to_data(ax, bounds_margin)

    # How many labels overflow the axes bounds
    label_bboxes = np.array([l.bounds for l in labels])
    mask = np.array([1, 1, -1, -1])
    hard_overflows = (ax_bbox * mask) > (label_bboxes * mask)
    n_hard_overflows = float(np.sum(hard_overflows))
    soft_overflows = (ax_bbox * mask + bounds_margin) > (label_bboxes * mask)
    n_soft_overflows = float(np.sum(soft_overflows))

    # How many labels intersect each other?
    hard_label_label_isects = {}
    soft_label_label_isects = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label_i = labels[i]
            label_j = labels[j]
            if shapely.intersects(label_i, label_j):
                hard_label_label_isects[i] = 1
                hard_label_label_isects[j] = 1
            if shapely.intersects(label_i.buffer(label_label_margin), label_j):
                soft_label_label_isects[i] = 1
                soft_label_label_isects[j] = 1

    # How many label-region intersections do we have?
    hard_label_region_isects = {}
    soft_label_region_isects = {}
    for i, label_i in enumerate(labels):
        for region_j in all_regions:
            if shapely.intersects(label_i, region_j):
                hard_label_region_isects[i] = 1.
            if shapely.intersects(label_i.buffer(label_region_margin), region_j):
                soft_label_region_isects[i] = 1.

    return {
        "hard_overflows": n_hard_overflows,
        "soft_overflows": n_soft_overflows,
        "hard_label_label_intersects": sum(hard_label_label_isects.values()),
        "soft_label_label_intersects": sum(soft_label_label_isects.values()),
        "hard_label_region_intersects": sum(hard_label_region_isects.values()),
        "soft_label_region_intersects": sum(soft_label_region_isects.values()),
    }
