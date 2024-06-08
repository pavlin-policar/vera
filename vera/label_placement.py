import io

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

try:
    from matplotlib.backend_bases import _get_renderer as matplot_get_renderer
except ImportError:
    matplot_get_renderer = None


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def fix_crossings(text_locations, label_locations, n_iter=3):
    # Find crossing lines and swap labels; repeat as required
    for n in range(n_iter):
        for i in range(text_locations.shape[0]):
            for j in range(text_locations.shape[0]):
                if intersect(
                    text_locations[i],
                    label_locations[i],
                    text_locations[j],
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


# From adjustText (https://github.com/Phlya/adjustText)
def get_bboxes(objs, r=None, expand=(1, 1), ax=None):
    ax = ax or plt.gca()
    r = r or get_renderer(ax.get_figure())
    return [i.get_window_extent(r).expanded(*expand) for i in objs]


# From adjustText (https://github.com/Phlya/adjustText)
def get_2d_coordinates(objs, expand=(1, 1)):
    try:
        ax = objs[0].axes
    except:
        ax = objs.axes
    bboxes = get_bboxes(objs, get_renderer(ax.get_figure()), expand, ax)
    xs = [
        (ax.convert_xunits(bbox.xmin), ax.convert_yunits(bbox.xmax)) for bbox in bboxes
    ]
    ys = [
        (ax.convert_xunits(bbox.ymin), ax.convert_yunits(bbox.ymax)) for bbox in bboxes
    ]
    coords = np.hstack([np.array(xs), np.array(ys)])
    return coords


# Adapted from datamapplot (https://github.com/TutteInstitute/datamapplot)
def initial_text_location_placement(
    embedding, label_locations, label_radius=None, radius_factor=0.25
):
    """Find an initial placement for labels.

    Parameters
    ----------
    embedding : np.ndarray
    label_locations : np.ndarray
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
    label_locations = label_locations - embedding_center_point

    if label_radius is None:
        centered_embedding = embedding - embedding_center_point
        dists_from_origin = np.linalg.norm(centered_embedding, axis=1)
        label_radius = np.max(dists_from_origin) * (1 + radius_factor)

    # Determine the angles of the label positions
    label_thetas = np.arctan2(label_locations.T[0], label_locations.T[1])

    # Construct a ring of possible label placements around the embedding, we
    # refer to these as spokes
    xs = np.linspace(0, 1, max(len(label_thetas) + 1, 8), endpoint=False)
    spoke_thetas = xs * 2 * np.pi

    # Rotate the spokes little by little and see how well it matches the label
    # locations, and select the rotation which achieves the best matching
    best_rotation = 0
    min_score = np.inf
    best_dists_to_labels = None
    for rotation in np.linspace(
        -np.pi / int(len(label_thetas) + 5), np.pi / int(len(label_thetas) + 5), 32
    ):
        # We can use unit vectors since we're calcualting cosine similarity
        test_spoke_label_locations = np.vstack([
            np.cos(spoke_thetas + rotation), np.sin(spoke_thetas + rotation),
        ]).T
        distances = pairwise_distances(
            label_locations, test_spoke_label_locations, metric="cosine"
        )
        # The score is the sum of the distances to the nearest spoke
        score = np.sum(np.min(distances, axis=1))
        if score < min_score:
            min_score = score
            best_rotation = rotation
            # Store the distances to each labels' nearest spoke for later
            best_dists_to_labels = np.min(distances, axis=1)

    # Convert the spoke locations to cartesian coordinates
    spoke_label_locations = np.vstack([
        label_radius * np.cos(spoke_thetas + best_rotation),
        label_radius * np.sin(spoke_thetas + best_rotation),
    ]).T

    # Sort the labels by distance to their best matching ring spoke
    label_order = np.argsort(best_dists_to_labels)
    taken = set()
    adjustment_dict_alt = {}
    for label_idx in label_order:
        # Compute the distance from the current label to the remaining available
        # spoke locations
        candidates = list(set(range(spoke_label_locations.shape[0])) - taken)
        candidate_distances = pairwise_distances(
            [label_locations[label_idx]],
            spoke_label_locations[candidates],
            metric="cosine",
        )
        selection = candidates[np.argmin(candidate_distances[0])]
        adjustment_dict_alt[label_idx] = selection
        taken.add(selection)

    label_locations = np.asarray([
        spoke_label_locations[adjustment_dict_alt[i]]
        for i in sorted(adjustment_dict_alt.keys())
    ])

    # Un-center label positions
    return label_locations + embedding_center_point
