import numpy as np
from sklearn.metrics import pairwise_distances

from vera.overlap_computations import (
    get_2d_coordinates,
    overlap_intervals,
    text_line_overlaps,
    intersect,
)

from matplotlib import pyplot as plt


def row_norm(an_array):
    return np.linalg.norm(an_array, axis=1)


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


def initial_text_location_placement(
    embedding, label_locations, base_radius=None, base_radius_factor=0.25
):
    # Find a center for label locations, ring radii, and how much to stretch theta; all heuristics
    mean_embedding_coord = np.mean(embedding, axis=0)
    centered_label_locations = label_locations - mean_embedding_coord

    if base_radius is None:
        centered_embedding = embedding - mean_embedding_coord
        base_radius = np.max(row_norm(centered_embedding)) + base_radius_factor * np.mean(
            row_norm(centered_embedding)
        )

    centered_label_locations = (
        centered_label_locations / row_norm(centered_label_locations)[:, None]
    )

    # Determine the angles of the label positions
    label_thetas = np.arctan2(
        centered_label_locations.T[0], centered_label_locations.T[1]
    )

    # Construct a ring of possible label placements around the embedding
    xs = np.linspace(0, 1, max(len(label_thetas) + 1, 8), endpoint=False)
    uniform_thetas = xs * 2 * np.pi

    # Find an optimal rotation of the ring to match the existing label locations
    optimal_rotation = 0.0
    min_score = np.inf
    for rotation in np.linspace(
        -np.pi / int(len(label_thetas) + 5), np.pi / int(len(label_thetas) + 5), 32
    ):
        test_label_locations = np.vstack(
            [
                base_radius * np.cos(uniform_thetas + rotation),
                base_radius * np.sin(uniform_thetas + rotation),
            ]
        ).T
        score = np.sum(
            pairwise_distances(
                centered_label_locations, test_label_locations, metric="cosine"
            ).min(axis=1)
        )
        if score < min_score:
            min_score = score
            optimal_rotation = rotation

    uniform_thetas += optimal_rotation

    # Convert the ring locations to cartesian coordinates
    uniform_label_locations = np.vstack(
        [base_radius * np.cos(uniform_thetas), base_radius * np.sin(uniform_thetas)]
    ).T

    # Sort labels by radius of the label location and pick the closest position in order;
    # This works surprisingly well
    order = np.argsort(-row_norm(label_locations - mean_embedding_coord))
    taken = set([])
    adjustment_dict_alt = {}
    for i in order:
        candidates = list(set(range(uniform_label_locations.shape[0])) - taken)
        candidate_distances = pairwise_distances(
            [centered_label_locations[i]],
            uniform_label_locations[candidates],
            metric="cosine",
        )
        selection = candidates[np.argmin(candidate_distances[0])]
        adjustment_dict_alt[i] = selection
        taken.add(selection)

    result = (
        np.asarray(
            [
                uniform_label_locations[adjustment_dict_alt[i]]
                for i in sorted(adjustment_dict_alt.keys())
            ]
        )
        + mean_embedding_coord
    )

    return result


def adjust_text_locations(
    text_locations,
    label_locations,
    label_text,
    font_size=12,
    fontfamily="DejaVu Sans",
    linespacing=0.95,
    expand=(1.5, 1.5),
    max_iter=100,
    label_size_adjustments=None,
    highlight=frozenset([]),
    highlight_label_keywords={},
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    # Add text to the axis and set up for optimization
    new_text_locations = text_locations.copy()
    texts = [
        ax.text(
            *new_text_locations[i],
            label_text[i],
            ha="center",
            ma="center",
            va="center",
            linespacing=linespacing,
            alpha=0.0,
            fontfamily=fontfamily,
            fontsize=(
                highlight_label_keywords.get("fontsize", font_size)
                if label_text[i] in highlight
                else font_size
            )
            + (
                label_size_adjustments[i] if label_size_adjustments is not None else 0.0
            ),
            fontweight="bold" if label_text[i] in highlight else "normal",
        )
        for i in range(label_locations.shape[0])
    ]

    coords = get_2d_coordinates(texts, expand=expand)
    xoverlaps = overlap_intervals(
        coords[:, 0], coords[:, 1], coords[:, 0], coords[:, 1]
    )
    xoverlaps = xoverlaps[xoverlaps[:, 0] != xoverlaps[:, 1]]
    yoverlaps = overlap_intervals(
        coords[:, 2], coords[:, 3], coords[:, 2], coords[:, 3]
    )
    yoverlaps = yoverlaps[yoverlaps[:, 0] != yoverlaps[:, 1]]
    overlaps = yoverlaps[(yoverlaps[:, None] == xoverlaps).all(-1).any(-1)]

    tight_coords = get_2d_coordinates(texts, expand=(0.9, 0.9))
    bottom_lefts = ax.transData.inverted().transform(tight_coords[:, [0, 2]])
    top_rights = ax.transData.inverted().transform(tight_coords[:, [1, 3]])
    coords_in_dataspace = np.vstack(
        [bottom_lefts.T[0], top_rights.T[0], bottom_lefts.T[1], top_rights.T[1]]
    ).T
    box_line_overlaps = text_line_overlaps(
        text_locations, label_locations, coords_in_dataspace
    )
    n_iter = 0

    # While we have overlaps, tweak the label positions
    while (len(overlaps) > 0 or len(box_line_overlaps) > 0) and n_iter < max_iter:
        # Check for text boxes overlapping each other
        coords = get_2d_coordinates(texts, expand=expand)
        xoverlaps = overlap_intervals(
            coords[:, 0], coords[:, 1], coords[:, 0], coords[:, 1]
        )
        xoverlaps = xoverlaps[xoverlaps[:, 0] != xoverlaps[:, 1]]
        yoverlaps = overlap_intervals(
            coords[:, 2], coords[:, 3], coords[:, 2], coords[:, 3]
        )
        yoverlaps = yoverlaps[yoverlaps[:, 0] != yoverlaps[:, 1]]
        overlaps = yoverlaps[(yoverlaps[:, None] == xoverlaps).all(-1).any(-1)]

        # Convert the text locations to polar coordinates, centered around the
        # mean position of all labels
        recentered_locations = new_text_locations - label_locations.mean(axis=0)
        radii = np.linalg.norm(recentered_locations, axis=1)
        thetas = np.arctan2(recentered_locations.T[1], recentered_locations.T[0])

        for left, right in overlaps:
            # adjust thetas
            direction = thetas[left] - thetas[right]
            if direction > np.pi or direction < -np.pi:
                thetas[left] -= 0.005 * np.sign(direction)
                thetas[right] += 0.005 * np.sign(direction)
            else:
                thetas[left] += 0.005 * np.sign(direction)
                thetas[right] -= 0.005 * np.sign(direction)

        # Check for indicator lines crossing text boxes
        recentered_locations = np.vstack(
            [radii * np.cos(thetas), radii * np.sin(thetas)]
        ).T
        new_text_locations = recentered_locations + label_locations.mean(axis=0)
        fix_crossings(new_text_locations, label_locations)
        for i, text in enumerate(texts):
            text.set_position(new_text_locations[i])

        tight_coords = get_2d_coordinates(texts, expand=expand)
        bottom_lefts = ax.transData.inverted().transform(tight_coords[:, [0, 2]])
        top_rights = ax.transData.inverted().transform(tight_coords[:, [1, 3]])
        coords_in_dataspace = np.vstack(
            [bottom_lefts.T[0], top_rights.T[0], bottom_lefts.T[1], top_rights.T[1]]
        ).T
        box_line_overlaps = text_line_overlaps(
            new_text_locations, label_locations, coords_in_dataspace
        )
        recentered_locations = new_text_locations - label_locations.mean(axis=0)
        radii = np.linalg.norm(recentered_locations, axis=1)
        thetas = np.arctan2(recentered_locations.T[1], recentered_locations.T[0])

        for i, j in box_line_overlaps:
            direction = np.arctan2(
                np.sum(coords_in_dataspace[i, 2:]) / 2.0 - label_locations[j, 1],
                np.sum(coords_in_dataspace[i, :2]) / 2.0 - label_locations[j, 0],
            ) - np.arctan2(
                text_locations[j, 1] - label_locations[j, 1],
                text_locations[j, 0] - label_locations[j, 0],
            )
            if direction > np.pi or direction < -np.pi:
                thetas[i] -= 0.005 * np.sign(direction)
                thetas[j] += 0.0025 * np.sign(direction)
            else:
                thetas[i] += 0.005 * np.sign(direction)
                thetas[j] -= 0.0025 * np.sign(direction)

        radii *= 1.003

        recentered_locations = np.vstack(
            [radii * np.cos(thetas), radii * np.sin(thetas)]
        ).T
        new_text_locations = recentered_locations + label_locations.mean(axis=0)
        fix_crossings(new_text_locations, label_locations)
        for i, text in enumerate(texts):
            text.set_position(new_text_locations[i])

        n_iter += 1

    return new_text_locations


# def estimate_font_size(
#     text_locations,
#     label_text,
#     initial_font_size,
#     fontfamily="DejaVu Sans",
#     linespacing=0.95,
#     expand=(1.5, 1.5),
#     ax=None,
# ):
#     if ax is None:
#         ax = plt.gca()
#
#     font_size = initial_font_size
#     overlap_percentage = 1.0
#     while overlap_percentage > 0.5 and font_size > 3.0:
#         texts = [
#             ax.text(
#                 *text_locations[i],
#                 label_text[i],
#                 ha="center",
#                 ma="center",
#                 va="center",
#                 linespacing=linespacing,
#                 alpha=0.0,
#                 fontfamily=fontfamily,
#                 fontsize=font_size,
#             )
#             for i in range(text_locations.shape[0])
#         ]
#         coords = get_2d_coordinates(texts, expand=expand)
#         xoverlaps = overlap_intervals(
#             coords[:, 0], coords[:, 1], coords[:, 0], coords[:, 1]
#         )
#         xoverlaps = xoverlaps[xoverlaps[:, 0] != xoverlaps[:, 1]]
#         yoverlaps = overlap_intervals(
#             coords[:, 2], coords[:, 3], coords[:, 2], coords[:, 3]
#         )
#         yoverlaps = yoverlaps[yoverlaps[:, 0] != yoverlaps[:, 1]]
#         overlaps = yoverlaps[(yoverlaps[:, None] == xoverlaps).all(-1).any(-1)]
#         overlap_percentage = len(overlaps) / (2 * text_locations.shape[0])
#         # remove texts
#         for t in texts:
#             t.remove()
#
#         font_size = 0.9 * font_size
#
#     return font_size
