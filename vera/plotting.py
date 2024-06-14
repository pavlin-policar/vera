import string
import warnings
from collections.abc import Iterable
from itertools import cycle, chain
from textwrap import wrap
from typing import Any, Union

import glasbey
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.text
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path

import vera.metrics as metrics
from vera.label_placement import initial_text_location_placement, get_2d_coordinates, fix_crossings
from vera.region import Density, Region
from vera.region_annotation import RegionAnnotation


def plot_feature(
    feature_names: Union[Any, list[Any]],
    df: pd.DataFrame,
    embedding: np.ndarray,
    binary=False,
    s=6,
    alpha=0.1,
    log=False,
    colors=None,
    threshold=0,
    zorder=1,
    title=None,
    ax: matplotlib.axes.Axes = None,
    agg="max",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    feature_names = np.atleast_1d(feature_names)
    feature_mask = df.columns.isin(feature_names)

    x = df.values[:, feature_mask]

    if colors is None:
        # colors = ["#fee8c8", "#e34a33"]
        # colors = ["#000000", "#7DF454"]
        colors = ["#000000", "#EA4736"]

    if binary:
        y = np.any(x > threshold, axis=1)
        ax.scatter(
            embedding[~y, 0],
            embedding[~y, 1],
            c=colors[0],
            s=s,
            alpha=alpha,
            rasterized=True,
            zorder=zorder,
        )
        ax.scatter(
            embedding[y, 0],
            embedding[y, 1],
            c=colors[1],
            s=s,
            alpha=alpha,
            rasterized=True,
            zorder=zorder,
        )
    else:
        if agg == "max":
            y = np.max(x, axis=1)
        elif agg == "sum":
            y = np.sum(x, axis=1)
        else:
            raise ValueError(f"Unrecognized aggregator `{agg}`")

        sort_idx = np.argsort(y)  # Trick to make higher values have larger zval

        if log:
            y = np.log1p(y)

        s = s * y

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "expression", [colors[0], colors[1]], N=256
        )
        ax.scatter(
            embedding[sort_idx, 0],
            embedding[sort_idx, 1],
            c=y[sort_idx],
            s=s[sort_idx],
            alpha=alpha,
            rasterized=True,
            cmap=cmap,
            zorder=zorder,
        )

    # Hide ticks and axis
    ax.set(xticks=[], yticks=[])
    ax.axis("equal")
    ax.set_box_aspect(1)

    marker_str = ", ".join(map(str, feature_names))
    if title is None:
        ax.set_title("\n".join(wrap(marker_str, 40)))
    else:
        ax.set_title(title)

    return ax


def plot_features(
    features: Union[list[Any], dict[str, list[Any]]],
    data: pd.DataFrame,
    embedding: np.ndarray,
    per_row=4,
    figwidth=24,
    binary=False,
    s=6,
    alpha=0.1,
    log=False,
    colors=None,
    threshold=0,
    return_ax=False,
    zorder=1,
    agg="max",
):
    n_rows = len(features) // per_row
    if len(features) % per_row > 0:
        n_rows += 1

    figheight = figwidth / per_row * n_rows
    fig, ax = plt.subplots(nrows=n_rows, ncols=per_row, figsize=(figwidth, figheight))

    ax = ax.ravel()

    if isinstance(features, dict):
        features_ = features.values()
    elif isinstance(features, list):
        features_ = features
    else:
        raise ValueError("features cannot be instance of `%s`" % type(features))

    # Handle lists of markers
    all_features = []
    for m in features_:
        if isinstance(m, list):
            for m_ in m:
                all_features.append(m_)
        else:
            all_features.append(m)
    assert all(
        f in data.columns for f in all_features
    ), "One or more of the specified features was not found in dataset"

    if colors is None:
        # colors = ["#fee8c8", "#e34a33"]
        # colors = ["#000000", "#7DF454"]
        colors = ["#000000", "#EA4736"]

    for idx, marker in enumerate(features_):
        plot_feature(
            marker,
            data,
            embedding,
            binary=binary,
            s=s,
            alpha=alpha,
            log=log,
            colors=colors,
            threshold=threshold,
            zorder=zorder,
            ax=ax[idx],
            agg=agg,
        )

        if isinstance(features, dict):
            title = ax.get_title()
            title = f"{list(features)[idx]}\n{title}"
            ax[idx].set_title(title)

        plt.tight_layout()

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if return_ax:
        return fig, ax


def get_cmap_colors(cmap: str):
    import matplotlib.cm

    return matplotlib.cm.get_cmap(cmap).colors


def get_cmap_hues(cmap: str):
    """Extract the hue values from a given colormap."""
    colors = get_cmap_colors(cmap)
    hues = [c[0] for c in colors.rgb_to_hsv(colors)]

    return np.array(hues)


def hue_colormap(
    hue: float, levels: Union[Iterable, int] = 10, min_saturation: float = 0
) -> mcolors.ListedColormap:
    """Create an HSV colormap with varying saturation levels"""
    if isinstance(levels, Iterable):
        hsv = [[hue, (s + min_saturation) / (1 + min_saturation), 1] for s in levels]
    else:
        num_levels = len(levels) if isinstance(levels, Iterable) else levels
        hsv = [[hue, s, 1] for s in np.linspace(min_saturation, 1, num=num_levels)]

    rgb = mcolors.hsv_to_rgb(hsv)
    cmap = mcolors.ListedColormap(rgb)

    return cmap


def enumerate_plots(ax: list[matplotlib.axes.Axes], offset=0.025, text_params={}):
    text_params_ = dict(
        va="top",
        ha="left",
        fontweight="bold",
        fontfamily="Arial Black",
        fontsize=10,
    )
    text_params_.update(text_params)
    for idx, ax_ in enumerate(ax):
        letter = string.ascii_lowercase[idx % len(string.ascii_lowercase)]
        ax[idx].text(
            offset,
            1 - offset,
            letter,
            transform=ax[idx].transAxes,
            **text_params_,
        )


def plot_density(
    density: Union[RegionAnnotation, Region, Density],
    embedding: np.ndarray = None,
    levels: Union[int, np.ndarray] = 5,
    skip_first: bool = True,
    ax: matplotlib.axes.Axes = None,
    cmap="RdBu_r",
    contour_kwargs: dict = {},
    contourf_kwargs: dict = {},
    scatter_kwargs: dict = {},
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    tck = None
    if isinstance(levels, Iterable):
        if skip_first:
            levels = levels[1:]
    else:
        if skip_first:
            tck = ticker.MaxNLocator(nbins=levels, prune="lower")

    contour_kwargs_ = {"zorder": 1, "linewidths": 1, "colors": "k", **contour_kwargs}
    contourf_kwargs_ = {"zorder": 1, "alpha": 0.5, **contourf_kwargs}

    x, y, z = density._get_xyz(scaled=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        ax.contourf(x, y, z, levels=levels, cmap=cmap, locator=tck, **contourf_kwargs_)
        ax.contour(x, y, z, levels=levels, locator=tck, **contour_kwargs_)

    if embedding is not None:
        scatter_kwargs_ = {
            "zorder": 1,
            "c": "k",
            "s": 6,
            "alpha": 0.1,
            "lw": 0,
        }
        scatter_kwargs_.update(scatter_kwargs)
        ax.scatter(embedding[:, 0], embedding[:, 1], **scatter_kwargs_, rasterized=True)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.set(box_aspect=1, aspect=1)

    return ax


def plot_densities(
    variables: list[RegionAnnotation],
    levels: Union[int, np.ndarray] = 5,
    skip_first: bool = True,
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
    contour_kwargs: dict = {},
    contourf_kwargs: dict = {},
    scatter_kwargs: dict = {},
):
    n_rows = len(variables) // per_row
    if len(variables) % per_row > 0:
        n_rows += 1

    figheight = figwidth / per_row * n_rows
    fig, ax = plt.subplots(nrows=n_rows, ncols=per_row, figsize=(figwidth, figheight))

    if len(variables) == 1:
        ax = np.array([ax])
    ax = ax.ravel()

    for idx, variable in enumerate(variables):
        ax[idx].set_title(variable.name)

        plot_density(
            variable.region.density,
            embedding=variable.region.embedding.X,
            levels=levels,
            skip_first=skip_first,
            ax=ax[idx],
            contour_kwargs=contour_kwargs,
            contourf_kwargs=contourf_kwargs,
            scatter_kwargs=scatter_kwargs,
        )

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if return_ax:
        return fig, ax


def _format_explanatory_variable(variable: RegionAnnotation, max_width=40):
    return "\n".join(wrap(str(variable.descriptor), width=max_width))


# def _format_explanatory_variable_group(var_group: RegionAnnotationGroup, max_width=40):
#     var_strings = [str(v) for v in var_group.descriptor.variables]
#     if max_width is not None:
#         var_strings = [wrap(s, width=max_width) for s in var_strings]
#     else:
#         # Ensure consistent format with wrapped version
#         var_strings = [[vs] for vs in var_strings]
#
#     # Flatten string parts
#     lines = reduce(operator.add, var_strings)
#
#     return "\n".join(lines)


def _plot_region(
    region_annotation: RegionAnnotation,
    ax: matplotlib.axes.Axes,
    fill_color: str,
    edge_color: str,
    fill_alpha: float,
    edge_alpha: float,
    lw: float,
    indicate_purity: bool = False
):
    # If no edge color is specified, use the same color as the fill
    if edge_color is None:
        edge_color = fill_color

    # The purity effect should never go below the following threshold
    if indicate_purity:
        purity_effect_size = 0.75
        purity_factor = metrics.purity(region_annotation) * purity_effect_size + (1 - purity_effect_size)
        edge_alpha *= purity_factor
        fill_alpha *= purity_factor

    fill_patches, edge_patches = [], []
    for geom in region_annotation.region.polygon.geoms:
        # Polygon plotting code taken from
        # https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib
        path = Path.make_compound_path(
            Path(np.asarray(geom.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in geom.interiors],
        )

        # Edge
        edge_patch = PathPatch(
            path,
            fill=False,
            edgecolor=edge_color,
            alpha=edge_alpha,
            lw=lw,
            zorder=10,
        )
        ax.add_patch(edge_patch)
        edge_patches.append(edge_patch)
        # Fill
        if fill_color is not None:
            fill_patch = PathPatch(
                path,
                fill=True,
                color=fill_color,
                alpha=fill_alpha,
                zorder=1,
            )
            ax.add_patch(fill_patch)
            fill_patches.append(fill_patch)

    return fill_patches, edge_patches


def plot_region(
    region_annotation: RegionAnnotation,
    ax=None,
    fill_color="tab:blue",
    edge_color=None,
    fill_alpha=0.25,
    edge_alpha=1,
    lw=1,
    draw_label=False,
    draw_scatterplot=True,
    highlight_members=True,
    member_color="tab:red",
    indicate_purity: bool = False,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    show: bool = False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    _plot_region(
        region_annotation,
        ax,
        fill_color=fill_color,
        edge_color=edge_color,
        fill_alpha=fill_alpha,
        edge_alpha=edge_alpha,
        lw=lw,
        indicate_purity=indicate_purity,
    )

    if draw_label:
        # Obtain the label string to draw over the region
        if isinstance(region_annotation, RegionAnnotation):
            label_str = _format_explanatory_variable(region_annotation)
        else:
            label_str = str(region_annotation)

        # Draw the lable on the largest polygon in the region
        largest_polygon = max(region_annotation.region.polygon.geoms, key=lambda x: x.area)
        label_kwargs_ = {
            "ha": "center",
            "va": "center",
            "ma": "center",
            "fontsize": 9,
            "zorder": 99,
            "color": fill_color,
        }
        label_kwargs_.update(label_kwargs)
        x, y = largest_polygon.centroid.coords[0]

        label = ax.text(x, y, label_str, **label_kwargs_)
        # label.set_bbox(dict(facecolor="white", alpha=0.75, edgecolor="white"))

    # Plot embedding scatter plot
    if draw_scatterplot:
        embedding = region_annotation.region.embedding.X
        scatter_kwargs_ = {
            "zorder": 1,
            "c": "#999999",
            "s": 6,
            "alpha": 1,
            "lw": 0,
        }
        scatter_kwargs_.update(scatter_kwargs)
        if highlight_members:
            other_color = scatter_kwargs_["c"]
            c = np.array([other_color, member_color])[region_annotation.descriptor.values.astype(int)]
            scatter_kwargs_["c"] = c
            scatter_kwargs_["alpha"] = 1

        ax.scatter(embedding[:, 0], embedding[:, 1], **scatter_kwargs_, rasterized=True)

    # Set title
    ax.set_title(region_annotation.descriptor.name)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.axis("equal")

    if show:
        ax.get_figure().show()

    return ax


def plot_regions(
    region_annotations: list[RegionAnnotation],
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
    fill_color="tab:blue",
    edge_color=None,
    fill_alpha=0.25,
    edge_alpha=1,
    lw=1,
    draw_labels=False,
    highlight_members=True,
    member_color="tab:red",
    indicate_purity: bool = False,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    show: bool = False,
):
    n_rows = len(region_annotations) // per_row
    if len(region_annotations) % per_row > 0:
        n_rows += 1

    figheight = figwidth / per_row * n_rows
    fig, ax = plt.subplots(nrows=n_rows, ncols=per_row, figsize=(figwidth, figheight))

    if len(region_annotations) == 1:
        ax = np.array([ax])
    ax = ax.ravel()

    for idx, variable in enumerate(region_annotations):
        plot_region(
            variable,
            ax=ax[idx],
            fill_color=fill_color,
            edge_color=edge_color,
            fill_alpha=fill_alpha,
            edge_alpha=edge_alpha,
            lw=lw,
            draw_label=draw_labels,
            highlight_members=highlight_members,
            member_color=member_color,
            indicate_purity=indicate_purity,
            scatter_kwargs=scatter_kwargs,
            label_kwargs=label_kwargs,
        )

    enumerate_plots(ax[:idx + 1])

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if show:
        fig.show()

    if return_ax:
        return fig, ax


def plot_region_with_subregions(
    region_annotation: RegionAnnotation,
    ax: matplotlib.axes.Axes = None,
    cmap: str = "tab10",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    plot_region(
        region_annotation,
        ax=ax,
        highlight_members=False,
        fill_color="#cccccc",
        edge_color="#666666",
    )

    hues = iter(cycle(get_cmap_colors(cmap)))
    for sub_ra, c in zip(region_annotation.contained_region_annotations, hues):
        plot_region(
            sub_ra,
            ax=ax,
            highlight_members=False,
            fill_color=c,
            draw_scatterplot=False,
        )

    # Set title
    ax.set_title(region_annotation.name)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.axis("equal")

    return ax


def plot_regions_with_subregions(
    region_annotations: list[RegionAnnotation],
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
    cmap: str = "tab10",
    show: bool = False,
):
    n_rows = len(region_annotations) // per_row
    if len(region_annotations) % per_row > 0:
        n_rows += 1

    figheight = figwidth / per_row * n_rows
    fig, ax = plt.subplots(nrows=n_rows, ncols=per_row, figsize=(figwidth, figheight))

    if len(region_annotations) == 1:
        ax = np.array([ax])
    ax = ax.ravel()

    for idx, ra in enumerate(region_annotations):
        plot_region_with_subregions(ra, ax=ax[idx], cmap=cmap)

    enumerate_plots(ax[:idx + 1])

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if show:
        fig.show()

    if return_ax:
        return fig, ax


def plot_annotation(
    region_annotations: list[RegionAnnotation],
    cmap: str = "tab10",
    ax: matplotlib.axes.Axes = None,
    indicate_purity: bool = False,
    indicate_membership: bool = False,
    only_color_inside_members: bool = True,
    draw_labels=True,
    ra_colors: dict = None,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    figwidth: int = 4,
    return_ax: bool = False,
    show: bool = False,
    min_embedding_size: float = 0.25,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(figwidth, figwidth), dpi=150)

    if ra_colors is None:
        # Glasbey crashes when requesting fewer colors than are in the cmap
        cmap_len = len(get_cmap_colors(cmap))
        request_len = max(cmap_len, len(region_annotations))
        cmap = glasbey.extend_palette(cmap, palette_size=request_len)
        ra_colors = {
            ra: mcolors.to_rgb(c) for ra, c in zip(region_annotations, cmap)
        }

    # Save region patches to be used for label placement. We don't need both 
    # fill and edge patches, so use either of the two. Here, we use fill patches
    region_patches = []
    for region_annotation in region_annotations:
        fill_patches, edge_patches = _plot_region(
            region_annotation,
            ax=ax,
            fill_color=ra_colors[region_annotation],
            edge_color=ra_colors[region_annotation],
            fill_alpha=0.25,
            edge_alpha=1,
            lw=1,
            indicate_purity=indicate_purity,
        )
        # region_patches.extend(fill_patches)
        region_patches.extend(region_annotation.region.polygon.geoms)

    embedding = region_annotations[0].region.embedding.X
    scatter_kwargs_ = {
        "zorder": 2,
        "s": 6,
        "alpha": 1,
        "lw": 0,
        **scatter_kwargs,
    }

    # Setup sample colors
    point_colors = np.array([mcolors.to_rgb("#aaaaaa")] * embedding.shape[0])

    if indicate_membership:
        # Set sample colors inside regions
        for region_annotation in region_annotations:
            if only_color_inside_members:
                group_indices = list(region_annotation.contained_members)
            else:
                group_indices = list(region_annotation.all_members)
            point_colors[group_indices] = ra_colors[region_annotation]

        # Desaturate colors slightly
        point_colors = mcolors.rgb_to_hsv(point_colors)
        point_colors[:, 1] *= 0.75
        point_colors = mcolors.hsv_to_rgb(point_colors)

    scatter_obj = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=point_colors,
        **scatter_kwargs_
    )

    if draw_labels:
        label_data = []
        for region_annotation in region_annotations:
            # Obtain the label string to draw over the region
            if isinstance(region_annotation, RegionAnnotation):
                label_str = _format_explanatory_variable(region_annotation)
            else:
                label_str = str(region_annotation)

            # Draw the lable on the largest polygon in the region
            largest_polygon = max(region_annotation.region.polygon.geoms, key=lambda x: x.area)
            x, y = largest_polygon.centroid.coords[0]

            label_data.append({"text": label_str, "pos": [x, y], "color": ra_colors[region_annotation]})

        label_positions = [lbl["pos"] for lbl in label_data]
        label_text_positions = initial_text_location_placement(
            embedding, label_positions, radius_factor=0.5
        )
        # label_text_positions[:] = 0
        fix_crossings(label_text_positions, label_positions)

        label_kwargs_ = dict(
            ha="center",
            ma="center",
            va="center",
            fontsize=7,
            # fontstretch="condensed",
            # fontweight="light",
            fontfamily="Helvetica Neue",
            zorder=99,
        )
        label_kwargs_.update(label_kwargs)

        label_objs = []
        for lbl_data, label_text_pos in zip(label_data, label_text_positions):
            # label_objs.append(ax.annotate(
            #     lbl_data["text"],
            #     lbl_data["pos"],
            #     xytext=label_text_pos,
            #     color=lbl_data["color"],
            #     **label_kwargs_,
            #     arrowprops={
            #         "arrowstyle": "-",
            #         "linewidth": 1,
            #         "color": lbl_data["color"],
            #     },
            # ))
            label_objs.append(
                ax.text(
                    *label_text_pos,
                    lbl_data["text"],
                    color=lbl_data["color"],
                    **label_kwargs_,
                )
            )
    
    # Calculate how small we want the final embedding to actually be, or, 
    # equivalently, what are the maximum limits we want to allow
    min_coords = np.min(embedding, axis=0)
    max_coords = np.max(embedding, axis=0)
    embedding_scale = np.max(max_coords - min_coords)
    max_axis_limits = embedding_scale / min_embedding_size

    # Rescale the axes so the text doesn't overflow the canvas
    rescale_axes(label_objs, ax, padding=0, max_axis_limits=max_axis_limits, scatter_obj=scatter_obj)

    # Where are the labels supposed to point to?
    label_targets = np.array([data["pos"] for data in label_data])

    # Optimize label positions so that there is minimal overlap and the labels 
    # are as compact as possible
    optimize_label_positions(label_objs, label_targets, region_patches, ax, scatter_obj=scatter_obj, max_axis_limits=max_axis_limits)

    import matplotlib.patches as mpatches
    coords = get_2d_coordinates(label_objs, expand=(1, 1))
    for x0, x1, y0, y1 in coords:
        x0y0t = ax.transData.inverted().transform([x0, y0])
        x1y1t = ax.transData.inverted().transform([x1, y1])
        wt, ht = x1y1t - x0y0t
        ax.add_patch(mpatches.Rectangle(x0y0t, wt, ht, fill=False))

    ax.set(xticks=[], yticks=[])
    # ax.set(box_aspect=1, aspect=1)

    if show:
        fig.show()

    if return_ax:
        return label_objs, fig, ax


def rescale_axes(
    label_objs: list[matplotlib.text.Text],
    ax: matplotlib.axes.Axes,
    max_iter: int = 25,
    padding: float = 0,
    eps: float = 0.1,
    scatter_obj: matplotlib.collections.LineCollection = None,
    max_axis_limits: float = 0.25,
):
    """Ensure that the labels fit onto the plot canvas. Because the label 
    fontsize is kept consistent, and rescaling the axes changes the size and 
    positions of the labels, this is run multiple times until a maximum number
    of iterations has been reached or until the label bounding boxes stop
    changing."""

    # TODO: Move this out of the function
    import logging
    logger = logging.getLogger("VERA")

    # Determine scatter plot bounds if available
    if scatter_obj is not None:
        scatter_positions = scatter_obj.get_offsets()
        sc_x_min, sc_y_min = np.min(scatter_positions, axis=0)
        sc_x_max, sc_y_max = np.max(scatter_positions, axis=0)

    coords = get_2d_coordinates(label_objs)
    for i in range(max_iter):
        # Get label bounding box limits
        bb_x_min, bb_y_min = ax.transData.inverted().transform(
            (coords[:, [0, 2]].copy().min(axis=0))
        )
        bb_x_max, bb_y_max = ax.transData.inverted().transform(
            (coords[:, [1, 3]].copy().max(axis=0))
        )

        # Apply padding to label bounding boxes if necessary
        if padding > 0:
            width = bb_x_max - bb_x_min
            height = bb_y_max - bb_y_min
            bb_x_min -= padding * width
            bb_x_max += padding * width
            bb_y_min -= padding * height
            bb_y_max += padding * height

        if scatter_obj is not None:
            x_new_min = min(bb_x_min, sc_x_min)
            y_new_min = min(bb_y_min, sc_y_min)
            x_new_max = max(bb_x_max, sc_x_max)
            y_new_max = max(bb_y_max, sc_y_max)
        else:
            x_new_min = bb_x_min
            y_new_min = bb_y_min
            x_new_max = bb_x_max
            y_new_max = bb_y_max

        # Force aspect ratio to 1
        # Determine which of the spans is larger
        x_span = x_new_max - x_new_min
        y_span = y_new_max - y_new_min
        long_span = max(x_span, y_span)
        # How much do we need to add to the shorter span to match the longer one
        x_span_diff = long_span - x_span
        y_span_diff = long_span - y_span

        ax.set_xlim(x_new_min - x_span_diff / 2, x_new_max + x_span_diff / 2)
        ax.set_ylim(y_new_min - y_span_diff / 2, y_new_max + y_span_diff / 2)

        # Get new coordinates after rescaling
        new_coords = get_2d_coordinates(label_objs)
        if np.allclose(coords, new_coords, atol=eps):
            logger.debug(f"Stopped after {i} iterations")
            break
        coords = new_coords


def optimize_label_positions(
    label_objs: list[matplotlib.text.Text],
    label_targets,
    region_objs: list[matplotlib.patches.PathPatch],
    ax: matplotlib.axes.Axes,
    scatter_obj: matplotlib.collections.LineCollection,
    eps: float = 0.1,
    max_axis_limits: float = None,
):
    import shapely

    # TODO: Move this out of the function
    import logging
    logger = logging.getLogger("VERA")

    for epoch in range(500):
        updates = np.zeros(shape=(len(label_objs), 2), dtype=float)

        # Get the bounding boxes of each label object as shapely objects
        label_coords = get_2d_coordinates(label_objs, expand=(1, 1))
        bounding_boxes = []
        bounding_box_centroids = []
        for x0, x1, y0, y1 in label_coords:
            x0, y0 = ax.transData.inverted().transform([x0, y0])
            x1, y1 = ax.transData.inverted().transform([x1, y1])
            bounding_box = shapely.Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
            bounding_boxes.append(bounding_box)
            bounding_box_centroids.append([
                bounding_box.centroid.xy[0][0],
                bounding_box.centroid.xy[1][0],
            ])
        bounding_box_centroids = np.array(bounding_box_centroids)

        # Compute gravity: center of mass to origin
        # F_gravity = -bounding_box_centroids
        F_gravity = np.zeros_like(updates)
        # Compute gravity: because labels are horizontally long, instead of 
        # using the center of mass, we will apply gravity to all four corner 
        # points of the bounding box. This should *hopefully* encourage wide 
        # bounding boxes to be more centrally positioned. Additionally, we will
        # apply stronger gravity to the x-coordinate than to the y-coordinate
        xy_gravity_weights = np.array([4, 1])
        for i, bb_i in enumerate(bounding_boxes):
            x_min, y_min, x_max, y_max = bb_i.bounds
            boundary_points = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ])
            # dists = np.linalg.norm(boundary_points, axis=1)
            # i_max = np.argmax(dists)
            F_gravity[i] -= xy_gravity_weights * np.sum(boundary_points, axis=0)

        # Label-target attraction
        F_attr = label_targets - bounding_box_centroids

        # Label-label repulsion
        F_label_rep = np.zeros_like(updates)
        for i in range(len(label_objs)):
            for j in range(i + 1, len(label_objs)):
                # Compute direction of repulsion
                repulsion_vector = bounding_box_centroids[i] - bounding_box_centroids[j]
                repulsion_vector /= np.linalg.norm(repulsion_vector)

                # Compute the distance between the nearest points on both 
                # bounding boxes
                dist = bounding_boxes[i].distance(bounding_boxes[j])
                # After a certain point, we don't really care how far apart the 
                # labels are
                margin = 5
                weight = 1 - min(margin, dist) / margin
                #weight = 1 / max(dist, 1)
                F_label_rep[i] += weight * repulsion_vector
                F_label_rep[j] -= weight * repulsion_vector

        # Label-region repulsion
        F_region_rep = np.zeros_like(updates)
        # path = region_objs[0].get_path()
        # print(shapely.Polygon(path.vertices))
        for i, bb_i in enumerate(bounding_boxes):
            for j, region_j in enumerate(region_objs):
                region_centroid = np.array(region_j.centroid.xy).ravel()

                # Compute direction of repulsion
                repulsion_vector = bounding_box_centroids[i] - region_centroid
                repulsion_vector /= np.linalg.norm(repulsion_vector)

                # Compute the distance between the nearest points on both 
                # bounding boxes
                dist = bb_i.distance(region_j)
                # After a certain point, we don't really care how far apart the 
                # labels are
                margin = 5
                weight = 1 - min(margin, dist) / margin
                #weight = 1 / max(dist, 1)
                F_region_rep[i] += weight * repulsion_vector

        updates = 0.001 * F_gravity + 0.001 * F_attr + 1 * F_region_rep + 1 * F_label_rep

        lr = 10
        for label_obj, label_update in zip(label_objs, updates):
            text_pos = np.array(label_obj.get_position())
            label_obj.set_position(text_pos + lr * label_update)

        rescale_axes(label_objs, ax, max_axis_limits=max_axis_limits, scatter_obj=scatter_obj)

        logger.debug("update norm", np.linalg.norm(updates))
        # Check if stopping criteria met
        if np.linalg.norm(updates) < eps:
            break


def plot_annotations(
    layouts: list[list[RegionAnnotation]],
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
    cmap: str = "tab10",
    indicate_purity: bool = False,
    indicate_membership: bool = True,
    only_color_inside_members: bool = True,
    variable_colors: dict = None,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    show: bool = False,
):
    n_rows = len(layouts) // per_row
    if len(layouts) % per_row > 0:
        n_rows += 1

    figheight = figwidth / per_row * n_rows
    fig, ax = plt.subplots(nrows=n_rows, ncols=per_row, figsize=(figwidth, figheight))

    if len(layouts) == 1:
        ax = np.array([ax])
    ax = ax.ravel()

    for idx, variables in enumerate(layouts):
        plot_annotation(
            variables,
            cmap=cmap,
            ax=ax[idx],
            indicate_purity=indicate_purity,
            indicate_membership=indicate_membership,
            only_color_inside_members=only_color_inside_members,
            ra_colors=variable_colors,
            scatter_kwargs=scatter_kwargs,
            label_kwargs=label_kwargs,
        )

    enumerate_plots(ax[:idx + 1])

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if show:
        fig.show()

    if return_ax:
        return fig, ax


def layout_variable_colors(
    layout: list[list[RegionAnnotation]],
    cmap="tab10",
) -> dict[RegionAnnotation, str]:
    all_region_annotations = set(chain.from_iterable(layout))

    # We use the region descriptors rule as color key
    region_annotation_keys = {ra: ra.descriptor for ra in all_region_annotations}

    # Glasbey crashes when requesting fewer colors than the cmap contains
    num_cmap_colors = len(get_cmap_colors(cmap))
    num_colors_to_request = max(num_cmap_colors, len(region_annotation_keys))
    cmap = glasbey.extend_palette(
        cmap, palette_size=num_colors_to_request, colorblind_safe=True
    )

    descriptor_color_mapping = {
        descriptor: mcolors.to_rgb(c)
        for descriptor, c in zip(region_annotation_keys.values(), cmap)
    }
    region_annotation_colors = {
        ra: descriptor_color_mapping[region_annotation_keys[ra]]
        for ra in all_region_annotations
    }

    return region_annotation_colors


def plot_discretization(
    region_annotations: list[RegionAnnotation],
    cmap: str = "viridis",
    hist_scatter_kwargs: dict = {},
    scatter_kwargs: dict = {},
    return_fig: bool = False,
    fig: matplotlib.figure.Figure = None,
):
    import matplotlib.gridspec as gridspec

    def _get_bin_edges_continuous(explanatory_variables: list[RegionAnnotation]):
        edges = [v.rule.lower for v in explanatory_variables]
        edges += [explanatory_variables[-1].rule.upper]
        if np.isinf(edges[0]):
            edges[0] = region_annotations.values.min()
        if np.isinf(edges[-1]):
            edges[-1] = region_annotations.values.max()

        return edges

    def _get_bin_edges_discrete(explanatory_variables: list[RegionAnnotation]):
        num_contained_variables = [len(v.contained_region_annotations) for v in explanatory_variables]
        edges = np.concatenate([[0], np.cumsum(num_contained_variables)]) + 0.5

        return edges

    def _get_sample_bin_indices(explanatory_variables: list[RegionAnnotation]):
        variable_values = np.vstack([v.values for v in explanatory_variables])
        variable_group_values = np.argmax(variable_values, axis=0)
        return variable_group_values

    unmerged_region_annotations = [
        base_ra
        for ra in region_annotations
        for base_ra in ra.contained_region_annotations
    ]

    if region_annotations.is_continuous:
        _get_bin_edges_func = _get_bin_edges_continuous
    elif region_annotations.is_discrete:
        _get_bin_edges_func = _get_bin_edges_discrete

    unmerged_feature_pt_bins = _get_sample_bin_indices(unmerged_region_annotations)
    unmerged_feature_bin_edges = _get_bin_edges_func(unmerged_region_annotations)
    merged_feature_pt_bins = _get_sample_bin_indices(region_annotations)
    merged_feature_bin_edges = _get_bin_edges_func(region_annotations)

    def plot_distribution_bins(x, bin_edges, x_bins, bins, ax, cmap=None, hist_scatter_kwargs={}):
        d, bins, *_ = ax.hist(
            x, bins=bins, alpha=0.5, edgecolor="k", align="mid", zorder=5
        )

        bin_width = bins[1] - bins[0]
        x_jitter = x + np.random.normal(0, bin_width * 0.05, size=x.shape)
        y_jitter = np.abs(np.random.normal(0, d.max() / 3, size=x.shape))

        hist_scatter_kwargs_ = {}
        hist_scatter_kwargs_.update(hist_scatter_kwargs)
        ax.scatter(x_jitter, y_jitter, c=x_bins, cmap=cmap, **hist_scatter_kwargs, rasterized=True)

        for edge in bin_edges[1:-1]:
            ax.axvline(edge, c="tab:red", lw=1)

        # ax.set_xticks(bin_edges[:-1] + 0.5)
        # ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Attribute Values", labelpad=3)
        ax.set_ylabel("Frequency", labelpad=3)
        ax.spines[["right", "top"]].set_visible(False)

        return ax

    v = region_annotations  # The variable in question
    embedding = v.region_annotations[0].region.embedding.X

    variable_values = v.values
    if pd.api.types.is_categorical_dtype(variable_values):
        variable_values = variable_values.codes.astype(float) + 1

    if fig is None:
        fig = plt.figure(figsize=(8, 6), dpi=100)

    fig.set_layout_engine("compressed")

    gs = gridspec.GridSpec(2, 2, height_ratios=(1 / 4, 3 / 4), hspace=0., wspace=0.15, figure=fig)
    # gs.tight_layout(fig, pad=0)

    # Unmerged feature bins
    ax0 = fig.add_subplot(gs[0, 0])

    if region_annotations.is_continuous:
        bins = 20
    elif region_annotations.is_discrete:
        bins = unmerged_feature_bin_edges

    plot_distribution_bins(
        variable_values,
        unmerged_feature_bin_edges,
        unmerged_feature_pt_bins,
        bins=bins,
        ax=ax0,
        cmap=cmap,
        hist_scatter_kwargs=hist_scatter_kwargs,
    )
    ax0.set_box_aspect(1 / 3)

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(embedding[:, 0], embedding[:, 1], c=unmerged_feature_pt_bins, cmap=cmap, **scatter_kwargs)
    ax.set(box_aspect=1, aspect=1)
    ax.set_xticks([]), ax.set_yticks([])

    # Merged feature bins
    ax = fig.add_subplot(gs[0, 1])

    if region_annotations.is_continuous:
        bins = 20
    elif region_annotations.is_discrete:
        bins = merged_feature_bin_edges

    plot_distribution_bins(
        variable_values,
        merged_feature_bin_edges,
        merged_feature_pt_bins,
        bins=bins,
        ax=ax,
        cmap=cmap,
        hist_scatter_kwargs=hist_scatter_kwargs,
    )
    ax.set_box_aspect(1 / 3)

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(embedding[:, 0], embedding[:, 1], c=merged_feature_pt_bins, cmap=cmap, **scatter_kwargs)
    ax.set(box_aspect=1, aspect=1)
    ax.set(xticks=[], yticks=[])

    fig.draw_without_rendering()  # to calculate the Axes positions in the layout
    pad = 0.02  # in fractions of the figure height
    fig.suptitle(region_annotations.name, fontsize=12, ha="center", y=ax0.get_position().y1 + pad, verticalalignment="bottom")

    if return_fig:
        return fig
