import warnings
from collections.abc import Iterable
from itertools import cycle
from textwrap import wrap
from typing import Any, Union

import matplotlib.colors as clr
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from veca.region import Density, Region
from veca.variables import ExplanatoryVariable, EmbeddingRegionMixin, Variable


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
    ax=None,
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

        cmap = clr.LinearSegmentedColormap.from_list(
            "expression", [colors[0], colors[1]], N=256
        )
        ax.scatter(
            embedding[sort_idx, 0],
            embedding[sort_idx, 1],
            c=y[sort_idx],
            s=s,
            alpha=alpha,
            rasterized=True,
            cmap=cmap,
            zorder=zorder,
        )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("equal")

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
) -> colors.ListedColormap:
    """Create an HSV colormap with varying saturation levels"""
    if isinstance(levels, Iterable):
        hsv = [[hue, (s + min_saturation) / (1 + min_saturation), 1] for s in levels]
    else:
        num_levels = len(levels) if isinstance(levels, Iterable) else levels
        hsv = [[hue, s, 1] for s in np.linspace(min_saturation, 1, num=num_levels)]

    rgb = colors.hsv_to_rgb(hsv)
    cmap = colors.ListedColormap(rgb)

    return cmap


def plot_density(
    density: Union[ExplanatoryVariable, Region, Density],
    embedding: np.ndarray = None,
    levels: Union[int, np.ndarray] = 5,
    skip_first: bool = True,
    ax=None,
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
        }
        scatter_kwargs_.update(scatter_kwargs)
        ax.scatter(embedding[:, 0], embedding[:, 1], **scatter_kwargs_)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis("equal")

    return ax


def plot_densities(
    variables: list[ExplanatoryVariable],
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
            embedding=variable.embedding.X,
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


def _add_region_info(variable: EmbeddingRegionMixin, ax, offset=0.025):
    info = [
        ("Purity", "purity"),
        ("Moran's I", "morans_i"),
        ("Geary's C", "gearys_c"),
    ]
    s_parts = [f"{name}: {getattr(variable, attr):.2f}" for name, attr in info]
    s = "\n".join(s_parts)
    txt = ax.text(
        1 - offset,
        1 - offset,
        s,
        transform=ax.transAxes,
        va="top",
        ha="right",
        linespacing=1.5,
    )
    return txt


def plot_region(
    variable: ExplanatoryVariable,
    ax=None,
    fill_color="tab:blue",
    edge_color=None,
    fill_alpha=0.25,
    edge_alpha=1,
    lw=1,
    draw_label=False,
    draw_detail=False,
    draw_scatterplot=True,
    highlight_members=True,
    member_color="tab:red",
    add_region_info=False,
    indicate_purity: bool = False,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    detail_kwargs: dict = {},
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # If no edge color is specified, use the same color as the fill
    if edge_color is None:
        edge_color = fill_color

    if indicate_purity:
        edge_alpha *= variable.purity
        fill_alpha *= variable.purity

    for geom in variable.region.polygon.geoms:
        # Polygon plotting code taken from
        # https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib
        path = Path.make_compound_path(
            Path(np.asarray(geom.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in geom.interiors],
        )
        # Fill
        if fill_color is not None:
            fill_patch = PathPatch(
                path,
                fill=True,
                color=fill_color,
                alpha=fill_alpha,
            )
            ax.add_patch(fill_patch)
        # Boundary
        edge_patch = PathPatch(
            path,
            fill=False,
            edgecolor=edge_color,
            alpha=edge_alpha,
            lw=lw,
            zorder=10,
        )
        ax.add_patch(edge_patch)

    if draw_label:
        # Draw the lable on the largest polygon in the region
        largest_polygon = max(variable.region.polygon.geoms, key=lambda x: x.area)
        label_kwargs_ = {
            "ha": "center",
            "va": "bottom" if draw_detail else "center",
            "fontsize": 9,
        }
        label_kwargs_.update(label_kwargs)
        x, y = largest_polygon.centroid.coords[0]
        label_parts = map(
            lambda x: "\n".join(wrap(x, width=80)),
            str(variable.plot_label).split("\n"),
        )
        label = ax.text(x, y, "\n".join(label_parts), **label_kwargs_)
        if draw_detail and variable.plot_detail is not None:
            detail_kwargs_ = {
                "ha": "center",
                "va": "top",
                "fontsize": 9,
            }
            detail_kwargs_.update(detail_kwargs)
            label = ax.text(x, y, f"({variable.plot_detail})", **detail_kwargs_)
        # label.set_bbox(dict(facecolor="white", alpha=0.75, edgecolor="white"))

    if add_region_info:
        _add_region_info(variable, ax)

    # Plot embedding scatter plot
    if draw_scatterplot:
        embedding = variable.embedding.X
        scatter_kwargs_ = {
            "zorder": 1,
            "c": "#999999",
            "s": 6,
            "alpha": 1,
        }
        scatter_kwargs_.update(scatter_kwargs)
        if highlight_members:
            other_color = scatter_kwargs_.get("c")
            c = np.array([other_color, member_color])[variable.values.astype(int)]
            scatter_kwargs_["c"] = c
            scatter_kwargs_["alpha"] = 1

        ax.scatter(embedding[:, 0], embedding[:, 1], **scatter_kwargs_)

    # Set title
    ax.set_title(variable.name)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis("equal")

    return ax


def plot_regions(
    variables: list[ExplanatoryVariable],
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
    fill_color="tab:blue",
    edge_color=None,
    fill_alpha=0.25,
    edge_alpha=1,
    lw=1,
    draw_labels=False,
    draw_details=False,
    highlight_members=True,
    member_color="tab:red",
    add_region_info=False,
    indicate_purity: bool = False,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    detail_kwargs: dict = {},
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
        plot_region(
            variable,
            ax=ax[idx],
            fill_color=fill_color,
            edge_color=edge_color,
            fill_alpha=fill_alpha,
            edge_alpha=edge_alpha,
            lw=lw,
            draw_label=draw_labels,
            draw_detail=draw_details,
            highlight_members=highlight_members,
            member_color=member_color,
            add_region_info=add_region_info,
            indicate_purity=indicate_purity,
            scatter_kwargs=scatter_kwargs,
            label_kwargs=label_kwargs,
            detail_kwargs=detail_kwargs,
        )

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if return_ax:
        return fig, ax


def plot_region_with_subregions(
    variable: ExplanatoryVariable,
    ax=None,
    cmap: str = "tab10",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    plot_region(
        variable,
        ax=ax,
        highlight_members=False,
        fill_color="#cccccc",
        edge_color="#666666",
    )

    hues = iter(cycle(get_cmap_colors(cmap)))
    for subvariable, c in zip(variable.contained_variables, hues):
        plot_region(
            subvariable,
            ax=ax,
            highlight_members=False,
            fill_color=c,
            draw_scatterplot=False,
        )

    # Set title
    ax.set_title(variable.name)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis("equal")

    return ax


def plot_regions_with_subregions(
    variables: list[ExplanatoryVariable],
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
    cmap: str = "tab10",
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
        plot_region_with_subregions(
            variable,
            ax=ax[idx],
            cmap=cmap,
        )

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if return_ax:
        return fig, ax


def plot_annotation(
    variables: list[ExplanatoryVariable],
    cmap: str = "tab10",
    ax=None,
    indicate_purity: bool = False,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    detail_kwargs: dict = {},
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    hues = iter(cycle(get_cmap_colors(cmap)))

    for variable in variables:
        plot_region(
            variable,
            fill_color=next(hues),
            ax=ax,
            draw_label=True,
            draw_detail=False,
            highlight_members=False,
            draw_scatterplot=False,
            indicate_purity=indicate_purity,
            label_kwargs=label_kwargs,
            detail_kwargs=detail_kwargs,
        )

    embedding = variables[0].embedding.X
    scatter_kwargs_ = {
        "zorder": 1,
        "c": "#aaaaaa",
        "s": 6,
        "alpha": 1,
        **scatter_kwargs,
    }
    ax.scatter(embedding[:, 0], embedding[:, 1], **scatter_kwargs_)

    # Clear title from drawn regions
    ax.set_title("")

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis("equal")

    return ax


def plot_annotations(
    layouts: list[list[ExplanatoryVariable]],
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
    cmap: str = "tab10",
    indicate_purity: bool = False,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    detail_kwargs: dict = {},
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
            scatter_kwargs=scatter_kwargs,
            label_kwargs=label_kwargs,
            detail_kwargs=detail_kwargs,
        )

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if return_ax:
        return fig, ax


def plot_discretization(
    variable: Variable,
    features: list[ExplanatoryVariable],
    cmap: str = "viridis",
    hist_scatter_alpha: float = 1,
    hist_scatter_size: float = 36,
):
    import matplotlib.gridspec as gridspec

    def _get_point_bins(variable_groups):
        variable_group_values = {}
        for k in variable_groups:
            if not k.is_continuous:
                continue
            variable_values = np.vstack([v.values for v in variable_groups[k]])
            variable_group_values[k] = np.argmax(variable_values, axis=0)
        return variable_group_values

    def _get_bin_edges(variable_groups):
        variable_group_edges = {}
        for k in variable_groups:
            if not k.is_continuous:
                continue
            variable_edges = [v.rule.lower for v in variable_groups[k]][1:]
            variable_group_edges[k] = variable_edges
        return variable_group_edges

    final_feature_var_groups = {v: v.explanatory_variables for v in features}
    unmerged_feature_var_groups = {
        v: expl_vars.contained_variables
        for v in features
        for expl_vars in v.explanatory_variables
    }

    final_feature_pt_bins = _get_point_bins(final_feature_var_groups)
    final_feature_bin_edges = _get_bin_edges(final_feature_var_groups)
    unmerged_feature_pt_bins = _get_point_bins(unmerged_feature_var_groups)
    unmerged_feature_bin_edges = _get_bin_edges(unmerged_feature_var_groups)

    def plot_distribution_bins(x, bin_edges, x_bins, ax, cmap=None, scatter_alpha=1, scatter_size=36):
        if len(bin_edges) > 1:
            bin_width = bin_edges[1] - bin_edges[0]
            bin_edges_ = [bin_edges[0] - bin_width] + bin_edges + [bin_edges[-1] + bin_width]
            d, bins, *_ = ax.hist(
                x, bins=bin_edges_, alpha=0.5, edgecolor="k", align="mid", zorder=5
            )
        else:
            d, bins, *_ = ax.hist(x, bins=20, alpha=0.5, edgecolor="k", align="mid", zorder=5)

        bin_width = bins[1] - bins[0]
        x_jitter = x + np.random.normal(0, bin_width * 0.05, size=x.shape)
        y_jitter = np.abs(np.random.normal(0, d.max() / 2, size=x.shape))
        ax.scatter(x_jitter, y_jitter, c=x_bins, cmap=cmap, alpha=scatter_alpha, s=scatter_size)

        for edge in bin_edges:
            ax.axvline(edge, c="tab:red", lw=2)

        ax.set_xlabel("Attribute Values")
        ax.set_ylabel("Frequency")
        ax.spines[["right", "top"]].set_visible(False)

        return ax

    v = variable  # The variable in question
    embedding = v.explanatory_variables[0].embedding.X

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=(1 / 3, 2 / 3), hspace=0.2, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    plot_distribution_bins(
        v.values,
        unmerged_feature_bin_edges[v],
        unmerged_feature_pt_bins[v],
        ax=ax,
        cmap=cmap,
        scatter_alpha=hist_scatter_alpha,
        scatter_size=hist_scatter_size
    )

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(embedding[:, 0], embedding[:, 1], c=unmerged_feature_pt_bins[v], cmap=cmap)
    ax.axis("equal"), ax.set_box_aspect(1)
    ax.set_xticks([]), ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    plot_distribution_bins(
        v.values,
        final_feature_bin_edges[v],
        final_feature_pt_bins[v],
        ax=ax,
        cmap=cmap,
        scatter_alpha=hist_scatter_alpha,
        scatter_size=hist_scatter_size
    )

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(embedding[:, 0], embedding[:, 1], c=final_feature_pt_bins[v], cmap=cmap)
    ax.axis("equal"), ax.set_box_aspect(1)
    ax.set_xticks([]), ax.set_yticks([])
