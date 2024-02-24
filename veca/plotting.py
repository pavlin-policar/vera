import operator
import string
import warnings
from collections.abc import Iterable
from functools import reduce
from itertools import cycle, chain
from textwrap import wrap
from typing import Any, Union

import glasbey
import matplotlib.colors as clr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from veca.region import Density, Region
from veca.variables import (
    ExplanatoryVariable, ExplanatoryVariableGroup, EmbeddingRegionMixin, Variable
)


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

        s = s * y

        cmap = clr.LinearSegmentedColormap.from_list(
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
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("equal")
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
        ax.scatter(embedding[:, 0], embedding[:, 1], **scatter_kwargs_, rasterized=True)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_box_aspect(1)
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


def _format_explanatory_variable(variable: ExplanatoryVariable, max_width=40):
    return "\n".join(wrap(variable.name, width=max_width))


def _format_explanatory_variable_group(var_group: ExplanatoryVariableGroup, max_width=40):
    var_strings = [str(v) for v in var_group.contained_variables]
    if max_width is not None:
        var_strings = [wrap(s, width=max_width) for s in var_strings]
    else:
        # Ensure consistent format with wrapped version
        var_strings = [[vs] for vs in var_strings]

    # Flatten string parts
    lines = reduce(operator.add, var_strings)

    return "\n".join(lines)


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
    show: bool = False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # If no edge color is specified, use the same color as the fill
    if edge_color is None:
        edge_color = fill_color

    # The purity effect should never go below the following threshold
    purity_effect_size = 0.75
    purity_factor = variable.purity * purity_effect_size + (1 - purity_effect_size)
    if indicate_purity:
        edge_alpha *= purity_factor
        fill_alpha *= purity_factor

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
                zorder=1,
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
            "zorder": 99,
        }
        label_kwargs_.update(label_kwargs)
        x, y = largest_polygon.centroid.coords[0]

        # Obtain the label string to draw over the region
        if isinstance(variable, ExplanatoryVariable):
            label_str = _format_explanatory_variable(variable)
        elif isinstance(variable, ExplanatoryVariableGroup):
            label_str = _format_explanatory_variable_group(variable)
        else:
            label_str = str(variable)

        label = ax.text(x, y, label_str, **label_kwargs_)
        if draw_detail and variable.plot_detail is not None:
            detail_kwargs_ = {
                "ha": "center",
                "va": "top",
                "fontsize": 9,
                "zorder": 99,
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

        ax.scatter(embedding[:, 0], embedding[:, 1], **scatter_kwargs_, rasterized=True)

    # Set title
    ax.set_title(variable.name)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.axis("equal")

    if show:
        ax.get_figure().show()

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
    show: bool = False,
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

        offset = 0.025
        l = string.ascii_lowercase[idx % len(string.ascii_lowercase)]
        ax[idx].text(
            offset, 1 - offset, l, transform=ax[idx].transAxes, va="top", ha="left",
            fontweight="bold", fontsize=16
        )

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if show:
        fig.show()

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
    ax.set_box_aspect(1)
    ax.axis("equal")

    return ax


def plot_regions_with_subregions(
    variables: list[ExplanatoryVariable],
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
    cmap: str = "tab10",
    show: bool = False,
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

        offset = 0.025
        l = string.ascii_lowercase[idx % len(string.ascii_lowercase)]
        ax[idx].text(
            offset, 1 - offset, l, transform=ax[idx].transAxes, va="top", ha="left",
            fontweight="bold", fontsize=16
        )

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if show:
        fig.show()

    if return_ax:
        return fig, ax


def plot_annotation(
    variables: list[ExplanatoryVariable],
    cmap: str = "tab10",
    ax=None,
    indicate_purity: bool = False,
    indicate_membership: bool = False,
    variable_colors: dict = None,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    detail_kwargs: dict = {},
    figwidth: int = 8,
    return_ax: bool = False,
    show: bool = False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(figwidth, figwidth))

    if variable_colors is None:
        # Glasbey crashes when requesting fewer colors than are in the cmap
        cmap_len = len(get_cmap_colors(cmap))
        request_len = max(cmap_len, len(variables))
        cmap = glasbey.extend_palette(cmap, palette_size=request_len)
        colors = iter(cmap)
        variable_colors = {
            variable: mcolors.to_rgb(next(colors)) for variable in variables
        }

    for variable in variables:
        plot_region(
            variable,
            fill_color=variable_colors[variable],
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
        "zorder": 2,
        "s": 6,
        "alpha": 1,
        **scatter_kwargs,
    }

    # Setup sample colors
    point_colors = np.array([mcolors.to_rgb("#aaaaaa")] * embedding.shape[0])

    if indicate_membership:
        # Set sample colors inside regions
        for variable in variables:
            group_indices = list(variable.contained_samples_tp())
            point_colors[group_indices] = variable_colors[variable]

        # Desaturate colors slightly
        point_colors = mcolors.rgb_to_hsv(point_colors)
        point_colors[:, 1] *= 0.75
        point_colors = mcolors.hsv_to_rgb(point_colors)

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=point_colors,
        rasterized=True,
        **scatter_kwargs_
    )

    # Clear title from drawn regions
    ax.set_title("")

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.axis("equal")

    if show:
        fig.show()

    if return_ax:
        return fig, ax


def plot_annotations(
    layouts: list[list[ExplanatoryVariable]],
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
    cmap: str = "tab10",
    indicate_purity: bool = False,
    indicate_membership: bool = True,
    variable_colors: dict = None,
    scatter_kwargs: dict = {},
    label_kwargs: dict = {},
    detail_kwargs: dict = {},
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
            variable_colors=variable_colors,
            scatter_kwargs=scatter_kwargs,
            label_kwargs=label_kwargs,
            detail_kwargs=detail_kwargs,
        )

        offset = 0.025
        l = string.ascii_lowercase[idx % len(string.ascii_lowercase)]
        ax[idx].text(
            offset, 1 - offset, l, transform=ax[idx].transAxes, va="top", ha="left",
            fontweight="bold", fontsize=16
        )

    # Hide remaining axes
    for idx in range(idx + 1, n_rows * per_row):
        ax[idx].axis("off")

    if show:
        fig.show()

    if return_ax:
        return fig, ax


def layout_variable_colors(layout, cmap="tab10"):
    layout_variables = list(chain.from_iterable(layout))

    # Glasbey crashes when requesting fewer colors than the cmap contains
    num_cmap_colors = len(get_cmap_colors(cmap))
    num_colors_to_request = max(num_cmap_colors, len(layout_variables))
    cmap = glasbey.extend_palette(
        cmap, palette_size=num_colors_to_request, colorblind_safe=True
    )

    # We use the variable rule as the key
    var_keys = {
        var_group: frozenset(v.rule for v in var_group.variables)
        for var_group in layout_variables
    }
    var_color_mapping = {
        var_keys[var_group]: mcolors.to_rgb(c)
        for var_group, c in zip(layout_variables, cmap)
    }
    var_group_colors = {
        var_group: var_color_mapping[var_keys[var_group]]
        for var_group in layout_variables
    }

    return var_group_colors


def plot_discretization(
    variable: Variable,
    cmap: str = "viridis",
    hist_scatter_kwargs: dict = {},
    scatter_kwargs: dict = {},
    return_fig: bool = False,
):
    import matplotlib.gridspec as gridspec

    def _get_bin_edges_continuous(explanatory_variables: list[ExplanatoryVariable]):
        edges = [v.rule.lower for v in explanatory_variables]
        edges += [explanatory_variables[-1].rule.upper]
        if np.isinf(edges[0]):
            edges[0] = variable.values.min()
        if np.isinf(edges[-1]):
            edges[-1] = variable.values.max()

        return edges

    def _get_bin_edges_discrete(explanatory_variables: list[ExplanatoryVariable]):
        num_contained_variables = [len(v.contained_variables) for v in explanatory_variables]
        edges = np.concatenate([[0], np.cumsum(num_contained_variables)]) + 0.5

        return edges

    def _get_sample_bin_indices(explanatory_variables: list[ExplanatoryVariable]):
        variable_values = np.vstack([v.values for v in explanatory_variables])
        variable_group_values = np.argmax(variable_values, axis=0)
        return variable_group_values

    unmerged_explanatory_variables = [
        expl_var
        for expl_vars in variable.explanatory_variables
        for expl_var in expl_vars.contained_variables
    ]

    if variable.is_continuous:
        _get_bin_edges_func = _get_bin_edges_continuous
    elif variable.is_discrete:
        _get_bin_edges_func = _get_bin_edges_discrete

    unmerged_feature_pt_bins = _get_sample_bin_indices(unmerged_explanatory_variables)
    unmerged_feature_bin_edges = _get_bin_edges_func(unmerged_explanatory_variables)
    merged_feature_pt_bins = _get_sample_bin_indices(variable.explanatory_variables)
    merged_feature_bin_edges = _get_bin_edges_func(variable.explanatory_variables)

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
            ax.axvline(edge, c="tab:red", lw=2)

        # ax.set_xticks(bin_edges[:-1] + 0.5)
        # ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Attribute Values", labelpad=3)
        ax.set_ylabel("Frequency", labelpad=3)
        ax.spines[["right", "top"]].set_visible(False)

        return ax

    v = variable  # The variable in question
    embedding = v.explanatory_variables[0].embedding.X

    variable_values = v.values
    if pd.api.types.is_categorical_dtype(variable_values):
        variable_values = variable_values.codes.astype(float) + 1

    fig = plt.figure(figsize=(8, 6), dpi=100)
    fig.suptitle(variable.name, fontsize=16, ha="center")
    gs = gridspec.GridSpec(2, 2, height_ratios=(1 / 4, 3 / 3), hspace=0., wspace=0.15)
    # gs.tight_layout(fig, pad=0)

    # Unmerged feature bins
    ax = fig.add_subplot(gs[0, 0])

    if variable.is_continuous:
        bins = 20
    elif variable.is_discrete:
        bins = unmerged_feature_bin_edges
    plot_distribution_bins(
        variable_values,
        unmerged_feature_bin_edges,
        unmerged_feature_pt_bins,
        bins=bins,
        ax=ax,
        cmap=cmap,
        hist_scatter_kwargs=hist_scatter_kwargs,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(embedding[:, 0], embedding[:, 1], c=unmerged_feature_pt_bins, cmap=cmap, **scatter_kwargs, rasterized=True)
    ax.axis("equal"), ax.set_box_aspect(1)
    ax.set_xticks([]), ax.set_yticks([])

    # Merged feature bins
    ax = fig.add_subplot(gs[0, 1])

    if variable.is_continuous:
        bins = 20
    elif variable.is_discrete:
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

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(embedding[:, 0], embedding[:, 1], c=merged_feature_pt_bins, cmap=cmap, **scatter_kwargs, rasterized=True)
    ax.axis("equal"), ax.set_box_aspect(1)
    ax.set_xticks([]), ax.set_yticks([])

    if return_fig:
        return fig
