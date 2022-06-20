from collections.abc import Iterable
from textwrap import wrap
from typing import Optional, Union

import matplotlib.colors as colors
import numpy as np
import pandas as pd


def plot_feature(
    feature_names,
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
    import matplotlib.pyplot as plt
    import matplotlib.colors as clr

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

    marker_str = ", ".join(feature_names)
    if title is None:
        ax.set_title("\n".join(wrap(marker_str, 30)))
    else:
        ax.set_title(title)

    return ax


def plot_features(
    features,
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
    import matplotlib.pyplot as plt

    n_rows = len(features) // per_row
    if len(features) % per_row > 0:
        n_rows += 1

    figheight = figwidth / per_row * n_rows
    fig, ax = plt.subplots(nrows=n_rows, ncols=per_row, figsize=(figwidth, figheight))

    ax = ax.ravel()
    for axi in ax:
        axi.set_axis_off()

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

    if return_ax:
        return fig, ax


def get_cmap_hues(cmap: str):
    """Extract the hue values from a given colormap."""
    import matplotlib.cm

    cm = matplotlib.cm.get_cmap(cmap)
    hues = [c[0] for c in colors.rgb_to_hsv(cm.colors)]

    return np.array(hues)


def hue_colormap(
    hue: float, levels: int = 10, min_saturation: float = 0
) -> colors.ListedColormap:
    """Create an HSV colormap with varying saturation levels"""
    hsv = [[hue, s, 1] for s in np.linspace(min_saturation, 1, num=levels)]
    rgb = colors.hsv_to_rgb(hsv)
    cmap = colors.ListedColormap(rgb)

    return cmap


def plot_feature_density(
    grid: np.ndarray,
    density: np.ndarray,
    embedding: Optional[np.ndarray] = None,
    levels: Union[int, np.ndarray] = 5,
    skip_first: bool = True,
    ax=None,
    cmap="RdBu_r",
):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    n_grid_points = int(np.sqrt(grid.shape[0]))  # always a square grid
    xs, ys = np.unique(grid[:, 0]), np.unique(grid[:, 1])

    z = density.reshape(n_grid_points, n_grid_points).T

    tck = None
    if isinstance(levels, Iterable):
        if skip_first:
            levels = levels[1:]
    else:
        if skip_first:
            tck = ticker.MaxNLocator(nbins=levels, prune="lower")

    ax.contourf(xs, ys, z, levels=levels, cmap=cmap, zorder=1, locator=tck)
    ax.contour(
        xs, ys, z, levels=levels, linewidths=1, colors="k", zorder=1, locator=tck
    )

    if embedding is not None:
        ax.scatter(embedding[:, 0], embedding[:, 1], c="k", s=6, zorder=1, alpha=0.1)

    return ax


def plot_feature_densities(
    features: list,
    grid: np.ndarray,
    densities: pd.DataFrame,
    embedding: Optional[np.ndarray] = None,
    per_row: int = 4,
    figwidth: int = 24,
    return_ax: bool = False,
):
    import matplotlib.pyplot as plt

    n_rows = len(features) // per_row
    if len(features) % per_row > 0:
        n_rows += 1

    figheight = figwidth / per_row * n_rows
    fig, ax = plt.subplots(nrows=n_rows, ncols=per_row, figsize=(figwidth, figheight))

    ax = ax.ravel()
    for axi in ax:
        axi.set_axis_off()

    for idx, feature in enumerate(features):
        ax[idx].set_title(feature)

        plot_feature_density(grid, densities.loc[feature].values, embedding, ax=ax[idx])
        # import matplotlib.ticker as ticker
        # ticker.MaxNLocator(prune="lower")

    if return_ax:
        return fig, ax
