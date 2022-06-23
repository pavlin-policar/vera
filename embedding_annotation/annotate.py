import functools
from itertools import cycle
from typing import Union, Callable, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

import embedding_annotation.plotting as pl


def _ensure_normed(p: np.ndarray) -> np.ndarray:
    """Ensure that a vector is a valid probability distribution and sums to 1."""
    if not np.allclose(np.sum(p), 1):
        p = p / p.sum()
    return p


def kl_divergence(p1, p2):
    p1, p2 = _ensure_normed(p1), _ensure_normed(p2)
    return np.sum(p1 * np.log(p1 / np.maximum(p2, 1e-8)))


def jeffreys_divergence(p1, p2):
    return 0.5 * kl_divergence(p1, p2) + 0.5 * kl_divergence(p2, p1)


def jensen_shannon_divergence(p1, p2):
    p1, p2 = _ensure_normed(p1), _ensure_normed(p2)

    m = 0.5 * (p1 + p2)
    return 0.5 * kl_divergence(p1, m) + 0.5 * kl_divergence(p2, m)


def hellinger_distance(p1, p2):
    p1, p2 = _ensure_normed(p1), _ensure_normed(p2)
    return 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(p1) - np.sqrt(p2))


def bhattacharyya_distance(p1, p2):
    p1, p2 = _ensure_normed(p1), _ensure_normed(p2)
    return -np.log(np.sum(np.sqrt(p1) * np.sqrt(p2)))


def overlap_index(p1, p2):
    p1, p2 = _ensure_normed(p1), _ensure_normed(p2)
    return np.sum(np.minimum(p1, p2))


def overlap_distance(p1, p2):
    p1, p2 = _ensure_normed(p1), _ensure_normed(p2)
    return 1 - (0.5 * np.sum(np.abs(p1 - p2)))


density_metrics = {
    "sym-kl": jeffreys_divergence,
    "jeffreys-divergence": jeffreys_divergence,
    "js-divergence": jensen_shannon_divergence,
    "hellinger": hellinger_distance,
    "bhattacharyya": bhattacharyya_distance,
    "overlap": overlap_distance,
}


def estimate_feature_densities(
    features: list,
    embedding: np.ndarray,
    feature_matrix: pd.DataFrame,
    log=False,
    n_grid_points: int = 100,
) -> Tuple[np.ndarray, pd.DataFrame]:
    densities = []

    for feature in features:
        x = feature_matrix[feature].values
        if log:
            x = np.log1p(x)

        kde = FFTKDE().fit(embedding, weights=x)
        grid, points = kde.evaluate(n_grid_points)

        densities.append(points)

    densities = pd.DataFrame(densities, index=features)
    densities = densities / densities.sum(axis=1).values[:, None]  # normalize to 1

    return grid, densities


def group_similar_features(
    features: list,
    densities: pd.DataFrame,
    metric: Union[str, Callable] = "js-divergence",
    similarity_threshold: float = 0.1,
    plot_dendrogram: bool = False,
):
    # We only care about the densities that appear in the feature list
    densities = densities.loc[features]

    if isinstance(metric, str):
        if metric not in density_metrics:
            raise ValueError(
                f"Unrecognized distance metric `{metric}`. Available metrics are "
                f"{', '.join(density_metrics.keys())}."
            )
        metric = density_metrics[metric]

    # Return the absolute function of a wrapper. Needed because, for some reason,
    # the probability metrics sometimes return negative numbers (e.g. -0).
    def abs_(f):
        @functools.wraps(f)
        def _f(*args, **kwargs):
            return np.abs(f(*args, **kwargs))

        return _f

    # Perform complete-linkage hierarchical clustering to ensure that all the
    # features have at most the specified threshold distance between them
    distances = pdist(densities.values, metric=abs_(metric))
    Z = linkage(distances, method="complete")
    cluster_assignment = fcluster(Z, t=similarity_threshold, criterion="distance")
    cluster_assignment = cluster_assignment - 1  # clusters from linkage start at 1

    if plot_dendrogram:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        fig = plt.figure(figsize=(24, 6))
        dendrogram(Z, color_threshold=0.1)
        ax = fig.get_axes()[0]
        ax.axhline(similarity_threshold, linestyle="dashed", c="k")

    clusts = {
        f"Cluster {cid}": np.array(features)[cluster_assignment == cid].tolist()
        for cid in np.unique(cluster_assignment)
    }

    # Sum together the feature densities to obtain the cluster density
    clust_densities = pd.DataFrame(
        {cid: densities.loc[features].sum(axis=0) for cid, features in clusts.items()},
    ).T
    clust_densities = clust_densities / clust_densities.sum(axis=1).values[:, None]

    return clusts, clust_densities


def highest_density_interval(density: np.ndarray, hdi: float = 0.95) -> np.ndarray:
    """Zero out values that are not in the highest density interval."""
    sorted_vals = sorted(density, reverse=True)
    hdi_idx = np.argwhere(np.cumsum(sorted_vals) >= (hdi * np.sum(density)))[0, 0]
    density = np.where(density > sorted_vals[hdi_idx], density, 0)
    return density


class AnnotationMap:
    def __init__(
        self,
        grid: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        densities: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.grid = grid
        self.embedding = embedding

        self._densities = {}
        self._scaled_densities = {}

        if densities is not None:
            for k, d in densities.items():
                self.add(k, d)

    def add(self, name: str, density: np.ndarray):
        if name in self._densities:
            raise KeyError(f"Density `{name}` already exists!")
        self._densities[name] = density
        self._scaled_densities[name] = density / density.max()

    def remove(self, name: str):
        del self._densities[name]
        del self._scaled_densities[name]

    def plot_annotation(self, levels: int = 5, cmap: str = "tab10", ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        hues = iter(cycle(pl.get_cmap_hues(cmap)))
        levels_ = np.linspace(0, 1, num=levels)  # scaled densities always in [0, 1]

        for key, density in self._scaled_densities.items():
            pl.plot_feature_density(
                self.grid,
                density,
                levels=levels_,
                skip_first=True,
                cmap=pl.hue_colormap(next(hues), levels=levels, min_saturation=0.1),
                ax=ax,
            )

        if self.embedding is not None:
            ax.scatter(
                self.embedding[:, 0],
                self.embedding[:, 1],
                c="k",
                s=6,
                zorder=1,
                alpha=0.1,
            )

        return ax

    @staticmethod
    def _density_dict_to_array(d: dict) -> np.ndarray:
        """Convert the dictionary with numpy arrays as values to a stacked array."""
        return np.vstack(list(d.values()))

    def rank_overlap_densities(self, densities: pd.DataFrame) -> pd.DataFrame:
        results = []
        for name, new_d in densities.iterrows():
            worst_overlap = 0
            for existing_d in self._densities.values():
                score = overlap_index(existing_d, new_d)
                if score > worst_overlap:
                    worst_overlap = score

            results.append({"feature": name, "score": worst_overlap})

        return pd.DataFrame(results)

    def plot_overlap_with(
        self,
        density: np.ndarray,
        plot_annotations: bool = True,
        levels: int = 5,
        cmap: str = "tab10",
        ax=None,
    ):
        if plot_annotations:
            ax = self.plot_annotation(levels=levels, cmap=cmap, ax=ax)

        density_scaled = density / density.max()

        worst_overlap = 0
        worst_density = np.zeros_like(density)
        for existing_d in self._scaled_densities.values():
            p_overlap = np.minimum(existing_d, density_scaled)
            score = np.sum(p_overlap)
            if score > worst_overlap:
                worst_overlap = score
                worst_density = p_overlap

        levels_ = np.linspace(0, 1, num=levels)  # scaled densities always in [0, 1]
        pl.plot_feature_density(
            self.grid,
            worst_density,
            levels=levels_,
            skip_first=True,
            cmap=pl.hue_colormap(1, levels=levels, min_saturation=0.25),
            ax=ax,
            contourf_kwargs={"alpha": 1, "zorder": 3},
            contour_kwargs={"zorder": 4, "linewidths": 1, "colors": "k"},
        )
