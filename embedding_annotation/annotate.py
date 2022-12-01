import functools
from itertools import cycle
from typing import Callable, Optional

import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

import embedding_annotation.plotting as pl
import embedding_annotation.graph as g


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
    log: bool = False,
    n_grid_points: int = 100,
) -> tuple[np.ndarray, pd.DataFrame]:
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
    overlap_threshold: float = 0.9,
):
    # We only care about the densities that appear in the feature list
    densities = densities.loc[features]

    # Create a similarity weighted graph with edges appearing only if they have
    # overlap > overlap_threshold
    distances = pdist(densities.values, metric=overlap_index)
    graph = g.distances_to_graph(distances, threshold=overlap_threshold)
    node_labels = dict(enumerate(densities.index.values))
    graph = g.label_nodes(graph, node_labels)

    # Once we construct the graph, find the max-cliques. These will serve as our
    # merged "clusters"
    # cliques = g.max_cliques(graph)
    # clusts = {f"Cluster {cid}": vs for cid, vs in enumerate(cliques, start=1)}
    connected_components = g.connected_components(graph)
    clusts = {
        f"Cluster {cid}": list(c) for cid, c in enumerate(connected_components, start=1)
    }

    # Sum together the feature densities to obtain the cluster density
    clust_densities = pd.DataFrame(
        {cid: densities.loc[features].sum(axis=0) for cid, features in clusts.items()},
    ).T
    clust_densities = clust_densities / clust_densities.sum(axis=1).values[:, None]

    return clusts, clust_densities


def group_similar_features_dendrogram(
    features: list,
    densities: pd.DataFrame,
    metric: str | Callable = "js-divergence",
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


def readonly_array(x: np.ndarray) -> np.ndarray:
    """Ensure that the arrays are readonly and can't be changed later on."""
    if not isinstance(x, np.ndarray):
        return x
    if x.flags.writeable:
        x = x.copy()
        x.setflags(write=False)

    return x


class ReadonlyDict(dict):
    def __setitem__(self, key, value):
        raise TypeError()

    def __delitem__(self, key):
        raise TypeError()

    def clear(self):
        raise TypeError()

    def popitem(self):
        raise TypeError()

    def update(self, *args, **kwargs):
        raise TypeError()


class AnnotationMap:
    def __init__(
        self,
        grid: np.ndarray,
        embedding: np.ndarray,
        densities: dict[str, np.ndarray] = None,
    ):
        self.grid = readonly_array(grid)
        self.embedding = readonly_array(embedding)

        if densities is None:
            densities = {}

        self.densities = ReadonlyDict({
            k: readonly_array(v) for k, v in densities.items()
        })
        self.scaled_densities = ReadonlyDict({
            k: readonly_array(v / v.max()) for k, v in self.densities.items()
        })

    def add(self, name: str, density: np.ndarray):
        if name in self.densities:
            raise KeyError(f"Density `{name}` already exists!")

        # Create new annotation map object=
        new_densities = self.densities.copy() | {name: density}  # create shallow copy
        new_annmap = AnnotationMap(self.grid, self.embedding, new_densities)

        return new_annmap

    def remove(self, name: str):
        new_densities = self.densities.copy()  # create shallow copy
        del new_densities[name]
        return AnnotationMap(self.grid, self.embedding, new_densities)

    def __len__(self):
        return len(self.densities)

    def __contains__(self, item):
        return item in self.densities

    @staticmethod
    def _density_dict_to_array(d: dict) -> np.ndarray:
        """Convert the dictionary with numpy arrays as values to a stacked array."""
        return np.vstack(list(d.values()))

    @property
    def joint_density(self):
        """Calculate the joint density of the scaled densities."""
        if not self.densities:
            return np.zeros(self.grid.shape[0])
        vals = np.sum(self._density_dict_to_array(self.scaled_densities), axis=0)
        return vals / vals.sum()

    def plot_annotation(
        self,
        levels: int = 5,
        cmap: str = "tab10",
        ax=None,
        contour_kwargs: Optional[dict] = {},
        contourf_kwargs: Optional[dict] = {},
        scatter_kwargs: Optional[dict] = {},
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        hues = iter(cycle(pl.get_cmap_hues(cmap)))
        levels_ = np.linspace(0, 1, num=levels)  # scaled densities always in [0, 1]

        for key, density in self.scaled_densities.items():
            pl.plot_feature_density(
                self.grid,
                density,
                levels=levels_,
                skip_first=True,
                cmap=pl.hue_colormap(next(hues), levels=levels, min_saturation=0.1),
                ax=ax,
                contourf_kwargs=contourf_kwargs,
                contour_kwargs=contour_kwargs,
            )

        if self.embedding is not None:
            scatter_kwargs_ = {
                "zorder": 1,
                "c": "k",
                "s": 6,
                "alpha": 0.1,
                **scatter_kwargs,
            }
            ax.scatter(self.embedding[:, 0], self.embedding[:, 1], **scatter_kwargs_)

        return ax

    def rank_overlap_densities(self, densities: pd.DataFrame) -> pd.DataFrame:
        results = []
        for name, new_d in densities.iterrows():
            total_overlap = 0  # the overlap will be the sum of overlaps with each density
            for existing_d in self.densities.values():
                total_overlap += overlap_index(existing_d, new_d)

            results.append({"feature": name, "score": total_overlap})

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
            ax = self.plot_annotation(
                levels=levels, cmap=cmap, ax=ax, contourf_kwargs={"alpha": 0}
            )

        density_scaled = density / density.max()

        worst_overlap = 0
        worst_density = np.zeros_like(density)
        for existing_d in self.scaled_densities.values():
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
