import functools
from typing import Callable

import contourpy
import shapely.geometry as geom
import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

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
    return 1 - overlap_index(p1, p2)
    # return 1 - (0.5 * np.sum(np.abs(p1 - p2)))


density_metrics = {
    "sym-kl": jeffreys_divergence,
    "jeffreys-divergence": jeffreys_divergence,
    "js-divergence": jensen_shannon_divergence,
    "hellinger": hellinger_distance,
    "bhattacharyya": bhattacharyya_distance,
    "overlap": overlap_distance,
}


class Density:
    def __init__(self, name: str, grid: np.ndarray, values: np.ndarray):
        self.name = name
        self.grid = grid
        self.values = values / np.sum(values)  # sum to one
        self.scaled_values = values / values.max()  # max=1

        self._contour_cache = {}
        self._polygon_cache = {}

    def highest_density_interval(self, hdi: float = 0.95) -> "Density":
        """Zero out values that are not in the highest density interval."""
        sorted_vals = sorted(self.values, reverse=True)
        hdi_idx = np.argwhere(np.cumsum(sorted_vals) >= hdi)[0, 0]
        density = np.where(self.values > sorted_vals[hdi_idx], self.values, 0)
        density /= density.sum()  # re-normalize density
        return Density(f"HDI({self.name}, {hdi:.2f})", self.grid, density)

    def __add__(self, other: "Density"):
        if not np.allclose(self.grid, other.grid):
            raise RuntimeError("Grids must match when adding two density objects")

        return CompositeDensity([self.values, other.values])

    def get_xyz(self, scaled: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_grid_points = int(np.sqrt(self.grid.shape[0]))  # always a square grid
        x, y = np.unique(self.grid[:, 0]), np.unique(self.grid[:, 1])
        vals = [self.values, self.scaled_values][scaled]
        z = vals.reshape(n_grid_points, n_grid_points).T
        return x, y, z

    def get_contours_at(self, level: float) -> list[np.ndarray]:
        if level in self._contour_cache:
            return self._contour_cache[level]

        x, y, z = self.get_xyz(scaled=True)

        contour_generator = contourpy.contour_generator(
            x, y, z, corner_mask=False, chunk_size=0
        )
        self._contour_cache[level] = contour_generator.lines(level)

        return self._contour_cache[level]

    def get_polygons_at(self, level: float) -> geom.MultiPolygon:
        if level in self._polygon_cache:
            return self._polygon_cache[level]

        self._polygon_cache[level] = geom.MultiPolygon(
            [geom.Polygon(c) for c in self.get_contours_at(level)]
        )

        return self._polygon_cache[level]


class CompositeDensity(Density):
    def __init__(self, name: str, densities: list[Density]):
        self.name = name or " + ".join(d.name for d in densities)
        self.grid = densities[0].grid

        self.orig_densities = densities
        joint_density = np.sum(np.vstack([d.values for d in densities]), axis=0)
        self.values = joint_density / joint_density.sum()
        self.scaled_values = joint_density / joint_density.max()

        self._contour_cache = {}
        self._polygon_cache = {}


def contour_overlap_area(d1: Density, d2: Density, level: float = 0.25) -> float:
    c1: geom.MultiPolygon = d1.get_polygons_at(level)
    c2: geom.MultiPolygon = d2.get_polygons_at(level)
    return c1.intersection(c2).area


def estimate_feature_densities(
    features: list[str],
    embedding: np.ndarray,
    feature_matrix: pd.DataFrame,
    log: bool = False,
    n_grid_points: int = 100,
) -> dict[str, Density]:
    densities = {}

    for feature in features:
        x = feature_matrix[feature].values
        if log:
            x = np.log1p(x)

        kde = FFTKDE().fit(embedding, weights=x)
        grid, points = kde.evaluate(n_grid_points)

        densities[feature] = Density(feature, grid, points)

    return densities


def density_dict_to_df(densities: dict[str, Density]) -> pd.DataFrame:
    x = np.vstack([d.values for d in densities.values()])
    return pd.DataFrame(x, index=densities.keys())


def group_similar_features(
    features: list[str],
    densities: dict[Density],
    overlap_threshold: float = 0.9,
):
    # We only care about the densities that appear in the feature list
    densities = {d: v for d, v in densities.items() if d in features}
    densities_df = density_dict_to_df(densities)

    # Create a similarity weighted graph with edges appearing only if they have
    # overlap > overlap_threshold
    distances = pdist(densities_df.values, metric=overlap_index)
    graph = g.distances_to_graph(distances, threshold=overlap_threshold)
    node_labels = dict(enumerate(densities.keys()))
    graph = g.label_nodes(graph, node_labels)

    # Once we construct the graph, find the max-cliques. These will serve as our
    # merged "clusters"
    # cliques = g.max_cliques(graph)
    # clusts = {f"Cluster {cid}": vs for cid, vs in enumerate(cliques, start=1)}
    connected_components = g.connected_components(graph)
    clusts = {
        f"Cluster {cid}": list(c) for cid, c in enumerate(connected_components, start=1)
    }

    clust_densities = {
        cid: CompositeDensity(name=cid, densities=[
            d for d in densities.values() if d.name in features
        ])
        for cid, features in clusts.items()
    }

    return clusts, clust_densities


def group_similar_features_dendrogram(
    features: list,
    densities: pd.DataFrame,
    metric: str | Callable = "js-divergence",
    similarity_threshold: float = 0.1,
    plot_dendrogram: bool = False,
):
    # We only care about the densities that appear in the feature list
    densities_df = density_dict_to_df(densities).loc[features]

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
    distances = pdist(densities_df.values, metric=abs_(metric))
    Z = linkage(distances, method="complete")
    cluster_assignment = fcluster(Z, t=similarity_threshold, criterion="distance")
    cluster_assignment = cluster_assignment - 1  # clusters from linkage start at 1

    if plot_dendrogram:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        fig = plt.figure(figsize=(24, 6))
        dendrogram(Z, color_threshold=similarity_threshold)
        ax = fig.get_axes()[0]
        ax.axhline(similarity_threshold, linestyle="dashed", c="k")

    clusts = {
        f"Cluster {cid}": np.array(features)[cluster_assignment == cid].tolist()
        for cid in np.unique(cluster_assignment)
    }

    clust_densities = {
        cid: CompositeDensity(name=cid, densities=[
            d for d in densities.values() if d.name in features
        ])
        for cid, features in clusts.items()
    }

    return clusts, clust_densities


def optimize_layout(densities: dict[str, Density], max_overlap: float = 0) -> list[list[str]]:
    density_names = sorted(list(densities.keys()))

    N = len(densities)
    overlap = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            overlap[i, j] = overlap[j, i] = contour_overlap_area(
                densities[density_names[i]], densities[density_names[j]],
            )
    graph = g.distances_to_graph(overlap, threshold=max_overlap)
    node_labels = dict(enumerate(density_names))
    graph = g.label_nodes(graph, node_labels)

    independent_sets = g.independent_sets(graph)

    return independent_sets
