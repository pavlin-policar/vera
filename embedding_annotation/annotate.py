from collections import Counter
from typing import Callable

import contourpy
import numpy as np
import pandas as pd
import shapely.geometry as geom
from KDEpy import FFTKDE
from scipy.cluster.hierarchy import linkage, fcluster

import embedding_annotation.graph as g


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


def _density_pdist(densities: dict[str, Density], f: Callable):
    densities = list(densities.values())

    n = len(densities)
    out_size = (n * (n - 1)) // 2
    result = np.zeros(out_size, dtype=np.float64)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            result[k] = f(densities[i], densities[j])
            k += 1
    return result


def intersection_area(d1: Density, d2: Density, level: float = 0.25) -> float:
    c1: geom.MultiPolygon = d1.get_polygons_at(level)
    c2: geom.MultiPolygon = d2.get_polygons_at(level)
    return c1.intersection(c2).area


def intersection_over_union(d1: Density, d2: Density, level: float = 0.25) -> float:
    c1: geom.MultiPolygon = d1.get_polygons_at(level)
    c2: geom.MultiPolygon = d2.get_polygons_at(level)
    return c1.intersection(c2).area / c1.union(c2).area


def intersection_over_union_dist(d1: Density, d2: Density, level: float = 0.25) -> float:
    c1: geom.MultiPolygon = d1.get_polygons_at(level)
    c2: geom.MultiPolygon = d2.get_polygons_at(level)
    return 1 - (c1.intersection(c2).area / c1.union(c2).area)


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
    threshold: float = 0.9,
    method: str = "max-cliques",
):
    # We only care about the densities that appear in the feature list
    densities = {k: d for k, d in densities.items() if k in features}

    # Create a similarity weighted graph with edges appearing only if they have
    # IoU > threshold
    distances = _density_pdist(densities, intersection_over_union)
    graph = g.similarities_to_graph(distances, threshold=threshold)
    node_labels = dict(enumerate(densities.keys()))
    graph = g.label_nodes(graph, node_labels)

    # Once we construct the graph, find the max-cliques. These will serve as our
    # merged "clusters"
    if method == "max-cliques":
        cliques = g.max_cliques(graph)
        clusts = {f"Cluster {cid}": vs for cid, vs in enumerate(cliques, start=1)}
    elif method == "connected-components":
        connected_components = g.connected_components(graph)
        clusts = {
            f"Cluster {cid}": list(c) for cid, c in enumerate(connected_components, start=1)
        }
    else:
        raise ValueError(
            f"Unrecognized method `{method}`. Can be one of `max-cliques`, "
            f"`connected-components`"
        )

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
    threshold: float = 0.1,
    plot_dendrogram: bool = False,
):
    # We only care about the densities that appear in the feature list
    densities = {k: d for k, d in densities.items() if k in features}

    # Perform complete-linkage hierarchical clustering to ensure that all the
    # features have at most the specified threshold distance between them
    distances = _density_pdist(densities, intersection_over_union_dist)
    Z = linkage(distances, method="complete")
    cluster_assignment = fcluster(Z, t=threshold, criterion="distance")
    cluster_assignment = cluster_assignment - 1  # clusters from linkage start at 1

    # Re-label the clusters so clusters with more elements come first
    cluster_counts = Counter(cluster_assignment)
    cluster_mapping = {k: i for i, (k, _) in enumerate(cluster_counts.most_common())}
    cluster_assignment = np.array([cluster_mapping[c] for c in cluster_assignment])

    if plot_dendrogram:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        fig = plt.figure(figsize=(24, 6))
        dendrogram(Z, color_threshold=threshold)
        ax = fig.get_axes()[0]
        ax.axhline(threshold, linestyle="dashed", c="k")

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
    density_names = list(densities.keys())

    overlap = _density_pdist(densities, intersection_area)
    graph = g.similarities_to_graph(overlap, threshold=max_overlap)
    node_labels = dict(enumerate(density_names))
    graph = g.label_nodes(graph, node_labels)

    independent_sets = g.independent_sets(graph)

    return independent_sets
