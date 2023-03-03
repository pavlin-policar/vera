from collections import Counter
from typing import Callable, Any

import contourpy
import numpy as np
import pandas as pd
import shapely.geometry as geom
from KDEpy import FFTKDE
from scipy.cluster.hierarchy import linkage, fcluster

import embedding_annotation.graph as g
from embedding_annotation.data import Variable


class Density:
    def __init__(self, grid: np.ndarray, values: np.ndarray):
        self.grid = grid
        self.values = values / values.sum()
        self.values_scaled = values / values.max()

    def _get_xyz(self, scaled: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_grid_points = int(np.sqrt(self.grid.shape[0]))  # always a square grid
        x, y = np.unique(self.grid[:, 0]), np.unique(self.grid[:, 1])
        vals = [self.values, self.values_scaled][scaled]
        z = vals.reshape(n_grid_points, n_grid_points).T
        return x, y, z

    def get_contours_at(self, level: float) -> list[np.ndarray]:
        x, y, z = self._get_xyz(scaled=True)

        contour_generator = contourpy.contour_generator(
            x, y, z, corner_mask=False, chunk_size=0
        )
        return contour_generator.lines(level)

    def get_polygons_at(self, level: float) -> geom.MultiPolygon:
        return geom.MultiPolygon(
            [geom.Polygon(c) for c in self.get_contours_at(level)]
        )

    def __add__(self, other: "Density") -> "CompositeDensity":
        if not isinstance(other, Density):
            raise ValueError(
                f"Cannot merge `{self.__class__.__name__}` with object of type "
                f"`{other.__class__.__name__}`"
            )
        return CompositeDensity([self, other])


class CompositeDensity(Density):
    def __init__(self, densities: list[Density]):
        self.base_densities = densities
        joint_density = np.sum(np.vstack([d.values for d in densities]), axis=0)
        grid = densities[0].grid
        # TODO: Check grids
        super().__init__(grid, joint_density)


class Region:
    def __init__(self, feature: Variable, density: Density, level: float = 0.25):
        self.feature = feature
        self.level = level
        self.density = density

        self.polygon = density.get_polygons_at(level)

    @property
    def region_parts(self):
        return self.polygon.geoms

    @property
    def num_parts(self):
        return len(self.region_parts)

    def __add__(self, other: "Region"):
        if not np.allclose(self.density.grid, other.density.grid):
            raise RuntimeError("Grids must match when adding two density objects")

        return CompositeRegion([self.density, other.density])

    def __repr__(self):
        n = self.num_parts
        return f"Region: `{str(self.feature)}`, {n} part{'s'[:n^1]}"

    def __eq__(self, other: "Region") -> bool:
        """We will check for equality only on the basis of the variable."""
        if not isinstance(other, Region):
            return False
        return self.feature == other.feature

    def __hash__(self):
        """Hashing only on the basis of the variable."""
        return hash(self.feature)


class CompositeRegion(Region):
    def __init__(self, feature: str, regions: list[Region]):
        self.name = feature or " + ".join(d.feature for d in regions)
        self.grid = regions[0].grid

        self._base_densities = regions
        joint_density = np.sum(np.vstack([d.density.values for d in regions]), axis=0)
        self._density = joint_density / joint_density.sum()
        self._density_scaled = joint_density / joint_density.max()


def estimate_feature_densities(
    features: list[Any],
    embedding: np.ndarray,
    feature_matrix: pd.DataFrame,
    log: bool = False,
    n_grid_points: int = 100,
    kernel: str = "gaussian",
) -> dict[Any, Density]:
    densities = {}

    for feature in features:
        x = feature_matrix[feature].values
        if log:
            x = np.log1p(x)

        kde = FFTKDE(kernel=kernel).fit(embedding, weights=x)
        grid, points = kde.evaluate(n_grid_points)

        densities[feature] = Density(grid, points)

    return densities


def find_regions(
    densities: dict[Variable, Density],
    level: float = 0.25,
) -> list[Region]:
    """Identify regions for each feature at a specified contour level."""
    return [
        Region(variable, density, level=level)
        for variable, density in densities.items()
    ]


def _density_pdist(densities: dict[str, Region], f: Callable):
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


def intersection_area(r1: Region, r2: Region) -> float:
    p1, p2 = r1.polygon, r2.polygon
    return p1.intersection(p2).area


def intersection_over_union(r1: Region, r2: Region) -> float:
    p1, p2 = r1.polygon, r2.polygon
    return p1.intersection(p2).area / p1.union(p2).area


def intersection_over_union_dist(r1: Region, r2: Region) -> float:
    return 1 - intersection_over_union(r1, r2)


def inbetween_convex_hull_ratio(r1: Region, r2: Region) -> float:
    """Calculate the ratio between the area of the empty space and the polygon
    areas if we were to compute the convex hull around both p1 and p2"""
    p1, p2 = r1.polygon, r2.polygon

    total = (p1 | p2).convex_hull
    # Remove convex hulls of p1 and p2 from total area
    inbetween = total - p1.convex_hull - p2.convex_hull
    # Re-add p1 and p2 to total_area
    total = inbetween | p1 | p2

    return inbetween.area / total.area


def stage_1_merge_candidates(
    regions: list[Region],
    overlap_threshold: float = 0.75,
):
    result = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            r1, r2 = regions[i], regions[j]
            if not r1.feature.can_merge_with(r2.feature):
                continue

            p1, p2 = r1.polygon, r2.polygon

            overlap_ij = p1.intersection(p2).area / p1.area
            overlap_ji = p2.intersection(p1).area / p2.area
            overlap = max(overlap_ij, overlap_ji)

            result.append({"region_1": r1, "region_2": r2, "overlap": overlap})

    df = pd.DataFrame(result)
    df = df.loc[df["overlap"] >= overlap_threshold]
    return df


def stage_1_merge_regions(
    regions: list[Region],
    merge_features = None,
    overlap_threshold: float = 0.75,
) -> list[Region]:
    """

    Parameters
    ----------
    regions: list[Region]
        The regions to be merged. This list can contain regions that should not
        be merged as well.
    merge_features
    overlap_threshold: float
        If merge candidates are provided, this value is ignored.

    Returns
    -------
    list[Region]
        A new list of regions, where similar regions have been merged.
    """
    # Create copy, we don't want to modify the original list
    regions = list(regions)

    def _merge_regions(merge_features: tuple[Region, Region]):
        """Merge all the regions in the list of tuples."""
        for r1, r2 in merge_features:
            new_region = Region(
                r1.feature.merge_with(r2.feature), r1.density + r2.density
            )
            regions.remove(r1), regions.remove(r2), regions.append(new_region)

    if merge_features is not None:
        # TODO: This won't work,
        _merge_regions(merge_features)
    else:
        while (merge_features := stage_1_merge_candidates(regions, overlap_threshold)).shape[0] > 0:
            _merge_regions(
                merge_features[["region_1", "region_2"]].itertuples(index=False)
            )

    return regions


def group_similar_features(
    features: list[str],
    densities: dict[Region],
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
            f"Cluster {cid}": list(c)
            for cid, c in enumerate(connected_components, start=1)
        }
    else:
        raise ValueError(
            f"Unrecognized method `{method}`. Can be one of `max-cliques`, "
            f"`connected-components`"
        )

    clust_densities = {
        cid: CompositeRegion(
            feature=cid, regions=[d for d in densities.values() if d.feature in features]
        )
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
        cid: CompositeRegion(
            feature=cid, regions=[d for d in densities.values() if d.feature in features]
        )
        for cid, features in clusts.items()
    }

    return clusts, clust_densities


def optimize_layout(
    densities: dict[str, Region], max_overlap: float = 0
) -> list[list[str]]:
    density_names = list(densities.keys())

    overlap = _density_pdist(densities, intersection_area)
    graph = g.similarities_to_graph(overlap, threshold=max_overlap)
    node_labels = dict(enumerate(density_names))
    graph = g.label_nodes(graph, node_labels)

    independent_sets = g.independent_sets(graph)

    return independent_sets
