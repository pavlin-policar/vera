from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from sklearn import neighbors


def pdist(l: list[Any], metric: Callable):
    n = len(l)
    out_size = (n * (n - 1)) // 2
    result = np.zeros(out_size, dtype=np.float64)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            result[k] = metric(l[i], l[j])
            k += 1
    return result


def dict_pdist(d: dict[Any, Any], metric: Callable):
    return pdist(list(d.values()), metric=metric)


def max_shared_sample_pct(v1: "ExplanatoryVariable", v2: "ExplanatoryVariable") -> float:
    v1_samples, v2_samples = v1.contained_samples, v2.contained_samples
    shared_samples = v1_samples & v2_samples
    v1_shared_sample_pct = len(shared_samples) / len(v1_samples)
    v2_shared_sample_pct = len(shared_samples) / len(v2_samples)
    return max(v1_shared_sample_pct, v2_shared_sample_pct)


def shared_sample_pct(v1: "ExplanatoryVariable", v2: "ExplanatoryVariable") -> float:
    v1_samples, v2_samples = v1.contained_samples, v2.contained_samples
    return len(v1_samples & v2_samples) / len(v1_samples | v2_samples)


def intersection_area(v1: "ExplanatoryVariable", v2: "ExplanatoryVariable") -> float:
    p1, p2 = v1.region.polygon, v2.region.polygon
    return p1.intersection(p2).area


def intersection_percentage(v1: "ExplanatoryVariable", v2: "ExplanatoryVariable") -> float:
    """The maximum percentage of the overlap between two regions."""
    p1, p2 = v1.region.polygon, v2.region.polygon
    i = p1.intersection(p2).area
    return max(i / p1.area, i / p2.area)


def intersection_over_union(v1: "ExplanatoryVariable", v2: "ExplanatoryVariable") -> float:
    p1, p2 = v1.region.polygon, v2.region.polygon
    return p1.intersection(p2).area / p1.union(p2).area


def intersection_over_union_dist(r1: "ExplanatoryVariable", r2: "ExplanatoryVariable") -> float:
    """Like intersection over union, but in distance form."""
    return 1 - intersection_over_union(r1, r2)


def inbetween_convex_hull_ratio(v1: "ExplanatoryVariable", v2: "ExplanatoryVariable") -> float:
    """Calculate the ratio between the area of the empty space and the polygon
    areas if we were to compute the convex hull around both p1 and p2"""
    p1, p2 = v1.region.polygon, v2.region.polygon

    total = (p1 | p2).convex_hull
    # Remove convex hulls of p1 and p2 from total area
    inbetween = total - p1.convex_hull - p2.convex_hull
    # Re-add p1 and p2 to total_area
    total = inbetween | p1 | p2

    return inbetween.area / total.area


def adjacency_matrix(
    x: np.ndarray,
    scale: float,
    weighting: str = "gaussian",
    n_jobs: int = 1,
) -> sp.csr_matrix:
    if weighting == "gaussian":
        return _gaussian_adjacency_matrix(x, scale=scale, n_jobs=n_jobs)
    elif weighting == "uniform":
        return _uniform_adjacency_matrix(x, scale=scale, n_jobs=n_jobs)
    else:
        raise ValueError(
            f"Unrecognized weighting scheme `{weighting}`. Must be one of "
            f"`gaussian`, `uniform`."
        )


def _uniform_adjacency_matrix(
    x: np.ndarray, scale: float, n_jobs: int = 1
) -> sp.csr_matrix:
    adj = neighbors.radius_neighbors_graph(
        x, radius=scale, metric="euclidean", include_self=False, n_jobs=n_jobs
    )
    return adj


def _gaussian_adjacency_matrix(
    x: np.ndarray, scale: float, n_jobs: int = 1
) -> sp.csr_matrix:
    n_samples = x.shape[0]

    nn = neighbors.NearestNeighbors(
        radius=scale * 3, metric="euclidean", n_jobs=n_jobs
    ).fit(x)
    neighbor_distances, neighbor_idx = nn.radius_neighbors()

    neighbor_weights = []
    for row in neighbor_distances:
        neighbor_weights.append(stats.norm(0, scale).pdf(row))

    indices, weights, indptr = [], [], [0]
    for idx, w in zip(neighbor_idx, neighbor_weights):
        assert len(idx) == len(w)
        indices.extend(idx)
        weights.extend(w)
        indptr.append(indptr[-1] + len(idx))

    adj = sp.csr_matrix((weights, indices, indptr), shape=(n_samples, n_samples))
    return adj


def morans_i(x: np.ndarray, adj: sp.spmatrix) -> np.ndarray:
    assert (
        x.shape[0] == adj.shape[0]
    ), "Feature matrix dimensions do not match adjacency matrix."

    N = x.shape[0]
    W = adj.sum()

    x_centered = x - x.mean(axis=0)
    n = np.sum(x_centered * (adj.tocsr().dot(x_centered)), axis=0)
    d = np.sum(x_centered**2, axis=0)

    return N / W * n / (d + 1e-16)


def gearys_c(x: np.ndarray, adj: sp.spmatrix) -> np.ndarray:
    assert (
        x.shape[0] == adj.shape[0]
    ), "Feature matrix dimensions do not match adjacency matrix."
    if x.ndim == 1:
        x = x[:, np.newaxis]
    adj = adj.tocoo()

    N = x.shape[0]
    W = adj.sum()

    diff = (x[adj.row, :] - x[adj.col, :]) ** 2
    n = adj.data.dot(diff)
    d = np.sum((x - np.mean(x, axis=0)) ** 2, axis=0)

    score = (N - 1) / (2 * W) * n / (d + 1e-16)

    if score.size == 1:  # if a single attribute is passed, the size is (1,)
        score = score[0]

    return score
