import functools
from typing import Union, Callable, Tuple

from KDEpy import FFTKDE
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


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


density_metrics = {
    "sym-kl": jeffreys_divergence,
    "jeffreys-divergence": jeffreys_divergence,
    "js-divergence": jensen_shannon_divergence,
    "hellinger": hellinger_distance,
    "bhattacharyya": bhattacharyya_distance,
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
