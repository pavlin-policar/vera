from functools import cached_property

import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
import shapely.geometry as geom
from sklearn import neighbors
import KDEpy

from veca.region import Density


def kth_neighbor_distance(x: np.ndarray, k_neighbors: int, n_jobs: int = 1) -> float:
    """Find the median distance of each point's k-th nearest neighbor."""
    nn = neighbors.NearestNeighbors(n_neighbors=k_neighbors, n_jobs=n_jobs)
    nn.fit(x)
    distances, indices = nn.kneighbors()

    return np.median(distances[:, -1])


def estimate_embedding_scale(embedding: np.ndarray, scale_factor: float = 1) -> float:
    """Estimate the scale of the embedding."""
    k_neighbors = int(np.floor(np.sqrt(embedding.shape[0])))
    scale = kth_neighbor_distance(embedding, k_neighbors)
    scale *= scale_factor
    return scale


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


class Embedding:
    def __init__(self, embedding: np.ndarray, scale_factor: float = 1, n_density_grid_points: int = 100):
        self.X = embedding
        self.scale_factor = scale_factor
        self.n_density_grid_points = n_density_grid_points

    @cached_property
    def adj(self):
        return adjacency_matrix(self.X, scale=self.scale, weighting="gaussian")

    @cached_property
    def scale(self):
        return estimate_embedding_scale(self.X, self.scale_factor)

    @property
    def shape(self):
        return self.X.shape

    @cached_property
    def points(self):
        return [geom.Point(p) for p in self.X]

    @cached_property
    def _density_grid(self):
        return KDEpy.utils.autogrid(self.X, self.scale * 3, self.n_density_grid_points)

    def esimtimate_density(self, values: np.ndarray, kernel: str = "gaussian") -> Density:
        kde = KDEpy.FFTKDE(kernel=kernel, bw=self.scale).fit(self.X, weights=values)
        kde_esimates = kde.evaluate(self._density_grid)
        return Density(self._density_grid, kde_esimates)
