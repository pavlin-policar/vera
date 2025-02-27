import numbers
from functools import cached_property

import numpy as np
import shapely.geometry as geom
from sklearn import neighbors
import KDEpy


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


class Embedding:
    def __init__(self, embedding: np.ndarray, scale_factor: float = 1, n_density_grid_points: int = 100):
        self.X = embedding
        self.scale_factor = scale_factor
        self.n_density_grid_points = n_density_grid_points

    @cached_property
    def scale(self) -> float:
        return estimate_embedding_scale(self.X, self.scale_factor)

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    @cached_property
    def points(self) -> list[geom.Point]:
        return [geom.Point(p) for p in self.X]

    @cached_property
    def _density_grid(self):
        return KDEpy.utils.autogrid(self.X, self.scale * 3, self.n_density_grid_points)

    def estimate_density(self, values: np.ndarray, kernel: str = "gaussian") -> "Density":
        # Avoid circular imports
        from vera.region import Density

        if isinstance(values, numbers.Number):
            values = values * np.ones(self.X.shape[0])

        kde = KDEpy.FFTKDE(kernel=kernel, bw=self.scale).fit(self.X, weights=values)
        kde_esimates = kde.evaluate(self._density_grid)
        return Density(self._density_grid, kde_esimates)

    def __eq__(self, other):
        if not isinstance(other, Embedding):
            return False

        return np.allclose(self.X, other.X)
