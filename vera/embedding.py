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


def _check_sample_weights(sample_weights, n_samples: int) -> np.ndarray:
    sample_weights = np.asarray(sample_weights, dtype=float)
    if sample_weights.shape != (n_samples,):
        raise ValueError(
            f"`sample_weights` must hold one weight per sample, i.e., have "
            f"shape ({n_samples},), got {sample_weights.shape}."
        )
    if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights < 0):
        raise ValueError("`sample_weights` must be finite and non-negative.")
    if np.any(sample_weights > 1):
        raise ValueError("`sample_weights` must not exceed one.")
    if not np.any(sample_weights > 0):
        raise ValueError("`sample_weights` must not all be zero.")
    return sample_weights


class Embedding:
    """A two-dimensional embedding and the density estimates computed on it.

    Parameters
    ----------
    embedding : np.ndarray
        The embedding coordinates, of shape (n_samples, 2).
    scale_factor : float, default=1
        Multiplies the estimated embedding scale, which sets the KDE bandwidth.
    n_density_grid_points : int, default=100
        Number of grid points per axis on which densities are evaluated.
    sample_weights : np.ndarray, optional
        Per-sample weights in [0, 1], e.g., derived from a sample-wise
        distortion measure of the embedding. A weight is the fraction of the
        sample's contribution that is kept: it multiplies the indicator values
        that act as KDE weights. Each weighted density is scaled by the peak of
        its unweighted counterpart, so the contour level applies to the
        unweighted density. A variable whose samples are all down-weighted
        therefore yields a smaller region, and none once its density stays
        below the contour level. Region membership and the rates computed on it
        are unaffected.
    """

    def __init__(
        self,
        embedding: np.ndarray,
        scale_factor: float = 1,
        n_density_grid_points: int = 100,
        sample_weights: np.ndarray = None,
    ):
        self.X = embedding
        self.scale_factor = scale_factor
        self.n_density_grid_points = n_density_grid_points
        if sample_weights is not None:
            sample_weights = _check_sample_weights(sample_weights, embedding.shape[0])
        self.sample_weights = sample_weights

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

        # A sample with no measurement carries no weight, so it shapes no
        # region. Left as NaN it would poison the whole density estimate.
        weights = np.nan_to_num(values, nan=0.0)
        density = self._kde(weights, kernel)
        if self.sample_weights is None:
            return Density(self._density_grid, density)

        # KDEpy normalizes the weights, so the weighted density is rescaled by
        # the fraction of the variable's weight that the sample weights keep
        weighted = weights * self.sample_weights
        weighted_density = self._kde(weighted, kernel)
        if np.any(weighted > 0):
            weighted_density *= weighted.sum() / weights.sum()
        return Density(self._density_grid, weighted_density, peak=density.max())

    def _kde(self, weights: np.ndarray, kernel: str) -> np.ndarray:
        """Density of the weighted samples on the grid, integrating to one, or
        zero everywhere if no sample carries weight."""
        if not np.any(weights > 0):
            return np.zeros(self._density_grid.shape[0])
        kde = KDEpy.FFTKDE(kernel=kernel, bw=self.scale).fit(self.X, weights=weights)
        return kde.evaluate(self._density_grid)

    def __eq__(self, other):
        if not isinstance(other, Embedding):
            return False

        return np.allclose(self.X, other.X)
