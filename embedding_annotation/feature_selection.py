import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from sklearn import neighbors
from sklearn.neighbors import radius_neighbors_graph


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


def _uniform_adjacency_matrix(x: np.ndarray, scale: float, n_jobs: int = 1) -> sp.csr_matrix:
    adj = radius_neighbors_graph(
        x, radius=scale, metric="euclidean", include_self=False, n_jobs=n_jobs
    )
    return adj


def _gaussian_adjacency_matrix(x: np.ndarray, scale: float, n_jobs: int = 1) -> sp.csr_matrix:
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


def morans_i(
    embedding: np.ndarray,
    features: pd.DataFrame,
    scale: float = None,
    moran_threshold: float = 0.1,
    fdr_threshold: float = 0.01,
    n_jobs: int = 1,
    adj: sp.csr_matrix = None,
) -> pd.DataFrame:
    if embedding.shape[0] != features.shape[0]:
        raise ValueError(
            f"The number of samples in the feature matrix ({features.shape[0]}) does"
            f" not match the number of samples in the embedding ({embedding.shape[0]})."
        )

    # Construct adjacency matrix from the embedding
    if adj is None:
        adj = adjacency_matrix(
            embedding, scale=scale, weighting="gaussian", n_jobs=n_jobs
        )

    # Calculate Moran's I and the associated p-values
    scores = _morans_i(features.values, adj)
    pvals = _analytical_pvals(scores, features.values, adj)
    pvals_adjusted = FDR(pvals, m=features.shape[1])

    # Perform filtering and prepare final result
    df = pd.DataFrame.from_dict(
        {
            "feature": features.columns,
            "morans_i": scores,
            "pvalue": pvals,
            "fdr": pvals_adjusted,
        },
    )
    df = df.loc[df["fdr"] <= fdr_threshold]
    df = df.loc[df["morans_i"] >= moran_threshold]

    return df.reset_index(drop=True)


def _morans_i(x: np.ndarray, adj: sp.spmatrix) -> np.ndarray:
    assert (
        x.shape[0] == adj.shape[0]
    ), "Feature matrix dimensions do not match adjacency matrix."

    N = x.shape[0]
    W = adj.sum()

    x_centered = x - x.mean(axis=0)
    n = np.sum(x_centered * (adj.tocsr().dot(x_centered)), axis=0)
    d = np.sum(x_centered**2, axis=0)

    return N / W * n / (d + 1e-16)


def _analytical_pvals(
    scores: np.ndarray, x: np.ndarray, adj: sp.spmatrix
) -> np.ndarray:
    N = float(x.shape[0])
    W = float(adj.sum())

    expected = -1 / (N - 1)

    # Variance
    S1 = 1 / 2 * np.sum((adj + adj.T).power(2))
    S2 = np.sum((adj.sum(axis=0) + adj.sum(axis=1).T).A ** 2)

    n = 1 / N * np.sum((x - x.mean(axis=0)) ** 4, axis=0)
    d = (1 / N * np.sum((x - x.mean(axis=0)) ** 2, axis=0)) ** 2
    S3 = n / np.maximum(d, 1e-16)

    S4 = (N**2 - 3 * N + 3) * S1 - N * S2 + 3 * W**2
    S5 = (N**2 - N) * S1 - 2 * N * S2 + 6 * W**2

    variance = (N * S4 - S3 * S5) / (
        (N - 1) * (N - 2) * (N - 3) * W**2
    ) - expected**2

    z_scores = (scores - expected) / (np.sqrt(variance) + 1e-16)

    pvals = 1 - stats.norm.cdf(z_scores)

    return pvals


def FDR(p_values, dependent=False, m=None, ordered=False):
    if p_values is None or len(p_values) == 0 or (m is not None and m <= 0):
        return None

    p_values = np.array(p_values)
    if m is None:
        m = len(p_values)
    if not ordered:
        ordered = (np.diff(p_values) >= 0).all()
        if not ordered:
            indices = np.argsort(p_values)
            p_values = p_values[indices]

    if dependent:  # correct q for dependent tests
        m *= sum(1 / np.arange(1, m + 1))

    fdrs = (p_values * m / np.arange(1, len(p_values) + 1))[::-1]
    fdrs = np.array(np.minimum.accumulate(fdrs)[::-1])
    if not ordered:
        fdrs[indices] = fdrs.copy()

    return fdrs
