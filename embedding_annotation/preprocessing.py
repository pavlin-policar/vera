from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn import neighbors

from embedding_annotation import metrics
from embedding_annotation.data import (
    ingest,
    IntervalRule,
    DerivedVariable,
    EqualityRule,
    ExplanatoryVariable,
)
from embedding_annotation.region import Density, Region


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


def _discretize(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Discretize all continuous variables in the data frame."""
    df_ingested = ingest(df)

    # We can only discretize continuous columns
    cont_cols = [c for c in df_ingested.columns if c.is_continuous]
    othr_cols = [c for c in df_ingested.columns if not c.is_continuous]
    df_cont = df_ingested[cont_cols]
    df_othr = df_ingested[othr_cols]

    # If there are no continuous features to be discretized, return
    if len(cont_cols) == 0:
        return df_ingested

    # Use k-means for discretization
    from sklearn.preprocessing import KBinsDiscretizer

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        strategy="kmeans",
        encode="onehot-dense",
    )
    x_discretized = discretizer.fit_transform(df_cont.values)

    # Create derived features
    derived_features = []

    for variable, bin_edges in zip(cont_cols, discretizer.bin_edges_):
        # Ensure open intervals
        bin_edges = np.array(bin_edges)
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf

        for lower, upper in zip(bin_edges, bin_edges[1:]):
            rule = IntervalRule(lower, upper, value_name=variable.name)
            v = DerivedVariable(variable, rule)
            derived_features.append(v)

    assert len(derived_features) == len(
        discretizer.get_feature_names_out()
    ), "The number of derived features do not match discretization output!"

    df_cont = pd.DataFrame(x_discretized, columns=derived_features, index=df.index)
    return pd.concat([df_othr, df_cont], axis=1)


def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Create one-hot encodings for all discrete variables in the data frame."""
    df_ingested = ingest(df)

    # We can only discretize continuous columns
    disc_cols = [c for c in df_ingested.columns if c.is_discrete]
    if len(disc_cols) == 0:
        return df_ingested

    othr_cols = [c for c in df_ingested.columns if not c.is_discrete]
    df_disc = df_ingested[disc_cols]
    df_othr = df_ingested[othr_cols]

    x_onehot = pd.get_dummies(df_disc).values

    # Create derived features
    derived_features = []
    for variable in df_disc.columns:
        for category in variable.categories:
            rule = EqualityRule(category, value_name=variable.name)
            v = DerivedVariable(variable, rule)
            derived_features.append(v)

    assert (
        len(derived_features) == x_onehot.shape[1]
    ), "The number of derived features do not match one-hot output!"

    df_disc = pd.DataFrame(x_onehot, columns=derived_features, index=df.index)
    return pd.concat([df_othr, df_disc], axis=1)


def generate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    return _one_hot(_discretize(ingest(df)))


def generate_explanatory_features():
    ...


def merge_overfragmented_candidates(
    variables: list[ExplanatoryVariable],
    min_purity_gain=0.05,
    min_sample_overlap=0.5,
):
    variable_groups = defaultdict(list)
    for v in variables:
        variable_groups[v.base_variable].append(v)

    candidates = []
    for variable_group in variable_groups.values():
        for i in range(len(variable_group)):
            for j in range(i + 1, len(variable_group)):
                v1, v2 = variable_group[i], variable_group[j]
                if not v1.can_merge_with(v2):
                    continue

                new_variable = v1.merge_with(v2)
                purity_gain = new_variable.purity / np.maximum(v1.purity, v2.purity) - 1
                shared_sample_pct = metrics.shared_sample_pct(v1, v2)
                if (
                    purity_gain >= min_purity_gain
                    and shared_sample_pct >= min_sample_overlap
                ):
                    candidates.append(
                        {
                            "feature_1": v1,
                            "feature_2": v2,
                            "purity_gain": purity_gain,
                            "sample_overlap": shared_sample_pct,
                        }
                    )

    candidates = pd.DataFrame(
        candidates, columns=["feature_1", "feature_2", "purity_gain", "sample_overlap"]
    )

    # If a feature is to be merged with more than one variable, allow only a
    # single merge. Pick the merge with the largest gain
    candidates = candidates.sort_values("purity_gain", ascending=False)

    seen, idx_to_drop = set(), []
    for idx, row in candidates.iterrows():
        pair = frozenset([row["feature_1"], row["feature_2"]])
        if any(len(pair & s) > 0 for s in seen):
            idx_to_drop.append(idx)
        else:
            seen.add(pair)
    candidates.drop(index=idx_to_drop, inplace=True)

    return candidates.reset_index(drop=True)


def merge_overfragmented(
    variables: list[ExplanatoryVariable],
    min_purity_gain=0.05,
    min_sample_overlap=0.5,
):
    variables = set(variables)

    def _merge_variables(variables_to_merge: tuple[Any, Any]):
        """Merge all the regions in the list of tuples."""
        nonlocal variables

        # Sometimes, a feature should be merged more than once, so we can't
        # remove it immediately after merge
        variables_to_remove = set()
        for v1, v2 in variables_to_merge:
            variables.add(v1.merge_with(v2))
            variables_to_remove.update([v1, v2])

        variables -= variables_to_remove

    while (
        variables_to_merge := merge_overfragmented_candidates(
            variables,
            min_purity_gain=min_purity_gain,
            min_sample_overlap=min_sample_overlap,
        )
    ).shape[0] > 0:
        _merge_variables(
            variables_to_merge[["feature_1", "feature_2"]].itertuples(index=False)
        )

    return list(variables)


def filter_explanatory_features(features, min_purity, min_spatial_correlation):
    ...


def generate_explanatory_features(
    df: pd.DataFrame,
    embedding: np.ndarray,
    n_grid_points: int = 100,
    kernel: str = "gaussian",
    scale_factor: int = 1,
    contour_level: float = 0.25,
) -> list[ExplanatoryVariable]:
    # Convert the data frame so that it contains derived features
    df_derived = generate_derived_features(df)

    # Estimate the scale of the embedding
    scale = estimate_embedding_scale(embedding, scale_factor)

    # Create explanatory variables from each of the derived features
    explanatory_features = []
    for v in df_derived.columns.tolist():
        values = df_derived[v].values
        density = Density.from_embedding(
            embedding, values, n_grid_points=n_grid_points, kernel=kernel, bw=scale
        )
        region = Region.from_density(density=density, level=contour_level)
        explanatory_v = ExplanatoryVariable(
            v.base_variable,
            v.rule,
            values,
            region,
            embedding,
            scale=scale,
        )
        explanatory_features.append(explanatory_v)

    # Perform iterative merging

    return explanatory_features
