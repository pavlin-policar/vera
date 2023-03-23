from collections import defaultdict
from typing import Any, Union

import numpy as np
import pandas as pd

from embedding_annotation import metrics
from embedding_annotation.embedding import Embedding
from embedding_annotation.region import Density, Region
from embedding_annotation.rules import IntervalRule, EqualityRule
from embedding_annotation.variables import (
    DerivedVariable,
    ExplanatoryVariable,
    DiscreteVariable,
    ContinuousVariable,
    Variable,
)


def _pd_dtype_to_variable(col_name: Union[str, Variable], col_type) -> Variable:
    """Convert a column from a pandas DataFrame to a Variable instance.

    Parameters
    ----------
    col_name: str
    col_type: dtype

    Returns
    -------
    Variable

    """
    if isinstance(col_name, Variable):
        return col_name

    if pd.api.types.is_categorical_dtype(col_type):
        variable = DiscreteVariable(
            col_name,
            categories=col_type.categories.tolist(),
            ordered=col_type.ordered,
        )
    elif pd.api.types.is_numeric_dtype(col_type):
        variable = ContinuousVariable(col_name)
    else:
        raise ValueError(
            f"Only categorical and numeric dtypes supported! Got " f"`{col_type.name}`."
        )

    return variable


def ingest(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a pandas DataFrame to a DataFrame the library can understand.

    This really just creates a copy of the data frame, but swaps out the columns
    for instances of our `Variable` objects, so we know which derived
    variables can be merged later on.

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame

    """
    df_new = df.copy()
    new_index = map(lambda p: _pd_dtype_to_variable(*p), zip(df.columns, df.dtypes))
    df_new.columns = pd.Index(list(new_index))
    return df_new


def ingested_to_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df_new = pd.DataFrame(index=df.index)

    for column in df.columns:
        if isinstance(column, DerivedVariable):
            df_new[column.name] = pd.Categorical(df[column])
        elif isinstance(column, DiscreteVariable):
            col = pd.Categorical(
                df[column], ordered=column.ordered, categories=column.categories
            )
            df_new[column.name] = col
        elif isinstance(column, ContinuousVariable):
            df_new[column.name] = df[column]
        else:  # probably an uningested df column
            df_new[column] = df[column]

    return df_new


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

    # Sklearn discretization doesn't support NaNs, so impute median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    df_cont_imputed = imputer.fit_transform(df_cont.values)

    # Use k-means for discretization
    from sklearn.preprocessing import KBinsDiscretizer

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        strategy="kmeans",
        encode="onehot-dense",
    )
    x_discretized = discretizer.fit_transform(df_cont_imputed)

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


def generate_derived_features(
    df: pd.DataFrame, n_discretization_bins: int = 5
) -> pd.DataFrame:
    # Filter out features with identical values
    df = df.loc[:, df.nunique() > 1]

    df = _one_hot(_discretize(ingest(df), n_bins=n_discretization_bins))
    # Filter out columns with zero occurences
    df = df.loc[:, df.sum(axis=0) > 0]

    return df


def convert_derived_features_to_explanatory(
    df,
    embedding,
    scale_factor: float = 1,
    n_grid_points: int = 100,
    kernel: str = "gaussian",
    contour_level: float = 0.25,
):
    # Create embedding instance which will be shared across all explanatory
    # variables. The shared instance is necessary to avoid slow recomputation of
    # adjacency matrices
    embedding = Embedding(embedding, scale_factor=scale_factor)

    # Create explanatory variables from each of the derived features
    explanatory_features = []
    for v in df.columns.tolist():
        values = df[v].values
        density = Density.from_embedding(
            embedding,
            values,
            n_grid_points=n_grid_points,
            kernel=kernel,
        )
        region = Region.from_density(density=density, level=contour_level)
        explanatory_v = ExplanatoryVariable(
            v.base_variable,
            v.rule,
            values,
            region,
            embedding,
        )
        explanatory_features.append(explanatory_v)

    return explanatory_features


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
                moran_gain = (
                    new_variable.morans_i / np.maximum(v1.morans_i, v2.morans_i) - 1
                )
                shared_sample_pct = metrics.max_shared_sample_pct(v1, v2)
                if (
                    purity_gain >= min_purity_gain
                    and shared_sample_pct >= min_sample_overlap
                    and moran_gain >= 0
                ):
                    candidates.append(
                        {
                            "feature_1": v1,
                            "feature_2": v2,
                            "purity_gain": purity_gain,
                            "moran_gain": moran_gain,
                            "sample_overlap": shared_sample_pct,
                        }
                    )

    candidates = pd.DataFrame(
        candidates,
        columns=[
            "feature_1",
            "feature_2",
            "purity_gain",
            "moran_gain",
            "sample_overlap",
        ],
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
