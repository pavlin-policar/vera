import warnings
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import vera.graph as g
import vera.metrics as metrics
from vera.embedding import Embedding
from vera.region import Region
from vera.rules import IntervalRule, EqualityRule
from vera.variables import (
    Variable,
    DiscreteVariable,
    ContinuousVariable,
    IndicatorVariable,
)
from vera.region_annotations import RegionAnnotation, CompositeRegionAnnotation


def _pd_dtype_to_variable(col_name: Union[str, Variable], col_type, col_vals) -> Variable:
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
        vals = col_vals[1].values.codes.astype(float)
        vals[col_vals[1].values.isna()] = np.nan
        variable = DiscreteVariable(
            col_name,
            values=vals,
            categories=col_type.categories.tolist(),
            ordered=col_type.ordered,
        )
    elif pd.api.types.is_numeric_dtype(col_type):
        variable = ContinuousVariable(col_name, values=col_vals[1].values)
    else:
        raise ValueError(
            f"Only categorical and numeric dtypes supported! Got " f"`{col_type.name}`."
        )

    return variable


def ingest(df: pd.DataFrame) -> list[Variable]:
    """Convert a pandas DataFrame to a list of VERA variables."""
    return list(
        _pd_dtype_to_variable(*p) for p in zip(df.columns, df.dtypes, df.items())
    )


def ingested_to_pandas(variables: list[Variable]) -> pd.DataFrame:
    df_new = pd.DataFrame()

    for v in variables:
        if isinstance(v, IndicatorVariable):
            df_new[v.name] = pd.Categorical(v.values)
        elif isinstance(v, DiscreteVariable):
            vals = np.full_like(v.values, fill_value=np.nan, dtype=object)
            mask = ~np.isnan(v.values)
            vals[mask] = np.array(v.categories)[v.values[mask].astype(int)]
            # vals = np.array(v.categories)[v.values.astype(int)]
            col = pd.Categorical(vals, ordered=v.ordered, categories=v.categories)
            df_new[v.name] = col
        elif isinstance(v, ContinuousVariable):
            df_new[v.name] = v.values
        else:
            raise ValueError(f"Unrecognized variable type `{v.__class__.__name__}`!")

    return df_new


def __discretize_const(variable: ContinuousVariable) -> IndicatorVariable:
    """Convert constant features into discrete equality rules"""
    uniq_val = variable.values[0]
    rule = EqualityRule(uniq_val, value_name=variable.name)
    const_vals = np.ones(variable.values.shape[0])
    return [IndicatorVariable(variable, rule, const_vals)]


def __discretize_nonconst(variable: ContinuousVariable, n_bins: int) -> IndicatorVariable:
    # Discretize non-constant features
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.exceptions import ConvergenceWarning

    col_vals = pd.Series(variable.values)
    col_vals_non_nan = col_vals.dropna()

    n_bins = np.minimum(n_bins, col_vals_non_nan.nunique())

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        strategy="kmeans",
        encode="onehot-dense",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        x_discretized = discretizer.fit_transform(col_vals_non_nan.values[:, None])

    # We discretize the non-NaN values, so ensure that the rows containing
    # NaNs are re-inserted as zeros
    df_discretized = pd.DataFrame(x_discretized, index=col_vals_non_nan.index)
    df_discretized = df_discretized.reindex(col_vals.index, fill_value=0)

    # Prepare rules and variables
    bin_edges = discretizer.bin_edges_[0]

    # Ensure open intervals
    bin_edges = np.array(bin_edges)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    derived_vars = []
    for idx, (lower, upper) in enumerate(zip(bin_edges, bin_edges[1:])):
        rule = IntervalRule(lower, upper, value_name=variable.name)
        values = df_discretized.loc[:, idx].values
        v = IndicatorVariable(variable, rule, values)
        derived_vars.append(v)

    assert len(derived_vars) == len(df_discretized.columns), \
        "The number of derived features do not match discretization output!"

    return derived_vars


def _discretize(variables: list[Variable] | pd.DataFrame, n_bins: int = 5) -> list[Variable]:
    """Discretize all continuous variables in the data frame.

    TODO: This function does imputation, but this really shouldn't be done here
    It should be up to the user to ensure there are no NaNs in the data, or up
    to us to ignore them.
    """
    if not isinstance(variables, list):
        variables = ingest(variables)

    new_vars = []
    for v in variables:
        # If the variable is not continuous, there is nothing to do
        if not v.is_continuous:
            new_vars.append(v)

        else:
            if len(np.unique(v.values)) == 1:
                disc_vars = __discretize_const(v)
            else:
                disc_vars = __discretize_nonconst(v, n_bins)

            new_vars.extend(disc_vars)

    return new_vars


def _one_hot(variables: list[Variable] | pd.DataFrame) -> list[Variable]:
    """Create one-hot encodings for all discrete variables in the data frame."""
    if not isinstance(variables, list):
        variables = ingest(variables)

    new_vars = []
    for v in variables:
        # If the variable is not discrete, there is nothing to do
        if not v.is_discrete:
            new_vars.append(v)

        else:
            xi_onehot = pd.get_dummies(v.values).values.astype(np.float32)
            for idx, category in enumerate(v.categories):
                rule = EqualityRule(category, value_name=v.name)
                values = xi_onehot[:, idx]
                new_var = IndicatorVariable(v, rule, values)
                new_vars.append(new_var)

    return new_vars


def expand_df(
    df: pd.DataFrame, n_discretization_bins: int = 5, filter_constant: bool = True
) -> list[IndicatorVariable]:
    # Filter out features with identical values
    if filter_constant:
        df = df.loc[:, df.nunique(axis=0) > 1]

    variables = _one_hot(_discretize(ingest(df), n_bins=n_discretization_bins))

    # Filter out columns with zero occurences
    variables = [v for v in variables if v.values.sum() > 0]

    return variables


def generate_region_annotations(
    variables: list[Variable],
    embedding: Embedding | np.ndarray,
    scale_factor: float = 1,
    kernel: str = "gaussian",
    contour_level: float = 0.25,
) -> list[RegionAnnotation]:
    # Create embedding instance which will be shared across all explanatory
    # variables. The shared instance is necessary to avoid slow recomputation of
    # adjacency matrices
    if not isinstance(embedding, Embedding):
        embedding = Embedding(embedding, scale_factor=scale_factor)

    # Create explanatory variables from each of the derived features
    region_annotations = []
    for v in tqdm(variables):
        density = embedding.estimate_density(v.values, kernel=kernel)
        region = Region.from_density(
            embedding=embedding, density=density, level=contour_level
        )
        ra = RegionAnnotation(v, region)
        region_annotations.append(ra)

    return region_annotations


def merge_overfragmented(
    region_annotations: list[RegionAnnotation],
    min_sample_overlap: float = 0.5,
    min_purity_gain: float = 0.5,
) -> list[RegionAnnotation]:
    # If we only have a single variable, there is nothing to merge
    if len(region_annotations) == 1:
        return region_annotations

    def _dist(ra1, ra2):
        if not ra1.can_merge_with(ra2):
            return 0

        shared_sample_pct = metrics.max_shared_sample_pct(ra1, ra2)
        if shared_sample_pct < min_sample_overlap:
            return 0

        new_ra = ra1.merge_with(ra2)
        ra1_purity_gain = metrics.purity(new_ra) / metrics.purity(ra1) - 1
        ra2_purity_gain = metrics.purity(new_ra) / metrics.purity(ra2) - 1
        purity_gain = np.max([ra1_purity_gain, ra2_purity_gain])

        return int(purity_gain >= min_purity_gain)

    def _merge_round(region_annotations):
        # Group region annotatins based on base variables
        var_groups = defaultdict(list)
        for ra in region_annotations:
            var_groups[ra.variable.base_variable].append(ra)

        merged_ras = []
        for k, var_group in tqdm(var_groups.items()):
            dists = metrics.pdist(var_group, _dist)
            graph = g.similarities_to_graph(dists, threshold=0.5)
            node_labels = dict(enumerate(var_group))
            graph = g.label_nodes(graph, node_labels)
            merge_groups = g.connected_components(graph)

            for c in merge_groups:
                new_ra = CompositeRegionAnnotation(list(c))
                merged_ras.append(new_ra)

        return merged_ras

    # TODO: This can be sped up by only recomputing variable groups that have
    # changed in the last round. We now recompute all variable groups every time
    prev_len = len(region_annotations)
    while len(region_annotations := _merge_round(region_annotations)) < prev_len:
        prev_len = len(region_annotations)

    return region_annotations
