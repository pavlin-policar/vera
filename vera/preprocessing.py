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
    RegionDescriptor,
)
from vera.region_annotation import RegionAnnotation


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

    if isinstance(col_type, pd.CategoricalDtype):
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


def ingest(data: pd.Series | pd.DataFrame) -> list[Variable]:
    """Convert a pandas DataFrame to a list of VERA variables."""
    if isinstance(data, pd.Series):
        return _pd_dtype_to_variable(data.name, data.dtype, (0, data))
    elif isinstance(data, pd.DataFrame):
        return list(
            _pd_dtype_to_variable(*p) for p in zip(data.columns, data.dtypes, data.items())
        )
    else:
        raise TypeError(
            f"Cannot ingest object of type `{data.__class__.__name__}`. Only "
            f"pd.Series and pd.DataFrame are supported!"
        )


def ingested_to_pandas(variables: list[Variable]) -> pd.DataFrame:
    """Convert a list of VERA variables to a pandas dataframe."""
    df_new = pd.DataFrame()

    for v in variables:
        if isinstance(v, IndicatorVariable):
            df_new[v.name] = pd.Series(v.values)
        elif isinstance(v, DiscreteVariable):
            vals = np.full_like(v.values, fill_value=np.nan, dtype=object)
            mask = ~np.isnan(v.values)
            vals[mask] = np.array(v.categories)[v.values[mask].astype(int)]
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
    """Discretize non-constant continuous variables."""
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.exceptions import ConvergenceWarning

    col_vals = pd.Series(variable.values)
    col_vals_non_nan = col_vals.dropna()

    n_bins = np.minimum(n_bins, col_vals_non_nan.nunique())

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        strategy="kmeans",
        encode="onehot-dense",
        random_state=0,
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


def discretize(variable: ContinuousVariable, n_bins: int = 5) -> list[IndicatorVariable]:
    """Discretize a continuous variable."""
    if not isinstance(variable, ContinuousVariable):
        raise TypeError("Can only discretize continuous variables!")

    if len(np.unique(variable.values)) == 1:
        disc_vars = __discretize_const(variable)
    else:
        disc_vars = __discretize_nonconst(variable, n_bins)

    return disc_vars


def one_hot(variable: DiscreteVariable) -> list[IndicatorVariable]:
    """One-hot endcode a discrete variable."""
    if not isinstance(variable, DiscreteVariable):
        raise TypeError("Can only one-hot-encode discrete variables!")

    one_hot_vars = []
    for idx, category in enumerate(variable.categories):
        rule = EqualityRule(category, value_name=variable.name)
        values = np.astype(variable.values == idx, float)
        new_var = IndicatorVariable(variable, rule, values)
        one_hot_vars.append(new_var)

    return one_hot_vars


def expand(variables: list[Variable], n_discretization_bins: int = 5) -> list[list[IndicatorVariable]]:
    """Expand a list of variables into indicator variables via discretization or
    one-hot encoding."""
    var_groups = []
    for variable in variables:
        if variable.is_continuous:
            expanded_vars = discretize(variable, n_bins=n_discretization_bins)
        elif variable.is_discrete:
            expanded_vars = one_hot(variable)
        elif variable.is_indicator:
            expanded_vars = [variable]
        else:
            raise RuntimeError(f"Variable type not recognized, got `{variable}`")

        var_groups.append(expanded_vars)

    # Filter out columns with zero occurences. This can happen for categorical
    # variables with categories that never actually occur in the data
    var_groups = [[v for v in var_group if v.values.sum() > 0] for var_group in var_groups]
    # If the filtering removed all the variables from a particular variable,
    # remove that group. In practice, this should never happen.
    var_groups = [var_group for var_group in var_groups if len(var_group) > 0]

    return var_groups


def expand_df(
    df: pd.DataFrame,
    n_discretization_bins: int = 5,
    filter_constant_features: bool = True,
) -> list[list[IndicatorVariable]]:
    # Filter out features with identical values
    if filter_constant_features:
        df = df.loc[:, df.nunique(axis=0) > 1]

    variables = ingest(df)

    expanded = expand(variables, n_discretization_bins=n_discretization_bins)

    return expanded


def extract_region_annotations(
    variables: list[list[IndicatorVariable]],
    embedding: Embedding | np.ndarray,
    scale_factor: float = 1,
    kernel: str = "gaussian",
    contour_level: float = 0.25,
) -> list[RegionAnnotation]:

    def _generate_single(v):
        density = embedding.estimate_density(v.values, kernel=kernel)
        region = Region.from_density(
            embedding=embedding, density=density, level=contour_level
        )
        return RegionAnnotation(region, v)

    # Create embedding instance which will be shared across all explanatory
    # variables. The shared instance is necessary to avoid slow recomputation of
    # adjacency matrices
    if not isinstance(embedding, Embedding):
        embedding = Embedding(embedding, scale_factor=scale_factor)

    # Create explanatory variables from each of the derived features
    region_annotations = []
    num_regions_to_estimate = sum(map(len, variables))
    with tqdm(total=num_regions_to_estimate) as pbar:
        for var_group in variables:
            ra_group = []
            for v in var_group:
                ra_group.append(_generate_single(v))
                pbar.update(1)
            region_annotations.append(ra_group)

    return region_annotations


def merge_overfragmented(
    region_annotations: list[RegionAnnotation],
    min_sample_overlap: float = 0.5,
) -> list[RegionAnnotation]:
    # If we only have a single variable, there is nothing to merge
    if len(region_annotations) == 1:
        return region_annotations

    def _merge_region_annotations(region_annotations):
        # If there is a single region annotation, just return that
        if len(region_annotations) == 1:
            return region_annotations[0]

        merged_region = Region.merge(
            [ra.region for ra in region_annotations]
        )
        merged_descriptor = RegionDescriptor.merge(
            [ra.descriptor for ra in region_annotations]
        )
        return RegionAnnotation(
            region=merged_region,
            descriptor=merged_descriptor,
            source_region_annotations=region_annotations,
        )

    def _dist(ra1: RegionAnnotation, ra2: RegionAnnotation):
        if not ra1.can_merge_with(ra2):
            return 0

        shared_sample_pct = metrics.max_shared_sample_pct(ra1, ra2)
        if shared_sample_pct < min_sample_overlap:
            return 0

        return 1

    def _merge_round(region_annotations):
        # Group region annotatins based on base variables. At this point, all
        # region descriptors should be instances of indicator variables, and no
        # variable groups should be present
        ra_groups = defaultdict(list)
        for ra in region_annotations:
            ra_groups[ra.descriptor.base_variable].append(ra)

        merged_ras = []
        for k, var_group in ra_groups.items():
            dists = metrics.pdist(var_group, _dist)
            graph = g.similarities_to_graph(dists, threshold=0.5)
            node_labels = dict(enumerate(var_group))
            graph = g.label_nodes(graph, node_labels)
            merge_groups = g.connected_components(graph)

            for c in merge_groups:
                new_ra = _merge_region_annotations(list(c))
                merged_ras.append(new_ra)

        return merged_ras

    prev_len = len(region_annotations)
    while len(region_annotations := _merge_round(region_annotations)) < prev_len:
        prev_len = len(region_annotations)

    return region_annotations
