import warnings
from collections import defaultdict
from typing import Any, Iterable, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import vera.graph as g
import vera.metrics as metrics
from vera.embedding import Embedding
from vera.region import Region
from vera.rules import IntervalRule, EqualityRule, IndicatorRule
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
    elif pd.api.types.is_bool_dtype(col_type):
        # A boolean column takes two values and nothing in between, so it is
        # described by them rather than by bins cut through a range
        variable = DiscreteVariable(
            col_name,
            values=col_vals[1].to_numpy(dtype=float, na_value=np.nan),
            categories=[False, True],
            ordered=False,
        )
    elif pd.api.types.is_numeric_dtype(col_type):
        variable = ContinuousVariable(col_name, values=col_vals[1].values)
    else:
        raise ValueError(
            f"Only categorical and numeric dtypes supported! Got " f"`{col_type.name}`."
        )

    return variable


#: Value of ``indicator_columns`` selecting every column of the data.
ALL_COLUMNS = "all"

#: Ways of selecting the indicator columns of a data frame: ``"all"``, a
#: collection of column names, or a mapping from column name to display label.
IndicatorColumns = Union[str, Iterable, dict, None]


def _resolve_indicator_columns(
    data: pd.Series | pd.DataFrame, indicator_columns: IndicatorColumns
) -> dict[Any, Any]:
    """Map each column to be ingested as an indicator to its display label."""
    columns = [data.name] if isinstance(data, pd.Series) else list(data.columns)

    if indicator_columns is None:
        labels = {}
    elif isinstance(indicator_columns, str):
        if indicator_columns != ALL_COLUMNS:
            raise ValueError(
                f"`indicator_columns` accepts `{ALL_COLUMNS!r}` or a collection "
                f"of column names, got the string `{indicator_columns!r}`. A "
                f"single column has to be wrapped in a list."
            )
        labels = {c: c for c in columns}
    elif isinstance(indicator_columns, dict):
        labels = dict(indicator_columns)
    else:
        labels = {c: c for c in indicator_columns}

    missing = [c for c in labels if c not in columns]
    if missing:
        raise KeyError(
            f"`indicator_columns` names columns that the data does not "
            f"contain: {', '.join(map(repr, missing))}."
        )

    predefined = [c for c in labels if isinstance(c, Variable)]
    if predefined:
        raise ValueError(
            f"`indicator_columns` names columns that already carry a variable: "
            f"{', '.join(map(repr, predefined))}. A column named by a variable "
            f"is ingested as that variable."
        )

    return labels


def _indicator_values(name: Any, values: pd.Series) -> np.ndarray:
    """Validate a column of a data frame as indicator values."""
    # A categorical of booleans still answers to `is_bool_dtype`, and is a
    # discrete variable
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype) or not pd.api.types.is_bool_dtype(dtype):
        raise ValueError(
            f"Indicator column `{name}` has dtype `{dtype}`. An indicator "
            f"column has to be boolean: cast a 0/1 column with "
            f"`.astype(bool)`, and leave continuous and categorical columns "
            f"out of `indicator_columns` to have them discretized or one-hot "
            f"encoded."
        )

    # Missing values stay missing: a sample the column has no measurement for
    # is neither flagged nor unflagged, and takes no part in the variable's
    # region or in the rates it is scored on
    return values.to_numpy(dtype=float, na_value=np.nan)


def _indicator_variable(name: Any, values: pd.Series, label: Any) -> IndicatorVariable:
    if label is None:
        raise ValueError(
            "An indicator variable is labelled by the name of its column, so "
            "an unnamed column needs an explicit label."
        )

    indicator_values = _indicator_values(name, values)
    base_variable = ContinuousVariable(name, values=indicator_values)
    return IndicatorVariable(base_variable, IndicatorRule(label), indicator_values)


def ingest_indicators(
    data: pd.Series | pd.DataFrame, labels: dict = None
) -> Union[IndicatorVariable, list[IndicatorVariable]]:
    """Convert boolean columns of a pandas DataFrame to VERA indicator variables.

    Each column becomes a single indicator describing its positive case, so a
    column recording whether a gene is expressed is annotated `CD3`, and the
    samples that lack it are left undescribed.

    Columns have to be of boolean dtype: a 0/1 column is rejected rather than
    read as a flag, since only the caller knows whether its values are a
    measurement or a coincidence of encoding.

    Missing values are supported through pandas' nullable ``boolean`` dtype,
    and are carried through as NaN. A sample the column has no measurement for
    is not flagged and is not unflagged either: it takes no part in shaping the
    variable's region, and it is left out of the rates the variable is scored
    on rather than counted against it.

    Parameters
    ----------
    data: pd.Series or pd.DataFrame
    labels: dict
        The text annotating each column, keyed by column name. Columns absent
        from the mapping are annotated with their name.

    """
    labels = dict(labels) if labels is not None else {}

    if isinstance(data, pd.Series):
        columns = [data.name]
    elif isinstance(data, pd.DataFrame):
        columns = list(data.columns)
    else:
        raise TypeError(
            f"Cannot ingest object of type `{data.__class__.__name__}`. Only "
            f"pd.Series and pd.DataFrame are supported!"
        )

    unknown = [c for c in labels if c not in columns]
    if unknown:
        raise KeyError(
            f"`labels` names columns that the data does not contain: "
            f"{', '.join(map(repr, unknown))}."
        )

    if isinstance(data, pd.Series):
        return _indicator_variable(data.name, data, labels.get(data.name, data.name))

    return [
        _indicator_variable(name, col_vals, labels.get(name, name))
        for name, col_vals in data.items()
    ]


def ingest(
    data: pd.Series | pd.DataFrame, indicator_columns: IndicatorColumns = None
) -> Union[Variable, list[Variable]]:
    """Convert a pandas DataFrame to a list of VERA variables.

    A series is converted to a single variable.

    Parameters
    ----------
    data: pd.Series or pd.DataFrame
    indicator_columns: str or iterable or dict
        The columns holding boolean indicators, which are described by their
        positive case alone instead of being discretized or one-hot encoded.
        ``"all"`` selects every column, a collection of column names selects
        those columns, and a mapping selects its keys and annotates them with
        its values. See :func:`ingest_indicators`.

    """
    labels = _resolve_indicator_columns(data, indicator_columns)

    if isinstance(data, pd.Series):
        if labels:
            return ingest_indicators(data, labels=labels)
        return _pd_dtype_to_variable(data.name, data.dtype, (0, data))
    elif isinstance(data, pd.DataFrame):
        variables = []
        for col_name, col_type, col_vals in zip(
            data.columns, data.dtypes, data.items()
        ):
            if col_name in labels:
                variables.append(
                    _indicator_variable(col_name, col_vals[1], labels[col_name])
                )
            else:
                variables.append(
                    _pd_dtype_to_variable(col_name, col_type, col_vals)
                )
        return variables
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
            df_new[str(v.rule)] = pd.Series(v.values)
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


def __discretize_const(variable: ContinuousVariable) -> list[IndicatorVariable]:
    """Convert constant features into discrete equality rules"""
    measured = ~np.isnan(variable.values)
    uniq_val = variable.values[measured][0]
    rule = EqualityRule(uniq_val, value_name=variable.name)
    const_vals = np.where(measured, 1.0, np.nan)
    return [IndicatorVariable(variable, rule, const_vals)]


def _decimals(x: float, max_decimals: int = 15) -> int:
    """The fewest decimal places that write `x` exactly."""
    for d in range(max_decimals + 1):
        if np.isclose(round(x, d), x, rtol=1e-12, atol=0):
            return d
    return max_decimals


def _round_edge(edge: float, lo: float, hi: float) -> float:
    """The roundest number `t` with `lo < t <= hi`.

    A bin edge `t` sends the samples `x >= t` to the bin above it, so any `t`
    in the gap between the largest value below the edge and the smallest
    value at or above it splits the samples exactly as `edge` does. Steps of
    1, 5, and 2 times a power of ten are tried from coarse to fine, and of the
    multiples of the first step that lands in the gap, the one closest to
    `edge` is taken.
    """
    magnitude = max(abs(lo), abs(hi))
    k = int(np.floor(np.log10(magnitude))) + 1
    # Once the step is finer than the gap, a multiple lands in it; only float
    # round-off can get past that, and then the edge stays as it was
    while 10.0 ** (k - 1) >= (hi - lo) / 100:
        for mantissa, exponent in ((1, k), (5, k - 1), (2, k - 1)):
            step = mantissa * 10.0 ** exponent
            ms = np.arange(np.floor(lo / step), np.floor(hi / step) + 2)
            candidates = np.round(ms * step, max(0, -exponent))
            candidates = candidates[(candidates > lo) & (candidates <= hi)]
            if len(candidates):
                return float(candidates[np.argmin(np.abs(candidates - edge))])
        k -= 1
    return float(edge)


def _edge_precision(edges: np.ndarray, value_range: float, tol: float = 0.1) -> int:
    """The decimal places to write a variable's bin edges with.

    Enough to write every edge exactly, but no more than it takes to keep the
    rounding error within `tol` of the narrowest bin, measured between the
    inner edges (or across the range of the values, when there is only one).
    """
    widths = np.diff(edges) if len(edges) > 1 else np.array([value_range])
    widths = widths[widths > 0]
    if len(widths) == 0:
        return max(_decimals(e) for e in edges)
    width_precision = max(0, int(np.ceil(np.log10(0.5 / (tol * widths.min())))))
    return min(max(_decimals(e) for e in edges), width_precision)


def __discretize_nonconst(
    variable: ContinuousVariable, n_bins: int, random_state: Any = 0
) -> list[IndicatorVariable]:
    """Discretize non-constant continuous variables.

    Bins are half-open, `lower <= x < upper`. k-means places the edges, and
    each inner edge is then moved to the roundest number that splits the
    samples the same way, so the labels read `x < 4` rather than `x < 4.52`.
    """
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.exceptions import ConvergenceWarning

    col_vals = pd.Series(variable.values)
    col_vals_non_nan = col_vals.dropna()
    x = col_vals_non_nan.values

    n_bins = np.minimum(n_bins, col_vals_non_nan.nunique())

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        strategy="kmeans",
        encode="ordinal",
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        discretizer.fit(x[:, None])

    inner_edges = np.array([
        _round_edge(edge, lo=x[x < edge].max(), hi=x[x >= edge].min())
        for edge in discretizer.bin_edges_[0][1:-1]
    ])
    precision = _edge_precision(inner_edges, value_range=x.max() - x.min())

    # Bins are assigned from the rounded edges, which are what the rules
    # state. Only the measured values are discretized. A sample with no
    # measurement falls in no bin, and is re-inserted as missing in every one
    bin_idx = np.searchsorted(inner_edges, x, side="right")
    x_discretized = (bin_idx[:, None] == np.arange(len(inner_edges) + 1)).astype(float)
    df_discretized = pd.DataFrame(x_discretized, index=col_vals_non_nan.index)
    df_discretized = df_discretized.reindex(col_vals.index)

    # Ensure open intervals
    bin_edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])

    derived_vars = []
    for idx, (lower, upper) in enumerate(zip(bin_edges, bin_edges[1:])):
        rule = IntervalRule(lower, upper, value_name=variable.name, precision=precision)
        values = df_discretized.loc[:, idx].values
        v = IndicatorVariable(variable, rule, values)
        derived_vars.append(v)

    return derived_vars


def discretize(
    variable: ContinuousVariable, n_bins: int = 5, random_state: Any = 0
) -> list[IndicatorVariable]:
    """Discretize a continuous variable."""
    if not isinstance(variable, ContinuousVariable):
        raise TypeError("Can only discretize continuous variables!")

    # Bins are derived from the measured values, and a NaN is not one of them:
    # counted as a value of its own it makes a constant variable look like it
    # takes two
    measured_values = variable.values[~np.isnan(variable.values)]

    if len(measured_values) == 0:
        # Nothing to describe, and no value to name a rule after
        return []
    elif len(np.unique(measured_values)) == 1:
        disc_vars = __discretize_const(variable)
    else:
        disc_vars = __discretize_nonconst(variable, n_bins, random_state=random_state)

    return disc_vars


def one_hot(variable: DiscreteVariable) -> list[IndicatorVariable]:
    """One-hot endcode a discrete variable."""
    if not isinstance(variable, DiscreteVariable):
        raise TypeError("Can only one-hot-encode discrete variables!")

    # A sample with no category belongs to none of them, and says so in every
    # one of them rather than reading as a sample outside each category
    missing = np.isnan(variable.values)

    one_hot_vars = []
    for idx, category in enumerate(variable.categories):
        rule = EqualityRule(category, value_name=variable.name)
        values = np.astype(variable.values == idx, float)
        values[missing] = np.nan
        new_var = IndicatorVariable(variable, rule, values)
        one_hot_vars.append(new_var)

    return one_hot_vars


def expand(
    variables: list[Variable],
    n_discretization_bins: int = 5,
    random_state: Any = 0,
) -> list[list[IndicatorVariable]]:
    """Expand a list of variables into indicator variables via discretization or
    one-hot encoding."""
    var_groups = []
    for variable in variables:
        if variable.is_continuous:
            expanded_vars = discretize(
                variable, n_bins=n_discretization_bins, random_state=random_state
            )
        elif variable.is_discrete:
            expanded_vars = one_hot(variable)
        elif variable.is_indicator:
            expanded_vars = [variable]
        else:
            raise RuntimeError(f"Variable type not recognized, got `{variable}`")

        var_groups.append(expanded_vars)

    # Filter out columns with zero occurences. This can happen for categorical
    # variables with categories that never actually occur in the data
    var_groups = [
        [v for v in var_group if np.nansum(v.values) > 0] for var_group in var_groups
    ]
    # If the filtering removed all the variables from a particular variable,
    # remove that group. In practice, this should never happen.
    var_groups = [var_group for var_group in var_groups if len(var_group) > 0]

    return var_groups


def expand_df(
    df: pd.DataFrame,
    n_discretization_bins: int = 5,
    filter_constant_features: bool = True,
    indicator_columns: IndicatorColumns = None,
    random_state: Any = 0,
) -> list[list[IndicatorVariable]]:
    # The selection is validated against the full frame: a constant column is
    # dropped, not reported as missing
    labels = _resolve_indicator_columns(df, indicator_columns)

    # Filter out features with identical values
    if filter_constant_features:
        df = df.loc[:, df.nunique(axis=0) > 1]
        labels = {c: l for c, l in labels.items() if c in df.columns}

    variables = ingest(df, indicator_columns=labels)

    expanded = expand(
        variables,
        n_discretization_bins=n_discretization_bins,
        random_state=random_state,
    )

    return expanded


def extract_region_annotations(
    variables: list[list[IndicatorVariable]],
    embedding: Embedding | np.ndarray,
    scale_factor: float = 1,
    region_method: str = "kde",
    # KDE parameters
    kernel: str = "gaussian",
    contour_level: float = 0.25,
) -> list[RegionAnnotation]:
    """Extract region annotations for each indicator variable.

    Parameters
    ----------
    variables : list of lists of IndicatorVariable
    embedding : Embedding or np.ndarray
    scale_factor : float
        Controls both the KDE bandwidth and the rangeset edge-pruning
        threshold (via ``embedding.scale``).
    region_method : ``"kde"`` or ``"rangeset"``
        ``"kde"`` extracts regions via kernel density estimation and contour
        thresholding.  ``"rangeset"`` extracts regions via Delaunay
        triangulation with edge-length pruning (Sohns et al., 2021).
    kernel : str
        KDE kernel (only used when *method* is ``"kde"``).
    contour_level : float
        Density contour level (only used when *method* is ``"kde"``).
    """

    # Create embedding instance which will be shared across all explanatory
    # variables. The shared instance is necessary to avoid slow recomputation of
    # adjacency matrices
    if not isinstance(embedding, Embedding):
        embedding = Embedding(embedding, scale_factor=scale_factor)

    if region_method == "kde":
        def _generate_single(v):
            density = embedding.estimate_density(v.values, kernel=kernel)
            region = Region.from_density(
                embedding=embedding, density=density, level=contour_level
            )
            return RegionAnnotation(region, v)
    elif region_method == "rangeset":
        def _generate_single(v):
            region = Region.from_triangulation(
                embedding=embedding,
                member_mask=v.values,
            )
            return RegionAnnotation(region, v)
    else:
        raise ValueError(
            f"Unknown region_method '{region_method}'. "
            f"Supported methods are 'kde' and 'rangeset'."
        )

    # Create explanatory variables from each of the derived features
    region_annotations = []
    num_regions_to_estimate = sum(map(len, variables))
    with tqdm(total=num_regions_to_estimate) as pbar:
        for var_group in variables:
            ra_group = []
            for v in var_group:
                ra = _generate_single(v)
                # Drop regions with empty polygons (e.g. a variable without
                # members, or one whose samples all carry zero weight)
                if not ra.region.polygon.is_empty:
                    ra_group.append(ra)
                pbar.update(1)
            if ra_group:
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
