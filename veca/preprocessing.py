import warnings
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import veca.graph as g
import veca.metrics as metrics
import veca.variables
from veca.embedding import Embedding
from veca.region import Region
from veca.rules import IntervalRule, EqualityRule
from veca.variables import (
    DerivedVariable,
    ExplanatoryVariable,
    DiscreteVariable,
    ContinuousVariable,
    Variable,
)


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
        variable = DiscreteVariable(
            col_name,
            values=col_vals[1].values,
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
    new_index = map(lambda p: _pd_dtype_to_variable(*p), zip(df.columns, df.dtypes, df.items()))
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


def __impute_missing_continous_values(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    df_imputed = imputer.fit_transform(df.values)
    df_imputed = pd.DataFrame(
        df_imputed, columns=df.columns, index=df.index
    )

    return df_imputed


def __discretize_const_continuous_features(df: pd.DataFrame) -> pd.DataFrame:
    # Constant features are converted into discrete equality rules
    derived_features = []
    for variable in df.columns:
        # All values are the same, so we can just take the first one
        uniq_val = df.loc[0, variable]
        rule = EqualityRule(uniq_val, value_name=variable.name)
        v = DerivedVariable(variable, rule)
        derived_features.append(v)

    df_cont_const = pd.DataFrame(
        df.values, columns=derived_features, index=df.index
    )

    # Constant variables are discretized into binary indicator variables
    # indicating membership. Therefore, their values should all be set to 1
    df_cont_const.loc[:, :] = 1

    return df_cont_const


def __discretize_nonconst_continuous_features(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    # Discretize non-constant features
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.exceptions import ConvergenceWarning

    # Ensure that the number of bins is not larger than the number of unique
    # values
    n_bins = np.minimum(n_bins, df.nunique(axis=0).values)

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        strategy="kmeans",
        encode="onehot-dense",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        x_discretized = discretizer.fit_transform(df.values)

    derived_features = []
    for variable, bin_edges in zip(df.columns, discretizer.bin_edges_):
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

    df_cont_nonconst = pd.DataFrame(
        x_discretized, columns=derived_features, index=df.index
    )

    return df_cont_nonconst


def _discretize(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Discretize all continuous variables in the data frame.

    TODO: This function does imputation, but this really shouldn't be done here
    It should be up to the user to ensure there are no NaNs in the data, or up
    to us to ignore them.
    """
    df_ingested = ingest(df)

    # We can only discretize continuous columns
    cont_cols = [c for c in df_ingested.columns if c.is_continuous]
    othr_cols = [c for c in df_ingested.columns if not c.is_continuous]
    df_cont = df_ingested[cont_cols]
    df_cat = df_ingested[othr_cols]

    # If there are no continuous features to be discretized, return
    if not len(df_cont.columns):
        return df_ingested

    # Sklearn discretization doesn't support NaNs, so perform imputation
    df_cont_imputed = __impute_missing_continous_values(df_cont)

    # We can't perform discretization on constant features, so handle constant
    # and non-constant features separately
    nuniq = df_cont_imputed.nunique()

    # Handle constant continuous features
    cont_const_cols = nuniq.index[nuniq == 1].tolist()
    df_cont_const = df_cont_imputed[cont_const_cols]
    if len(cont_const_cols):
        df_cont_const = __discretize_const_continuous_features(df_cont_const)

    # Handle non-constant continuous features
    cont_nonconst_cols = nuniq.index[nuniq > 1].tolist()
    df_cont_nonconst = df_cont_imputed[cont_nonconst_cols]
    if len(cont_nonconst_cols):
        df_cont_nonconst = __discretize_nonconst_continuous_features(
            df_cont_nonconst, n_bins=n_bins,
        )

    return pd.concat([df_cat, df_cont_const, df_cont_nonconst], axis=1)


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

    x_onehot = pd.get_dummies(df_disc).values.astype(float)

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


def expand_to_indicator(
    df: pd.DataFrame, n_discretization_bins: int = 5, filter_constant: bool = True
) -> pd.DataFrame:
    # Filter out features with identical values
    if filter_constant:
        df = df.loc[:, df.nunique(axis=0) > 1]

    df = _one_hot(_discretize(ingest(df), n_bins=n_discretization_bins))

    # Filter out columns with zero occurences
    df = df.loc[:, df.sum(axis=0) > 0]

    return df


def generate_explanatory_features(
    df,
    embedding,
    scale_factor: float = 1,
    kernel: str = "gaussian",
    contour_level: float = 0.25,
):
    # Create embedding instance which will be shared across all explanatory
    # variables. The shared instance is necessary to avoid slow recomputation of
    # adjacency matrices
    if not isinstance(embedding, Embedding):
        embedding = Embedding(embedding, scale_factor=scale_factor)

    # Create explanatory variables from each of the derived features
    explanatory_features = []
    for v in tqdm(df.columns.tolist()):
        values = df[v].values
        density = embedding.estimate_density(values, kernel=kernel)
        region = Region.from_density(density=density, level=contour_level)
        explanatory_v = ExplanatoryVariable(
            v.base_variable,
            v.rule,
            values,
            region,
            embedding,
        )
        explanatory_v.base_variable.register_explanatory_variable(explanatory_v)
        explanatory_features.append(explanatory_v)

    return explanatory_features


def merge_overfragmented(
    variables: list[ExplanatoryVariable],
    min_purity_gain=0.05,
    min_sample_overlap=0.5,
    min_geary_gain=0,
):
    # If we only have a single variable, there is nothing to merge
    if len(variables) == 1:
        return variables

    def _dist(v1, v2):
        if not v1.can_merge_with(v2):
            return 0

        shared_sample_pct = metrics.min_shared_sample_pct(v1, v2)
        if shared_sample_pct < min_sample_overlap:
            return 0

        new_variable = v1.merge_with(v2)
        v1_purity_gain = new_variable.purity / v1.purity - 1
        v2_purity_gain = new_variable.purity / v2.purity - 1
        purity_gain = np.mean([v1_purity_gain, v2_purity_gain])

        v1_geary_gain = (1 - new_variable.gearys_c) / (1 - v1.gearys_c + 1e-16)
        v2_geary_gain = (1 - new_variable.gearys_c) / (1 - v2.gearys_c + 1e-16)
        geary_gain = np.mean([v1_geary_gain, v2_geary_gain])

        return int(
            purity_gain >= min_purity_gain and geary_gain >= min_geary_gain - 1e-4
        )

    def _merge_round(variables):
        variable_groups = defaultdict(list)
        for v in variables:
            variable_groups[v.base_variable].append(v)

        merged_variables = []
        for k, variable_group in tqdm(variable_groups.items()):
            dists = metrics.pdist(variable_group, _dist)
            graph = g.similarities_to_graph(dists, threshold=0.5)
            node_labels = dict(enumerate(variable_group))
            graph = g.label_nodes(graph, node_labels)
            merge_groups = g.connected_components(graph)
            # merge_groups = g.max_cliques(graph)  # TODO: max cliques puts nodes into multiple cliques

            # # Check which subgraphs each node appears in
            # node_occurence = defaultdict(list)
            # for subgraph in merge_groups:
            #     for node in subgraph:
            #         node_occurence[node].append(subgraph)

            for c in merge_groups:
                # Merging all the nodes in the graph into a single node. It is
                # important that the nodes are merged in the correct order, to
                # respect the `.can_merge_with` constraint

                # For some reason, this is automatically sorted so that the
                # variables can be merged immediately

                # TODO: Rules on explanatory variables can be sorted. Can we use
                # that to ensure the correct order?
                var_order = sorted(list(c), key=lambda x: x.rule)
                new_var = veca.variables.CompositeExplanatoryVariable(var_order)

                for node in var_order:
                    node.base_variable.unregister_explanatory_variable(node)
                new_var.base_variable.register_explanatory_variable(new_var)

                merged_variables.append(new_var)

        return merged_variables

    # TODO: This can be sped up by only recomputing variable groups that have
    # changed in the last round. We now recompute all variable groups every time
    prev_len = len(variables)
    while len(variables := _merge_round(variables)) < prev_len:
        prev_len = len(variables)

    return variables
