from collections import defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

import veca.graph as g
import veca.preprocessing as pp
from veca import metrics
from veca.variables import ExplanatoryVariable, ExplanatoryVariableGroup


def generate_explanatory_features(
    features: pd.DataFrame,
    embedding: np.ndarray,
    sample_size: int = 5000,
    n_discretization_bins: int = 5,
    scale_factor: float = 1,
    n_grid_points: int = 100,
    kernel: str = "gaussian",
    contour_level: float = 0.25,
    merge_min_purity_gain=0.5,
    merge_min_sample_overlap=0.5,
    random_state: Any = None,
    filter_trivial_groups: bool = True,
    return_unmerged_features: bool = False,
):
    # Sample the data if necessary. Running on large data sets can be very slow
    random_state = check_random_state(random_state)
    if sample_size is not None:
        num_samples = min(sample_size, features.shape[0])
        sample_idx = random_state.choice(
            features.shape[0], size=num_samples, replace=False
        )
        features = features.iloc[sample_idx]
        embedding = embedding[sample_idx]

    # Convert the data frame so that it contains derived features
    df_derived = pp.generate_derived_features(
        features, n_discretization_bins=n_discretization_bins
    )

    # Create explanatory variables from each of the derived features
    explanatory_features = pp.convert_derived_features_to_explanatory(
        df_derived,
        embedding,
        scale_factor=scale_factor,
        n_grid_points=n_grid_points,
        kernel=kernel,
        contour_level=contour_level,
    )

    # Perform iterative merging
    merged_explanatory_features = pp.merge_overfragmented(
        explanatory_features,
        min_purity_gain=merge_min_purity_gain,
        min_sample_overlap=merge_min_sample_overlap,
    )

    # Filter out groups that now only have a single explanatory variable
    if filter_trivial_groups:
        variable_groups = defaultdict(list)
        for v in merged_explanatory_features:
            variable_groups[v.base_variable].append(v)
        merged_explanatory_features = [
            v
            for v in merged_explanatory_features
            if len(variable_groups[v.base_variable]) > 1
        ]

    if return_unmerged_features:
        return merged_explanatory_features, explanatory_features
    else:
        return merged_explanatory_features


def filter_explanatory_features(
    features: list[ExplanatoryVariable],
    min_samples: int = 5,
    min_purity: float = 0.5,
    max_geary_index: float = 0.5,
):
    return [
        f
        for f in features
        if f.purity >= min_purity
        and f.gearys_c <= max_geary_index
        and f.num_all_samples >= min_samples
    ]


def group_similar_variables(
    variables: list[ExplanatoryVariable],
    metric: Callable = metrics.shared_sample_pct,
    metric_is_distance: bool = False,
    threshold: float = 0.9,
    method: str = "max-cliques",
):
    distances = metrics.pdist(variables, metric)
    g_func = [g.similarities_to_graph, g.distances_to_graph][metric_is_distance]
    graph = g_func(distances, threshold=threshold)

    node_labels = dict(enumerate(variables))
    graph = g.label_nodes(graph, node_labels)

    # Once we construct the graph, find the max-cliques. These will serve as our
    # merged "clusters"
    if method == "max-cliques":
        cliques = g.max_cliques(graph)
        clusts = {f"Cluster {cid}": vs for cid, vs in enumerate(cliques, start=1)}
    elif method == "connected-components":
        connected_components = g.connected_components(graph)
        clusts = {
            f"Cluster {cid}": list(c)
            for cid, c in enumerate(connected_components, start=1)
        }
    else:
        raise ValueError(
            f"Unrecognized method `{method}`. Can be one of `max-cliques`, "
            f"`connected-components`"
        )

    variable_groups = [
        ExplanatoryVariableGroup(variables=clust_vars, name=cid)
        for cid, clust_vars in clusts.items()
    ]

    return variable_groups


def find_layouts(
    variables: list[ExplanatoryVariable], max_overlap: float = 0.2
) -> list[list[Any]]:
    overlap = metrics.pdist(variables, metrics.max_shared_sample_pct)
    graph = g.similarities_to_graph(overlap, threshold=max_overlap)
    node_labels = dict(enumerate(variables))
    graph = g.label_nodes(graph, node_labels)

    independent_sets = g.independent_sets(graph)

    return independent_sets
