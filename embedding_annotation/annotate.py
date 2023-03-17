from typing import Any, Callable

import numpy as np
import pandas as pd

from embedding_annotation import graph as g
from embedding_annotation import metrics
from embedding_annotation.embedding import Embedding
from embedding_annotation.metrics import pdist, intersection_percentage
from embedding_annotation.preprocessing import generate_derived_features
from embedding_annotation.region import Density, Region
from embedding_annotation.variables import ExplanatoryVariable, ExplanatoryVariableGroup


def generate_explanatory_features(
    features: pd.DataFrame,
    embedding: np.ndarray,
    n_discretization_bins: int = 5,
    scale_factor: float = 1,
    n_grid_points: int = 100,
    kernel: str = "gaussian",
    contour_level: float = 0.25,
):
    # Convert the data frame so that it contains derived features
    df_derived = generate_derived_features(
        features, n_discretization_bins=n_discretization_bins
    )

    embedding = Embedding(embedding, scale_factor=scale_factor)

    # Create explanatory variables from each of the derived features
    explanatory_features = []
    for v in df_derived.columns.tolist():
        values = df_derived[v].values
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

    # Perform iterative merging

    return explanatory_features


def group_similar_variables(
    variables: list[ExplanatoryVariable],
    metric: Callable = metrics.shared_sample_pct,
    metric_is_distance: bool = False,
    threshold: float = 0.9,
    method: str = "max-cliques",
):
    distances = pdist(variables, metric)
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
    variables: list[ExplanatoryVariable], max_overlap: float = 0.05
) -> list[list[Any]]:
    overlap = pdist(variables, intersection_percentage)
    graph = g.similarities_to_graph(overlap, threshold=max_overlap)
    node_labels = dict(enumerate(variables))
    graph = g.label_nodes(graph, node_labels)

    independent_sets = g.independent_sets(graph)

    return independent_sets
