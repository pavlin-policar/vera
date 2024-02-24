from typing import Callable

import numpy as np
from scipy import stats as stats

from veca import metrics as metrics, graph as g
from veca.explain import _layout_scores
from veca.region import Region
from veca.variables import ExplanatoryVariable, ExplanatoryVariableGroup, Variable


DEFAULT_RANKING_FUNCS = [
    (_layout_scores.variable_occurs_in_all_regions, 30),
    (_layout_scores.mean_variable_occurence, 10),
    (_layout_scores.mean_purity, 5),
    (_layout_scores.sample_coverage, 2),
    (_layout_scores.num_base_vars, 2),
]


def group_similar_variables(
    variables: list[ExplanatoryVariable],
    metric: Callable = metrics.min_shared_sample_pct,
    metric_is_distance: bool = False,
    threshold: float = 0.9,
    method: str = "connected-components",
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
        cliques = list(map(g.nodes, cliques))
        clusts = {f"Cluster {cid}": vs for cid, vs in enumerate(cliques, start=1)}
    elif method == "connected-components":
        connected_components = g.connected_components(graph)
        connected_components = list(map(g.nodes, connected_components))
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


def enrich_var_group_with_background(
    var_group: ExplanatoryVariableGroup,
    background_vars: list[ExplanatoryVariable],
    threshold: float = 0.9,
):
    # Determine which base vars are present in the var group
    contained_base_vars = {v.base_variable for v in var_group.variables}

    to_add = []
    for background_v in background_vars:
        # If the background variable belongs to the same base variable
        # as already present in the data, skip that
        if background_v.base_variable in contained_base_vars:
            continue

        # Determine if the background var encompasses the current variable
        shared_samples = var_group.contained_samples & background_v.contained_samples
        pct_shared = len(shared_samples) / len(var_group.contained_samples)

        # If the region sufficiently overlaps, prepare the overlap for merge
        if pct_shared > threshold:
            new_polygon = background_v.region.polygon.intersection(
                var_group.region.polygon
            )
            cloned_bg_var = ExplanatoryVariable(
                background_v.base_variable,
                background_v.rule,
                background_v.values,
                Region(new_polygon),
                background_v.embedding
            )
            to_add.append(cloned_bg_var)

    # If we found any background variables to add to the var group, create a new
    # var group with all available explanatory vars
    if len(to_add) > 0:
        var_group = ExplanatoryVariableGroup(
            var_group.variables + to_add, name=var_group.name
        )

    return var_group


def enrich_panel_with_background(
    panel: list[ExplanatoryVariableGroup],
    background_vars: list[ExplanatoryVariable],
    threshold: float = 0.9,
):
    return [
        enrich_var_group_with_background(var_group, background_vars, threshold)
        for var_group in panel
    ]


def enrich_layout_with_background(
    layout: list[list[ExplanatoryVariableGroup]],
    background_vars: list[ExplanatoryVariable],
    threshold: float = 0.9,
):
    return [
        enrich_panel_with_background(panel, background_vars, threshold)
        for panel in layout
    ]


def generate_descriptive_layout(
    clusters,
    max_panels: int,
    max_overlap: float = 0.0,
    ranking_funcs=DEFAULT_RANKING_FUNCS,
):
    # Create a working copy of our clusters
    clusters = list(clusters)

    score_fns, score_weights = list(zip(*ranking_funcs))

    layout = []
    for _ in range(max_panels):
        # Generate all non-overlapping layouts
        overlap = metrics.pdist(clusters, metrics.max_shared_sample_pct)
        graph = g.similarities_to_graph(overlap, threshold=max_overlap)
        node_labels = dict(enumerate(clusters))
        graph = g.label_nodes(graph, node_labels)

        layouts = g.independent_sets(graph)

        # Sort the layouts according to our metrics
        scores = np.array([
            [score_fn(layout) for score_fn in score_fns]
            for layout in layouts
        ])
        rankings = stats.rankdata(scores, method="max", axis=0)

        score_weights = np.array(score_weights)
        mean_ranks = np.mean(rankings * score_weights, axis=1)

        # Select the layout with the highest weighted mean rank
        selected_layout = layouts[np.argmax(mean_ranks)]
        layout.append(selected_layout)

        # Remove variables in the current panel from the remaining variables
        for var in selected_layout:
            clusters.remove(var)

        # If we've run out of clusters to add to any new panel, we're done
        if len(clusters) == 0:
            break

    return layout


def descriptive(
    variables: list[Variable],
    max_panels: int = 4,
    merge_metric: Callable = metrics.min_shared_sample_pct,
    metric_is_distance: bool = False,
    merge_method: str = "connected-components",
    merge_threshold: float = 0.8,
    cluster_min_samples: int = 5,
    cluster_min_purity: float = 0.5,
    cluster_max_geary_index: float = 1,  # no geary filtering by default
    max_overlap: float = 0.0,
    enrich_with_background: bool = True,
    return_clusters: bool = False,
    ranking_funcs=DEFAULT_RANKING_FUNCS,
):
    explanatory_variables = [ex for v in variables for ex in v.explanatory_variables]

    # Split explanatory features into their polygons
    explanatory_variables_split = []
    for expl_var in explanatory_variables:
        explanatory_variables_split.extend(expl_var.split_region())

    # Remove any split parts that do not pass the filtering criteria
    explanatory_variables_split = filter_explanatory_features(
        explanatory_variables_split,
        min_samples=cluster_min_samples,
        min_purity=cluster_min_purity,
        max_geary_index=cluster_max_geary_index,
    )

    clusters = group_similar_variables(
        explanatory_variables_split,
        metric=merge_metric,
        metric_is_distance=metric_is_distance,
        threshold=merge_threshold,
        method=merge_method,
    )

    layout = generate_descriptive_layout(
        clusters,
        max_panels=max_panels,
        max_overlap=max_overlap,
        ranking_funcs=ranking_funcs,
    )

    if enrich_with_background:
        layout = enrich_layout_with_background(layout, explanatory_variables)

    if return_clusters:
        return layout, clusters
    else:
        return layout


def filter_explanatory_features(
    variables: list[ExplanatoryVariable],
    min_samples: int = 5,
    min_purity: float = 0.5,
    max_geary_index: float = 1,  # no filtering by Geary
):
    return [
        v
        for v in variables
        if v.purity >= min_purity
        and v.gearys_c <= max_geary_index
        and v.num_contained_samples >= min_samples
    ]
