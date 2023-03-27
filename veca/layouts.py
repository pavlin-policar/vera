from collections import defaultdict
from functools import reduce
from itertools import combinations
from typing import List, Callable, Any

import numpy as np

import veca.graph as g
import veca.metrics as metrics
from veca.variables import (
    Variable,
    ExplanatoryVariableGroup,
    VariableGroup,
    ExplanatoryVariable,
)


def merge_contrastive(variables: List[VariableGroup], threshold: float = 0.95):
    explanatory_variables = {v: v.explanatory_variables for v in variables}

    merge_candidates = {}

    for v1, v2 in combinations(explanatory_variables, 2):
        ex1, ex2 = explanatory_variables[v1], explanatory_variables[v2]

        # If the number of explantory variables does not match, we can't merge
        if len(ex1) != len(ex2):
            continue

        # if isinstace(v, veca.variables.ExplanatoryVariable):
        base_vars_1 = {v.base_variable for v in ex1}
        base_vars_2 = {v.base_variable for v in ex2}
        assert len(base_vars_1) == 1
        assert len(base_vars_2) == 1

        # See if we can find a bipartite matching
        expl_vars = ex1 + ex2
        expl_var_mapping = dict(enumerate(expl_vars))
        distances = metrics.pdist(expl_vars, metrics.shared_sample_pct)
        graph = g.similarities_to_graph(distances, threshold=threshold)
        graph = g.label_nodes(graph, expl_var_mapping)

        # The number of connected components should be the same as the
        # number of explanatory variables
        connected_components = g.connected_components(graph)
        if len(connected_components) != len(ex1):
            continue

        # Each connected component should contain both base variables to be merged
        connected_components = list(map(g.nodes, connected_components))

        all_components_have_two = True
        for c in connected_components:
            all_base_vars = set(v.base_variable for v in c)
            if len(all_base_vars) != 2:
                all_components_have_two = False
        if not all_components_have_two:
            continue

        merge_candidates[frozenset([v1, v2])] = connected_components

    graph = g.edgelist_to_graph(variables, list(merge_candidates))
    graph = g.to_undirected(graph)
    connected_components = g.connected_components(graph)

    merged_variables = []
    for c in connected_components:
        var_groups_to_merge = g.nodes(c)

        if len(var_groups_to_merge) == 1:
            merged_variables.append(var_groups_to_merge[0])
            continue

        # Which entries of the merge_candidates contain info on the bipartite mapping
        merge_candidate_keys = [
            p for p in list(merge_candidates) if len(p & set(var_groups_to_merge)) == 2
        ]
        edges = [e for k in merge_candidate_keys for e in merge_candidates[k]]

        all_nodes = reduce(lambda acc, x: set(x) | acc, edges, set())
        graph = g.edgelist_to_graph(all_nodes, edges)
        graph = g.to_undirected(graph)
        cc_parts = g.connected_components(graph)

        merged_expl_vars = [
            ExplanatoryVariableGroup(cc_part) for cc_part in map(g.nodes, cc_parts)
        ]
        merged_variables.append(VariableGroup(var_groups_to_merge, merged_expl_vars))

    return merged_variables


def contrastive(
    variables: list[Variable],
    merge_threshold: float = 0.95,
    filter_layouts: bool = True,
):
    variables = [VariableGroup([v], v.explanatory_variables) for v in variables]

    # See if we can merge diffent variables with almost perfectly overlap
    variables = merge_contrastive(variables, threshold=merge_threshold)

    # Compute metrics for each variable group
    min_overlap, mean_purity, num_vars, num_polygons = {}, {}, {}, {}
    layout_scores = {}
    for v in variables:
        overlaps = [
            metrics.shared_sample_pct(v1, v2)
            for v1, v2 in combinations(v.explanatory_variables, 2)
        ]
        # Select the minimum overlap between any two variables in the group
        min_overlap[v] = np.min(overlaps, initial=0)

        purities = [vi.purity for vi in v.explanatory_variables]
        mean_purity[v] = np.mean(purities)

        num_vars[v] = len(v.explanatory_variables)
        num_polygons[v] = sum(v.region.num_parts for v in v.explanatory_variables)

        # polygon_ratio = num_vars[v] / num_polygons[v]  # lower is worse, max=1
        # Polygon doesn't work well
        # TODO: Perhaps it would be better to rank these by overlap area?
        layout_scores[v] = np.mean(purities) * (1 - min_overlap[v])

    # Socres for variables with a single explanatory variable are 0
    for v in variables:
        if len(v.explanatory_variables) == 1:
            layout_scores[v] = 0

    sorted_keys = reversed(sorted(variables, key=layout_scores.get))
    sorted_groups = [k.explanatory_variables for k in sorted_keys]

    if filter_layouts:
        sorted_groups = [g for g in sorted_groups if len(g) > 1]

    return sorted_groups


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


def filter_explanatory_features(
    variables: list[ExplanatoryVariable],
    min_samples: int = 5,
    min_purity: float = 0.5,
    max_geary_index: float = 0.5,
):
    return [
        v
        for v in variables
        if v.purity >= min_purity
        and v.gearys_c <= max_geary_index
        and v.num_all_samples >= min_samples
    ]


def find_layouts(
    variables: list[ExplanatoryVariable], max_overlap: float = 0.2
) -> list[list[Any]]:
    overlap = metrics.pdist(variables, metrics.max_shared_sample_pct)
    graph = g.similarities_to_graph(overlap, threshold=max_overlap)
    node_labels = dict(enumerate(variables))
    graph = g.label_nodes(graph, node_labels)

    independent_sets = g.independent_sets(graph)

    return independent_sets


def rank_discovery_layouts(layouts):
    layout_idx = np.arange(len(layouts))

    # Determine the number of polygons each variable has
    num_polygons = np.zeros_like(layout_idx, dtype=int)
    num_variables = np.zeros_like(layout_idx, dtype=int)
    for idx, layout in enumerate(layouts):
        for var in layout:
            num_polygons[idx] += var.region.num_parts
            num_variables[idx] += 1

    polygon_ratio = num_polygons / num_variables  # lower is better, min=1

    # Determine the overlap area
    overlap_area = np.array([
        np.mean(metrics.pdist(layout, metrics.intersection_area))
        for layout in layouts
    ])

    score = polygon_ratio * overlap_area

    sort_idx = np.argsort(score)

    return [layouts[idx] for idx in sort_idx]


def discovery(
    variables: list[Variable],
    merge_metric: Callable = metrics.min_shared_sample_pct,
    metric_is_distance: bool = False,
    merge_method: str = "max-cliques",
    merge_threshold: float = 0.75,
    cluster_min_samples: int = 5,
    cluster_min_purity: float = 0.5,
    cluster_max_geary_index: float = 0.5,
    layout_max_overlap: float = 0.2,
    filter_layouts: bool = True,
    return_clusters: bool = False,
):
    explanatory_features = [ex for v in variables for ex in v.explanatory_variables]

    clusters = group_similar_variables(
        explanatory_features,
        metric=merge_metric,
        metric_is_distance=metric_is_distance,
        threshold=merge_threshold,
        method=merge_method,
    )

    filtered_clusters = filter_explanatory_features(
        clusters,
        min_samples=cluster_min_samples,
        min_purity=cluster_min_purity,
        max_geary_index=cluster_max_geary_index,
    )

    layouts = find_layouts(filtered_clusters, max_overlap=layout_max_overlap)

    layouts = rank_discovery_layouts(layouts)

    if filter_layouts:
        layouts = [l for l in layouts if len(l) > 1]

    if return_clusters:
        return layouts, filtered_clusters
    else:
        return layouts
