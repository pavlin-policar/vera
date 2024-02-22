from collections import defaultdict
from typing import Callable, Any

import numpy as np
from scipy import stats as stats

from veca import metrics as metrics, graph as g
from veca.variables import ExplanatoryVariable, ExplanatoryVariableGroup, Variable


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


def find_layouts(
    variables: list[ExplanatoryVariable], max_overlap: float = 0.2
) -> list[list[Any]]:
    overlap = metrics.pdist(variables, metrics.max_shared_sample_pct)
    graph = g.similarities_to_graph(overlap, threshold=max_overlap)
    node_labels = dict(enumerate(variables))
    graph = g.label_nodes(graph, node_labels)

    independent_sets = g.independent_sets(graph)

    return independent_sets


def rank_descriptive_layouts(layouts):
    layouts = list(map(tuple, layouts))

    # Determine the number of polygons each variable has
    num_polygons = defaultdict(int)
    num_variables = defaultdict(int)
    for layout in layouts:
        for v in layout:
            num_polygons[layout] += v.region.num_parts
            num_variables[layout] += 1

    polygon_ratio = {
        l: num_polygons[l] / num_variables[l] for l in layouts
    }  # lower is better, min=1

    # Determine the overlap area: shared sample pct, smaller is better
    overlap_area = {
        l: np.max(metrics.pdist(l, metrics.shared_sample_pct), initial=0.01) for l in layouts
    }

    # Determine how many of the data points are covered by the polygons
    n_data_points = layouts[0][0].embedding.shape[0]
    coverage = {}
    for l in layouts:
        contained_samples = set()
        for v in l:
            contained_samples.update(v.contained_samples)
        # TODO: This here isn't ideal, because the contained samples also include
        #  samples, which are incorrectly classified
        coverage[l] = 1 - len(contained_samples) / n_data_points

    # Ideally, we want about three variables
    pdf = stats.norm(loc=3, scale=2)

    layout_scores = {
        l: 0.25 * np.log(polygon_ratio[l])
        + 3 * -pdf.logpdf(num_variables[l])
        + 2 * np.log(overlap_area[l] + 0.0001)
        + 1 * np.log(coverage[l] + 0.0001)
        for l in layouts
    }

    candidate_layouts = list(layouts)
    sorted_layouts = []

    # We want to penalize subsequent uses of the same variable
    variable_penalties = defaultdict(int)
    while len(candidate_layouts):
        layout_penalties = {
            l: np.prod([variable_penalties[v] + 1 for v in l])
            for l in candidate_layouts
        }
        final_layout_scores = [
            layout_scores[l] + 1 * layout_penalties[l] for l in candidate_layouts
        ]

        next_layout = candidate_layouts[np.argmin(final_layout_scores)]
        candidate_layouts.remove(next_layout)
        sorted_layouts.append(next_layout)

        # Apply penalty
        for v in next_layout:
            variable_penalties[v] += 10
        for v in variable_penalties:
            variable_penalties[v] /= 1.5

    return list(map(list, sorted_layouts))


def descriptive(
    variables: list[Variable],
    merge_metric: Callable = metrics.min_shared_sample_pct,
    metric_is_distance: bool = False,
    merge_method: str = "max-cliques",
    merge_threshold: float = 0.75,
    cluster_min_samples: int = 5,
    cluster_min_purity: float = 0.5,
    cluster_max_geary_index: float = 1,  # no geary filtering by default
    layout_max_overlap: float = 0.2,
    filter_layouts: bool = True,
    return_clusters: bool = False,
):
    # TODO: When constructing descriptive layouts, it may be useful to split the
    # regions into the different polygon parts. We don't need to match the entire
    # region of a variable value to describe what a particular cluster corresponds
    # to
    explanatory_variables = [ex for v in variables for ex in v.explanatory_variables]

    clusters = group_similar_variables(
        explanatory_variables,
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

    layouts = rank_descriptive_layouts(layouts)

    if filter_layouts:
        layouts = [l for l in layouts if len(l) > 1]

    if return_clusters:
        return layouts, filtered_clusters
    else:
        return layouts


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
