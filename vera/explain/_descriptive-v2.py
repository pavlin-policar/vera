import random
from collections import defaultdict
from typing import Callable, Any

import numpy as np
from scipy import stats as stats

from vera import metrics as metrics, graph as g
from vera.variables import ExplanatoryVariable, ExplanatoryVariableGroup, Variable


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


def find_layouts(
    variables: list[ExplanatoryVariable], max_overlap: float = 0.2
) -> list[list[Any]]:
    overlap = metrics.pdist(variables, metrics.max_shared_sample_pct)
    graph = g.similarities_to_graph(overlap, threshold=max_overlap)
    node_labels = dict(enumerate(variables))
    graph = g.label_nodes(graph, node_labels)

    independent_sets = g.independent_sets(graph)

    return independent_sets

# ===
def layouts_of_size(filtered_clusters, size):
    layouts = find_layouts(filtered_clusters, max_overlap=0)
    layouts = [tuple(l) for l in layouts]

    if size <= 1:
        return tuple((l,) for l in layouts)

    result = []

    random.shuffle(layouts)
    for layout in layouts[:20]:
        # Copy available clusters and remove the clusters in the current layout
        remaining_clusters = list(filtered_clusters)
        for cluster in layout:
            remaining_clusters.remove(cluster)

        if len(remaining_clusters) > 0:
            for result_set in layouts_of_size(remaining_clusters, size - 1):
                tmp = (layout, *result_set)
                result.append(tmp)
        else:
            result.append((layout,))
    return tuple(result)


filtered_clusters = filter_explanatory_features(clusters, min_purity=0)
tmp = layouts_of_size(filtered_clusters, 4)
len(tmp), len(set(frozenset(frozenset(ti) for ti in t) for t in tmp))


# ===
purities = []
for layout in tmp:
    panel_purity = []
    for panel in layout:
        for variable in panel:
            panel_purity.append(variable.purity)
    purities.append(np.mean(panel_purity))

np.argmax(purities)

# ===
def mean_variable_occurence(panel):
    """Count the number of times each base variable appears in a layout."""
    base_variable_count = defaultdict(int)
    for variable_group in panel:
        for variable in variable_group.variables:
            base_variable_count[variable.base_variable] += 1

    return np.mean(list(base_variable_count.values()))

mvos = []
for layout in tmp:
    mvos_ = []
    for panel in layout:
        mvo = mean_variable_occurence(panel)
        mvos_.append(mvo / len(panel))
    mvos.append(np.mean(mvos_))

mvos = np.array(mvos)
np.argmax(mvos)

# ===
layout_balance = []
for layout in tmp:
    balances = []
    for panel in layout:
        balances.append(len(panel))
    layout_balance.append(np.std(np.array(balances)))

np.argmin(layout_balance)

# ===
idx = np.argmax(np.array(purities) * -np.array(layout_balance) * mvos)
idx = np.argmax(mvos)
idx

# ===
layout = list(tmp)[idx]
var_colors = vera.pl.layout_variable_colors(layout)
vera.pl.plot_annotations(layout, indicate_purity=False, variable_colors=var_colors, return_ax=True)
