from functools import reduce
from itertools import combinations
from typing import List

import numpy as np
from scipy import stats as stats

from veca import metrics as metrics, graph as g
from veca.variables import VariableGroup, ExplanatoryVariableGroup, Variable


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
    max_panels: int = 4,
    merge_threshold: float = 0.95,
    filter_layouts: bool = True,
):
    variables = [VariableGroup([v], v.explanatory_variables) for v in variables]

    # See if we can merge different variables with almost perfectly overlap
    variables = merge_contrastive(variables, threshold=merge_threshold)

    # Compute metrics for each variable group
    mean_overlap, mean_purity, num_vars, num_polygons = {}, {}, {}, {}
    layout_scores = {}
    for v in variables:
        # Compute the mean overlap between pairs of variables in the group
        overlaps = [
            metrics.shared_sample_pct(v1, v2)
            for v1, v2 in combinations(v.explanatory_variables, 2)
        ]
        mean_overlap[v] = np.mean(overlaps)

        purities = [vi.purity for vi in v.explanatory_variables]
        mean_purity[v] = np.mean(purities)

        num_vars[v] = len(v.explanatory_variables)
        num_polygons[v] = sum(v.region.num_parts for v in v.explanatory_variables)

        # Ideally, we want about three variables
        pdf = stats.norm(loc=3, scale=2)

        layout_scores[v] = (
            np.log(np.mean(purities))
            + np.log(1 - mean_overlap[v])
            + 0.1 * pdf.logpdf(num_vars[v])
        )

    # Scores for variables with a single explanatory variable are 0
    for v in variables:
        if len(v.explanatory_variables) == 1:
            layout_scores[v] = 0

    sorted_keys = reversed(sorted(variables, key=layout_scores.get))
    sorted_groups = [k.explanatory_variables for k in sorted_keys]

    if filter_layouts:
        sorted_groups = [g for g in sorted_groups if len(g) > 1]

    if max_panels is not None:
        sorted_groups = sorted_groups[:max_panels]

    return sorted_groups
