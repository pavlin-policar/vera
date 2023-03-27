from itertools import combinations
from typing import List

import numpy as np

import veca.graph as g
import veca.metrics as metrics
from veca.variables import Variable, ExplanatoryVariableGroup, VariableGroup


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

        # The number of connected components should be the same as the
        # number of explanatory variables
        connected_components = g.connected_components(graph)
        if len(connected_components) != len(ex1):
            continue

        # Each connected component should contain both base variables to be merged
        connected_components = list(map(g.nodes, connected_components))
        connected_components = [
            [expl_var_mapping[ci] for ci in c] for c in connected_components
        ]

        all_components_have_two = True
        for c in connected_components:
            all_base_vars = set(v.base_variable for v in c)
            if len(all_base_vars) != 2:
                all_components_have_two = False
        if not all_components_have_two:
            continue

        new_explanatory_variables = []
        for c in connected_components:
            assert len(c) == 2
            new_explanatory_variables.append(ExplanatoryVariableGroup([c[0], c[1]]))

        merge_candidates[frozenset([v1, v2])] = new_explanatory_variables

    graph = g.edgelist_to_graph(variables, merge_candidates)
    graph = g.to_undirected(graph)
    connected_components = g.connected_components(graph)

    merged_variables = []
    for c in connected_components:
        curr_node = next(iter(c))
        while len(c[curr_node]):
            next_node = next(iter(c[curr_node]))

            # perform merging logic
            new_node = VariableGroup(
                curr_node.variables | next_node.variables,
                merge_candidates[frozenset([curr_node, next_node])],
            )

            c = g.merge_nodes(c, curr_node, next_node, new_node)
            curr_node = next(iter(c))
        assert len(c) == 1
        merged_variables.append(curr_node)

    return merged_variables


def contrastive(variables: list[Variable]):
    variables = [VariableGroup([v], v.explanatory_variables) for v in variables]

    # See if we can merge diffent variables with almost perfectly overlap
    variables = merge_contrastive(variables)

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
    return sorted_groups


def discovery(variables: list[Variable]):
    ...
