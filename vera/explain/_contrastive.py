from functools import reduce
from itertools import combinations
from typing import List

import numpy as np
from scipy import stats as stats

from vera import metrics as metrics, graph as g
from vera.explain import _layout_scores
from vera.variables import VariableGroup, Variable
from vera.region_annotations import RegionAnnotationGroup

DEFAULT_RANKING_FUNCS = [
    (_layout_scores.mean_overlap, 5),
    (_layout_scores.mean_purity, 5),
    (_layout_scores.num_regions_matches_perception, 1),
]


def merge_contrastive(variables: List[VariableGroup], threshold: float = 0.95):
    explanatory_variables = {v: v.explanatory_variables for v in variables}

    merge_candidates = {}

    for v1, v2 in combinations(explanatory_variables, 2):
        ex1, ex2 = explanatory_variables[v1], explanatory_variables[v2]

        # If the number of explantory variables does not match, we can't merge
        if len(ex1) != len(ex2):
            continue

        # if isinstace(v, vera.variables.ExplanatoryVariable):
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
            RegionAnnotationGroup(cc_part) for cc_part in map(g.nodes, cc_parts)
        ]
        merged_variables.append(VariableGroup(var_groups_to_merge, merged_expl_vars))

    return merged_variables


def contrastive(
    variables: list[Variable],
    max_panels: int = 4,
    merge_threshold: float = 0.95,
    filter_layouts: bool = True,
    ranking_funcs=DEFAULT_RANKING_FUNCS,
):
    variables = [VariableGroup([v]) for v in variables]

    # See if we can merge different variables with almost perfectly overlap
    variables = merge_contrastive(variables, threshold=merge_threshold)

    # Construct candidate panels
    candidate_panels = [v.explanatory_variables for v in variables]

    if filter_layouts:
        candidate_panels = [panel for panel in candidate_panels if len(panel) > 1]

    # Compute scores for each candidate panel
    score_fns, score_weights = list(zip(*ranking_funcs))

    panel_scores = np.array([
        [score_fn(layout) for score_fn in score_fns]
        for layout in candidate_panels
    ])
    rankings = stats.rankdata(panel_scores, method="max", axis=0)

    score_weights = np.array(score_weights)
    mean_ranks = np.mean(rankings * score_weights, axis=1)

    # Panels corresponding to a variable with a single explanatory variable
    # aren't informative, so give them the lowest ranking
    single_expl_var_idx = [
        i for i in range(len(candidate_panels)) if len(candidate_panels) == 1
    ]
    mean_ranks[single_expl_var_idx] = -1

    # Select the layouts with the highest weighted mean rank
    selected_idx = np.argsort(mean_ranks)[::-1]
    if max_panels is not None:
        selected_idx = selected_idx[:max_panels]

    layout = [candidate_panels[i] for i in selected_idx]

    return layout
