from functools import reduce
from itertools import combinations

import numpy as np
from scipy import stats as stats

from vera import metrics as metrics, graph as g
from vera.explain import _layout_scores
from vera.utils import group_by_base_var, flatten
from vera.variables import VariableGroup
from vera.region_annotations import RegionAnnotationGroup, RegionAnnotation

DEFAULT_RANKING_FUNCS = [
    (_layout_scores.mean_overlap, 5),
    (_layout_scores.mean_purity, 5),
    (_layout_scores.num_regions_matches_perception, 1),
]


def merge_contrastive(region_annotations: list[list[RegionAnnotation]], threshold: float = 0.95):
    merge_candidates = {}

    for ra_group1, ra_group2 in combinations(region_annotations, 2):
        # If the number of explantory variables does not match, we can't merge
        if len(ra_group1) != len(ra_group2):
            continue

        # See if we can find a bipartite matching
        merged_ra = ra_group1 + ra_group2
        merged_ra_mapping = dict(enumerate(merged_ra))
        distances = metrics.pdist(merged_ra, metrics.shared_sample_pct)
        graph = g.similarities_to_graph(distances, threshold=threshold)
        graph = g.label_nodes(graph, merged_ra_mapping)

        # The number of connected components should be the same as the
        # number of explanatory variables
        connected_components = g.connected_components(graph)
        if len(connected_components) != len(ra_group1):
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

    graph = g.edgelist_to_graph(region_annotations, list(merge_candidates))
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
    region_annotations: list[list[RegionAnnotation]],
    max_panels: int = 4,
    merge_threshold: float = 0.95,
    filter_layouts: bool = True,
    ranking_funcs=DEFAULT_RANKING_FUNCS,
):
    # Although the region annotations may already be grouped by base variable,
    # don't trust the user with this
    region_annotations = group_by_base_var(flatten(region_annotations))

    # See if we can merge different variables with almost perfectly overlap
    # region_annotations = merge_contrastive(region_annotations, threshold=merge_threshold)

    # Construct candidate panels
    candidate_panels = region_annotations

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
