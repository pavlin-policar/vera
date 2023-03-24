from collections import defaultdict
from itertools import combinations

import numpy as np

from veca import metrics
from veca.variables import ExplanatoryVariable


def contrastive(variables: list[ExplanatoryVariable]):
    # Determine variable groups
    variable_groups = defaultdict(list)
    for v in variables:
        variable_groups[v.base_variable].append(v)

    # Sort each individual subgroup

    # Score each layout, if the layout contains a single variable, give it a score of zero

    # Compute metrics for each variable group
    min_overlap, mean_purity, num_vars, num_polygons = {}, {}, {}, {}
    layout_scores = {}
    for v, group in variable_groups.items():
        overlaps = [metrics.shared_sample_pct(v1, v2) for v1, v2 in combinations(group, 2)]
        min_overlap[v] = np.min(overlaps, initial=0)

        purities = [vi.purity for vi in group]
        mean_purity[v] = np.mean(purities)

        num_vars[v] = len(group)
        num_polygons[v] = sum(v.region.num_parts for v in group)

        # polygon_ratio = num_vars[v] / num_polygons[v]  # lower is worse, max=1
        # Polygon doesn't work well
        # TODO: Perhaps it would be better to rank these by overlap area?
        layout_scores[v] = np.mean(purities) * (1 - min_overlap[v])

    sorted_keys = reversed(sorted(variable_groups, key=layout_scores.get))
    sorted_groups = [variable_groups[k] for k in sorted_keys]
    return sorted_groups


def discovery(variables: list[ExplanatoryVariable]):
    ...

