from collections import defaultdict
from itertools import chain, combinations

import numpy as np
import scipy.stats as stats

from vera import metrics


def mean_variable_occurence(panel):
    """Count the number of times each base variable appears in a layout."""
    base_variable_count = defaultdict(int)
    for variable_group in panel:
        for variable in variable_group.variable_dict:
            base_variable_count[variable.base_variable] += 1

    return np.mean(list(base_variable_count.values()))


def variable_occurs_in_all_regions(panel, all_region_threshold=0.9):
    """Count the number of times each base variable appears in all the var groups of a layout."""
    num_regions = len(panel)

    base_variable_count = defaultdict(int)
    for variable_group in panel:
        for variable in variable_group.variable_dict:
            base_variable_count[variable.base_variable] += 1
    base_variable_freq = {k: v / num_regions for k, v in base_variable_count.items()}
    occurence = np.array(list(base_variable_freq.values()))

    num_occurences = np.sum(occurence >= all_region_threshold)
    # If we only have a single region, this panel isn't really any more informative
    num_occurences *= max(0, (num_regions - 1))

    return num_occurences


def mean_purity(layout):
    return np.mean([v.purity for v in layout])


def sample_coverage(layout):
    """What portion of all samples are covered by the var groups in the current layout"""
    num_all_samples = layout[0].embedding.shape[0]

    covered_samples = [v.contained_samples for v in layout]
    covered_samples = set(chain.from_iterable(covered_samples))
    pct_covered_samples = len(covered_samples) / num_all_samples

    return pct_covered_samples


def num_base_vars(layout):
    """How many different base variables are represented in the layout"""
    return len({v.base_variable for vg in layout for v in vg.variable_dict})


def mean_overlap(panel):
    """Scoring function for overlap, 1 - mean overlap; higher is better."""
    return 1 - np.mean([
        metrics.shared_sample_pct(v1, v2)
        for v1, v2 in combinations(panel, 2)
    ])


def num_regions_matches_perception(panel, target_num_regions=3):
    num_regions = len(panel)
    pdf = stats.norm(loc=target_num_regions, scale=2)
    return pdf.pdf(num_regions)
