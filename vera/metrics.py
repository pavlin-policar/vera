from typing import Any, Callable

import numpy as np

from vera.region_annotation import RegionAnnotation


def purity(ra: RegionAnnotation) -> float:
    contained_vals = ra.descriptor.values[list(ra.region.contained_samples)]
    if len(contained_vals) == 0:
        return 0
    return np.mean(contained_vals)


def pdist(l: list[Any], metric: Callable):
    n = len(l)
    out_size = (n * (n - 1)) // 2
    result = np.zeros(out_size, dtype=np.float64)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            result[k] = metric(l[i], l[j])
            k += 1
    return result


def dict_pdist(d: dict[Any, Any], metric: Callable):
    return pdist(list(d.values()), metric=metric)


def max_shared_sample_pct(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    v1_samples, v2_samples = ra1.contained_samples, ra2.contained_samples
    shared_samples = v1_samples & v2_samples
    v1_shared_sample_pct = len(shared_samples) / len(v1_samples)
    v2_shared_sample_pct = len(shared_samples) / len(v2_samples)
    return max(v1_shared_sample_pct, v2_shared_sample_pct)


def min_shared_sample_pct(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    v1_samples, v2_samples = ra1.contained_samples, ra2.contained_samples
    shared_samples = v1_samples & v2_samples
    v1_shared_sample_pct = len(shared_samples) / len(v1_samples)
    v2_shared_sample_pct = len(shared_samples) / len(v2_samples)
    return min(v1_shared_sample_pct, v2_shared_sample_pct)


def shared_sample_pct(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    """Aka the Jaccard similarity."""
    v1_samples, v2_samples = ra1.contained_samples, ra2.contained_samples
    return len(v1_samples & v2_samples) / (len(v1_samples | v2_samples) + 1e-8)  # TODO: should not happen
    return len(v1_samples & v2_samples) / len(v1_samples | v2_samples)


def intersection_area(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    p1, p2 = ra1.region.polygon, ra2.region.polygon
    return p1.intersection(p2).area


def intersection_percentage(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    """The maximum percentage of the overlap between two regions."""
    p1, p2 = ra1.region.polygon, ra2.region.polygon
    i = p1.intersection(p2).area
    return max(i / p1.area, i / p2.area)


def max_intersection_percentage(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    """The maximum percentage of the overlap between two regions."""
    p1, p2 = ra1.region.polygon, ra2.region.polygon
    i = p1.intersection(p2).area
    return max(i / p1.area, i / p2.area)


def intersection_over_union(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    p1, p2 = ra1.region.polygon, ra2.region.polygon
    return p1.intersection(p2).area / p1.union(p2).area


def intersection_over_union_dist(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    """Like intersection over union, but in distance form."""
    return 1 - intersection_over_union(ra1, ra2)


def inbetween_convex_hull_ratio(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    """Calculate the ratio between the area of the empty space and the polygon
    areas if we were to compute the convex hull around both p1 and p2"""
    p1, p2 = ra1.region.polygon, ra2.region.polygon

    total = (p1 | p2).convex_hull
    # Remove convex hulls of p1 and p2 from total area
    inbetween = total - p1.convex_hull - p2.convex_hull
    # Re-add p1 and p2 to total_area
    total = inbetween | p1 | p2

    return inbetween.area / total.area
