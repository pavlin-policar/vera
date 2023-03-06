from typing import Any, Callable

import numpy as np

from embedding_annotation.region import Region


def _dict_pdist(d: dict[Any, Any], metric: Callable):
    d = list(d.values())

    n = len(d)
    out_size = (n * (n - 1)) // 2
    result = np.zeros(out_size, dtype=np.float64)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            result[k] = metric(d[i], d[j])
            k += 1
    return result


def intersection_area(r1: Region, r2: Region) -> float:
    p1, p2 = r1.polygon, r2.polygon
    return p1.intersection(p2).area


def intersection_percentage(r1: Region, r2: Region) -> float:
    """The maximum percentage of the overlap between two regions."""
    p1, p2 = r1.polygon, r2.polygon
    i = p1.intersection(p2).area
    return max(i / p1.area, i / p2.area)


def intersection_over_union(r1: Region, r2: Region) -> float:
    p1, p2 = r1.polygon, r2.polygon
    return p1.intersection(p2).area / p1.union(p2).area


def intersection_over_union_dist(r1: Region, r2: Region) -> float:
    """Like intersection over union, but in distance form."""
    return 1 - intersection_over_union(r1, r2)


def inbetween_convex_hull_ratio(r1: Region, r2: Region) -> float:
    """Calculate the ratio between the area of the empty space and the polygon
    areas if we were to compute the convex hull around both p1 and p2"""
    p1, p2 = r1.polygon, r2.polygon

    total = (p1 | p2).convex_hull
    # Remove convex hulls of p1 and p2 from total area
    inbetween = total - p1.convex_hull - p2.convex_hull
    # Re-add p1 and p2 to total_area
    total = inbetween | p1 | p2

    return inbetween.area / total.area
