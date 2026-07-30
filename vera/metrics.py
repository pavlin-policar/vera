from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np

from vera.variables import IndicatorVariable, IndicatorVariableGroup

if TYPE_CHECKING:
    from vera.region_annotation import RegionAnnotation

EPS = 1e-12


def purity(ra: "RegionAnnotation") -> float:
    contained_vals = ra.descriptor.values[list(ra.region.contained_samples)]
    if len(contained_vals) == 0:
        return 0
    return np.mean(contained_vals)


def _shrunk_rate_and_base_rate(
    v: IndicatorVariable, ra: "RegionAnnotation", prior_strength: float
) -> tuple[float, float]:
    """Estimate the variable's in-region rate and its background base rate.

    The in-region rate is the posterior mean of a Beta-Binomial model with the
    prior centered on the background rate: with ``k`` positives among the
    ``n`` region samples,

        p = (k + prior_strength * q) / (n + prior_strength)

    ``prior_strength`` is the equivalent sample size of the prior, i.e. the
    region size at which the observed rate and the prior are weighted equally.
    Regions with few samples are thereby pulled toward the background rate, so
    a feature cannot rank highly on a handful of coincidental samples.
    """
    S = sorted(ra.region.contained_samples)
    q = float(v.values.mean())
    k = float(v.values[S].sum())
    n = len(S)
    p = (k + prior_strength * q) / max(n + prior_strength, EPS)
    return p, q


def _score_purity(p: float, q: float) -> float:
    return p


def _score_purity_gain(p: float, q: float) -> float:
    return p - q


def _score_lift(p: float, q: float) -> float:
    return p / max(q, EPS)


def _score_purity_lift(p: float, q: float) -> float:
    if p == 0:
        # lim_{p -> 0} p * log2(p / q) = 0, avoid 0 * inf = nan
        return 0.0
    return p * np.log2(p / max(q, EPS))


DESCRIPTOR_SCORING_METHODS = {
    # How much of the region the feature covers. Blind to how common the
    # feature is elsewhere.
    "purity": _score_purity,
    # Coverage minus background rate ("added value" in the subgroup-discovery
    # literature). Coverage dominates; features common across the whole
    # dataset are penalized. The preferred method for descriptive layouts.
    "purity_gain": _score_purity_gain,
    # Observed over expected rate. A contrastive measure: unbounded, and
    # favors rare features regardless of how little of the region they cover.
    "lift": _score_lift,
    # Coverage-weighted log-lift, the positive term of the KL divergence
    # between the in-region and background rates. A contrastive measure that
    # discounts rare features, but only logarithmically.
    "purity_lift": _score_purity_lift,
}


def descriptor_scores(
    ra: "RegionAnnotation",
    method: Union[str, Callable] = "purity_gain",
    prior_strength: float = 10,
) -> dict[IndicatorVariable, float]:
    """Score each of the region annotation's descriptor variables on how well
    it describes the region.

    Splitting deflates the contrastive scores: a variable concentrated in
    several separate regions has its full prevalence as the background rate,
    so `lift` and `purity_lift` are diluted by the variable's other modes.
    `purity_gain` is affected only additively and `purity` not at all.

    Parameters
    ----------
    ra: RegionAnnotation
    method: Union[str, Callable]
        One of `DESCRIPTOR_SCORING_METHODS`, or a callable
        ``(v, ra) -> float`` scoring a single indicator variable.
    prior_strength: float
        The equivalent sample size of the background-centered prior used to
        estimate in-region rates (see `_shrunk_rate_and_base_rate`). Keep at
        or above the smallest region size the pipeline admits; the descriptive
        pipeline's default `cluster_min_samples=5` makes 10 a 2x margin.
        Ignored when `method` is a callable.
    """
    descriptor = ra.descriptor
    if isinstance(descriptor, IndicatorVariableGroup):
        variables = list(descriptor.variables)
    elif isinstance(descriptor, IndicatorVariable):
        variables = [descriptor]
    else:
        raise TypeError(
            f"Cannot score descriptors of type `{descriptor.__class__.__name__}`!"
        )

    if callable(method):
        return {v: float(method(v, ra)) for v in variables}

    if method not in DESCRIPTOR_SCORING_METHODS:
        raise ValueError(
            f"Unrecognized scoring method `{method}`. Valid methods are "
            f"{sorted(DESCRIPTOR_SCORING_METHODS)} or a callable "
            f"(v, ra) -> float."
        )
    score_func = DESCRIPTOR_SCORING_METHODS[method]

    scores = {}
    for v in variables:
        p, q = _shrunk_rate_and_base_rate(v, ra, prior_strength)
        scores[v] = float(score_func(p, q))
    return scores


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
    if not v1_samples or not v2_samples:
        return 0
    shared_samples = v1_samples & v2_samples
    v1_shared_sample_pct = len(shared_samples) / len(v1_samples)
    v2_shared_sample_pct = len(shared_samples) / len(v2_samples)
    return max(v1_shared_sample_pct, v2_shared_sample_pct)


def min_shared_sample_pct(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    v1_samples, v2_samples = ra1.contained_samples, ra2.contained_samples
    if not v1_samples or not v2_samples:
        return 0
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
    if p1.is_empty or p2.is_empty:
        return 0
    i = p1.intersection(p2).area
    return max(i / p1.area, i / p2.area)


def max_intersection_percentage(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    """The maximum percentage of the overlap between two regions."""
    p1, p2 = ra1.region.polygon, ra2.region.polygon
    if p1.is_empty or p2.is_empty:
        return 0
    i = p1.intersection(p2).area
    return max(i / p1.area, i / p2.area)


def intersection_over_union(ra1: RegionAnnotation, ra2: RegionAnnotation) -> float:
    p1, p2 = ra1.region.polygon, ra2.region.polygon
    union_area = p1.union(p2).area
    if union_area == 0:
        return 0
    return p1.intersection(p2).area / union_area


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
