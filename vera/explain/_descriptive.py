from typing import Callable

import numpy as np
from scipy import stats as stats

from vera import metrics as metrics, graph as g
from vera.explain import _layout_scores
from vera.region import Region
from vera.utils import flatten
from vera.region_annotations import RegionAnnotation, RegionAnnotationGroup

DEFAULT_RANKING_FUNCS = [
    (_layout_scores.variable_occurs_in_all_regions, 30),
    (_layout_scores.mean_variable_occurence, 10),
    (_layout_scores.num_regions_matches_perception, 10),
    (_layout_scores.mean_purity, 5),
    (_layout_scores.sample_coverage, 2),
]


def group_similar_variables(
    region_annotations: list[RegionAnnotation],
    metric: Callable = metrics.min_shared_sample_pct,
    metric_is_distance: bool = False,
    threshold: float = 0.9,
    method: str = "connected-components",
):
    distances = metrics.pdist(region_annotations, metric)
    g_func = [g.similarities_to_graph, g.distances_to_graph][metric_is_distance]
    graph = g_func(distances, threshold=threshold)

    node_labels = dict(enumerate(region_annotations))
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
        RegionAnnotationGroup(region_annotations=clust_vars, name=cid)
        for cid, clust_vars in clusts.items()
    ]

    return variable_groups


def enrich_var_group_with_background(
    ra_group: RegionAnnotationGroup,
    background_ras: list[RegionAnnotation],
    threshold: float = 0.9,
):
    # Determine which base vars are present in the var group
    contained_base_vars = {ra.variable.base_variable for ra in ra_group.region_annotations}

    to_add = []
    for background_ra in background_ras:
        # If the background variable belongs to the same base variable
        # as already present in the data, skip that
        if background_ra.variable.base_variable in contained_base_vars:
            continue

        # Determine if the background var encompasses the current variable
        shared_samples = ra_group.contained_samples & background_ra.contained_samples
        pct_shared = len(shared_samples) / len(ra_group.contained_samples)

        # If the region sufficiently overlaps, prepare the overlap for merge
        if pct_shared > threshold:
            new_polygon = background_ra.region.polygon.intersection(
                ra_group.region.polygon
            )
            cloned_bg_var = RegionAnnotation(
                background_ra.variable,
                Region(background_ra.region.embedding, new_polygon),
            )
            to_add.append(cloned_bg_var)

    # If we found any background variables to add to the var group, create a new
    # var group with all available explanatory vars
    if len(to_add) > 0:
        ra_group = RegionAnnotationGroup(
            ra_group.region_annotations + to_add, name=ra_group.name
        )

    return ra_group


def enrich_panel_with_background(
    panel: list[RegionAnnotationGroup],
    background_vars: list[RegionAnnotation],
    threshold: float = 0.9,
):
    return [
        enrich_var_group_with_background(var_group, background_vars, threshold)
        for var_group in panel
    ]


def enrich_layout_with_background(
    layout: list[list[RegionAnnotationGroup]],
    background_vars: list[RegionAnnotation],
    threshold: float = 0.9,
):
    return [
        enrich_panel_with_background(panel, background_vars, threshold)
        for panel in layout
    ]


def generate_descriptive_layout(
    clusters,
    max_panels: int | None = None,
    max_overlap: float = 0.0,
    ranking_funcs=DEFAULT_RANKING_FUNCS,
):
    # Create a working copy of our clusters
    clusters = list(clusters)

    if max_panels is None:
        max_panels = np.inf

    score_fns, score_weights = list(zip(*ranking_funcs))

    # We keep generating new panels until we reach the panel limit or run out of
    # explanatory variables to put into the panels
    layout = []
    while len(layout) < max_panels and len(clusters) > 0:
        # Generate all non-overlapping layouts
        overlap = metrics.pdist(clusters, metrics.max_shared_sample_pct)
        graph = g.similarities_to_graph(overlap, threshold=max_overlap)
        node_labels = dict(enumerate(clusters))
        graph = g.label_nodes(graph, node_labels)

        layouts = g.independent_sets(graph)

        # Sort the panels according to our metrics
        panel_scores = np.array([
            [score_fn(layout) for score_fn in score_fns]
            for layout in layouts
        ])
        rankings = stats.rankdata(panel_scores, method="max", axis=0)

        score_weights = np.array(score_weights)
        mean_ranks = np.mean(rankings * score_weights, axis=1)

        # Select the layout with the highest weighted mean rank
        selected_layout = layouts[np.argmax(mean_ranks)]
        layout.append(selected_layout)

        # Remove variables in the current panel from the remaining variables
        for var in selected_layout:
            clusters.remove(var)

    return layout


def descriptive(
    region_annotations: list[list[RegionAnnotation]],
    max_panels: int | None = 4,
    merge_metric: Callable = metrics.min_shared_sample_pct,
    metric_is_distance: bool = False,
    merge_method: str = "connected-components",
    merge_threshold: float = 0.8,
    cluster_min_samples: int = 5,
    cluster_min_purity: float = 0.5,
    max_overlap: float = 0.0,
    enrich_with_background: bool = True,
    background_enrichment_threshold: float = 0.9,
    return_clusters: bool = False,
    ranking_funcs=DEFAULT_RANKING_FUNCS,
):
    region_annotations = flatten(region_annotations)

    # Split explanatory features into their polygons
    ra_split = [ra_part for ra in region_annotations for ra_part in ra.split()]

    # Remove any split parts that do not pass the filtering criteria
    ra_split = filter_region_annotations(
        ra_split,
        min_samples=cluster_min_samples,
        min_purity=cluster_min_purity,
    )

    clusters = group_similar_variables(
        ra_split,
        metric=merge_metric,
        metric_is_distance=metric_is_distance,
        threshold=merge_threshold,
        method=merge_method,
    )

    layout = generate_descriptive_layout(
        clusters,
        max_panels=max_panels,
        max_overlap=max_overlap,
        ranking_funcs=ranking_funcs,
    )

    if enrich_with_background:
        layout = enrich_layout_with_background(
            layout,
            region_annotations,
            threshold=background_enrichment_threshold,
        )

    if return_clusters:
        return layout, clusters
    else:
        return layout


def filter_region_annotations(
    region_annotations: list[RegionAnnotation],
    min_samples: int = 5,
    min_purity: float = 0.5,
):
    return [
        ra
        for ra in region_annotations
        if metrics.purity(ra) >= min_purity
        and len(ra.contained_samples) >= min_samples
    ]
