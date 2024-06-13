from typing import Callable

import numpy as np
from scipy import stats as stats

from vera import metrics as metrics, graph as g
from vera.explain import _layout_scores
from vera.region import Region
from vera.utils import flatten
from vera.region_annotation import RegionAnnotation
from vera.variables import RegionDescriptor

DEFAULT_RANKING_FUNCS = [
    (_layout_scores.variable_occurs_in_all_regions, 30),
    (_layout_scores.mean_variable_occurence, 10),
    (_layout_scores.num_regions_matches_perception, 10),
    (_layout_scores.mean_purity, 5),
    (_layout_scores.sample_coverage, 2),
]


def descriptive_merge(
    region_annotations: list[RegionAnnotation],
    metric: Callable = metrics.min_shared_sample_pct,
    metric_is_distance: bool = False,
    threshold: float = 0.9,
    method: str = "connected-components",
):
    distances = metrics.pdist(region_annotations, metric)
    g_func = [g.similarities_to_graph, g.distances_to_graph][metric_is_distance]
    graph = g_func(distances, threshold=threshold)

    # The distance conversion constructs a graph with integer indices as nodes,
    # so we convert the nodes to our region annotations here. The result is a
    # graph on region annotations
    node_labels = dict(enumerate(region_annotations))
    graph = g.label_nodes(graph, node_labels)

    if method == "max-cliques":
        subgraphs = g.max_cliques(graph)
    elif method == "connected-components":
        subgraphs = g.connected_components(graph)
    else:
        raise ValueError(
            f"Unrecognized method `{method}`. Can be one of `max-cliques`, "
            f"`connected-components`"
        )

    # The above two methods produce subgraphs of the original graphs as output,
    # so we next conver this to lists of graph nodes (region annotations).
    ra_groups = list(map(g.nodes, subgraphs))

    cluster_region_annotations = []
    for cluster_id, ra_group in enumerate(ra_groups, start=1):
        merged_descriptor = RegionDescriptor.merge_descriptors(
            [ra.descriptor for ra in ra_group]
        )
        merged_region = Region.merge_regions([ra.region for ra in ra_group])
        merged_ra = RegionAnnotation(
            region=merged_region, descriptor=merged_descriptor
        )

        cluster_region_annotations.append(merged_ra)

    return cluster_region_annotations


def enrich_with_background(
    region_annotation: RegionAnnotation,
    background: list[RegionAnnotation],
    threshold: float = 0.9,
):
    # Determine which base vars are present in the var group
    contained_base_vars = {v for v in region_annotation.descriptor.contained_variables}

    to_add = []
    for background_ra in background:
        # If the region annotation belongs to the same base variable as already
        # present in the data, skip it, since the RA in this region annotation
        # is more specific to the present region than it would be if we merged
        # the present RA with the background RA
        if background_ra.descriptor.base_variable in contained_base_vars:
            continue

        # Determine if the background var sufficiently encompasses the current variable
        shared_samples = region_annotation.contained_samples & background_ra.contained_samples
        pct_shared = len(shared_samples) / len(region_annotation.contained_samples)

        # If the region sufficiently overlaps, the descriptor should be added
        if pct_shared > threshold:
            to_add.append(background_ra.descriptor)

    # If we found any background descriptors to add to the region annotation,
    # create a new region annotation with all available descriptors
    if len(to_add) > 0:
        region_annotation = RegionAnnotation(
            descriptor=RegionDescriptor.merge_descriptors([region_annotation.descriptor, *to_add]),
            region=region_annotation.region,
            source_region_annotations=region_annotation.source_region_annotations,
        )

    return region_annotation


def enrich_panel_with_background(
    panel: list[RegionAnnotation],
    background: list[RegionAnnotation],
    threshold: float = 0.9,
):
    return [
        enrich_with_background(var_group, background, threshold)
        for var_group in panel
    ]


def enrich_layout_with_background(
    layout: list[list[RegionAnnotation]],
    background: list[RegionAnnotation],
    threshold: float = 0.9,
):
    return [
        enrich_panel_with_background(panel, background, threshold)
        for panel in layout
    ]


def generate_descriptive_layout(
    region_annotations: list[RegionAnnotation],
    max_panels: int | None = None,
    max_overlap: float = 0.0,
    ranking_funcs=DEFAULT_RANKING_FUNCS,
):
    # Create a working copy of our clusters
    region_annotations = list(region_annotations)

    if max_panels is None:
        max_panels = np.inf

    score_fns, score_weights = list(zip(*ranking_funcs))

    # We keep generating new panels until we reach the panel limit or run out of
    # explanatory variables to put into the panels
    layout = []
    while len(layout) < max_panels and len(region_annotations) > 0:
        # Generate all non-overlapping layouts
        overlap = metrics.pdist(region_annotations, metrics.max_shared_sample_pct)
        graph = g.similarities_to_graph(overlap, threshold=max_overlap)
        node_labels = dict(enumerate(region_annotations))
        graph = g.label_nodes(graph, node_labels)

        layouts = g.independent_sets(graph)

        # Sort the panels according to their metric scores
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
            region_annotations.remove(var)

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

    merged_region_annotations = descriptive_merge(
        ra_split,
        metric=merge_metric,
        metric_is_distance=metric_is_distance,
        threshold=merge_threshold,
        method=merge_method,
    )

    layout = generate_descriptive_layout(
        merged_region_annotations,
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
