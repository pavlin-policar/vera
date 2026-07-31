from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from tqdm import tqdm

import vera.preprocessing as pp
from vera.region_annotation import RegionAnnotation
from vera.variables import Variable


def generate_region_annotations(
    features: pd.DataFrame,
    embedding: np.ndarray,
    sample_size: int = 5000,
    filter_constant: bool = True,
    n_discretization_bins: int = 5,
    scale_factor: float = 1,
    region_method: str = "kde",
    kernel: str = "gaussian",
    contour_level: float = 0.25,
    merge_min_sample_overlap: float = 0.8,
    filter_uninformative: bool = True,
    random_state: Any = None,
) -> list[list[RegionAnnotation]]:
    """
    Generate region annotations for variables in a features DataFrame and an
    embedding.

    This function samples the data if it exceeds a given sample size, expands
    each feature into indicator variables (via discretization or one-hot 
    encoding), and generates  region annotations using either KDE contouring or
    rangeset triangulation methods. Optionally, overfragmented regions are
    iteratively merged, and uninformative variables (those described by a single
    (those described by a single region) can be filtered out.

    Parameters
    ----------
    features : pd.DataFrame
        Explanatory features. A column named by an
        :class:`~vera.variables.IndicatorVariable` is used as-is instead of
        being discretized or one-hot encoded; this is how a binary feature is
        described by its positive case alone. Such a variable forms a group of
        one, so it survives only with ``filter_uninformative=False``.
    embedding : np.ndarray
        Low-dimensional embedding of the data to explain.
    sample_size : int, default=5000
        Maximum number of samples to use; if the data has more rows, it is 
        randomly subsampled.
    filter_constant : bool, default=True
        If True, constant (uninformative) features are filtered out.
    n_discretization_bins : int, default=5
        Number of bins used for discretizing continuous variables.
    scale_factor : float, default=1
        Controls the KDE bandwidth and/or rangeset edge-cutoff threshold.
    method : {"kde", "rangeset"}, default="kde"
        Region extraction method.
    kernel : str, default="gaussian"
        KDE kernel; only used if method="kde".
    contour_level : float, default=0.25
        Density contour level for region extraction; only used if method="kde".
    merge_min_sample_overlap : float, default=0.8
        Minimum overlap (fraction of shared samples) required for merging
        overfragmented region annotations.
    filter_uninformative : bool, default=True
        If True, variables described by only a single region annotation are
        filtered out.
    random_state : Any, default=None
        Random state for reproducibility of sampling and of the k-means
        discretization of continuous variables.

    Returns
    -------
    region_annotations : list[list[RegionAnnotation]]
        List of lists, where each inner list contains
        :class:`RegionAnnotation` objects describing the regions associated with
        one variable or variable group.
    """
    # Sample the data if necessary. Running on large data sets can be very slow
    random_state = check_random_state(random_state)
    if sample_size is not None and features.shape[0] > sample_size:
        num_samples = min(sample_size, features.shape[0])
        sample_idx = random_state.choice(
            features.shape[0], size=num_samples, replace=False
        )
        # A column name can itself be a variable, in which case the values it
        # carries are the ones used downstream, and pandas indexing leaves them
        # untouched
        columns = [
            c.subset(sample_idx) if isinstance(c, Variable) else c
            for c in features.columns
        ]
        features = features.iloc[sample_idx].set_axis(columns, axis="columns")
        embedding = embedding[sample_idx]

    # Convert the data frame to VERA feature objects
    variables = pp.expand_df(
        features,
        n_discretization_bins=n_discretization_bins,
        filter_constant_features=filter_constant,
        random_state=random_state,
    )

    # Generate explanatory region annotations from each of the derived features
    region_annotations = pp.extract_region_annotations(
        variables,
        embedding,
        scale_factor=scale_factor,
        region_method=region_method,
        kernel=kernel,
        contour_level=contour_level,
    )

    # Perform iterative merging on every single region annotation group
    region_annotations = [
        pp.merge_overfragmented(
            ra_group, min_sample_overlap=merge_min_sample_overlap
        )
        for ra_group in tqdm(region_annotations)
    ]

    # Filter annotation groups if the variable is described by a single region
    if filter_uninformative:
        region_annotations = [
            ra_group for ra_group in region_annotations if len(ra_group) > 1
        ]

    return region_annotations
