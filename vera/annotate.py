from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from tqdm import tqdm

import vera.preprocessing as pp
from vera.region_annotation import RegionAnnotation


def generate_region_annotations(
    features: pd.DataFrame,
    embedding: np.ndarray,
    sample_size: int = 5000,
    filter_constant: bool = True,
    n_discretization_bins: int = 5,
    scale_factor: float = 1,
    method: str = "kde",
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
        Explanatory features.
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
        Random state for reproducibility of sampling.

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
        features = features.iloc[sample_idx]
        embedding = embedding[sample_idx]

    # Convert the data frame to VERA feature objects
    variables = pp.expand_df(
        features,
        n_discretization_bins=n_discretization_bins,
        filter_constant_features=filter_constant,
    )

    # Generate explanatory region annotations from each of the derived features
    region_annotations = pp.extract_region_annotations(
        variables,
        embedding,
        scale_factor=scale_factor,
        method=method,
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


# def generate_indicator_explanatory_features(
#     features: pd.DataFrame,
#     embedding: np.ndarray,
#     sample_size: int = 5000,
#     filter_constant: bool = True,
#     threshold: str | float = "auto",
#     scale_factor: float = 1,
#     kernel: str = "gaussian",
#     contour_level: float = 0.25,
#     merge_min_sample_overlap=0.8,
#     merge_min_purity_gain=0.5,
#     random_state: Any = None,
# ):
#     # Sample the data if necessary. Running on large data sets can be very slow
#     random_state = check_random_state(random_state)
#     if sample_size is not None:
#         num_samples = min(sample_size, features.shape[0])
#         sample_idx = random_state.choice(
#             features.shape[0], size=num_samples, replace=False
#         )
#         features = features.iloc[sample_idx]
#         embedding = embedding[sample_idx]
#
#     # Filter out features with identical values
#     if filter_constant:
#         df = df.loc[:, df.nunique(axis=0) > 1]
#
#     df = pp.ingest(df)
#
#     # Determine binary features
#     if threshold == "auto":
#         df = _one_hot(_discretize(ingest(df), n_bins=2))
#     else:
#         df_binary = df > threshold
#
#     # Create explanatory variables from each of the derived features
#     explanatory_features = pp.generate_region_annotations(
#         df_binary,
#         embedding,
#         scale_factor=scale_factor,
#         kernel=kernel,
#         contour_level=contour_level,
#     )
#
#     # Perform iterative merging
#     merged_explanatory_features = pp.merge_overfragmented(
#         explanatory_features,
#         min_sample_overlap=merge_min_sample_overlap,
#         min_purity_gain=merge_min_purity_gain,
#     )
#
#     # Generate list of base variables, sorted for consistency
#     base_variables = sorted(
#         list(set(v.base_variable for v in merged_region_annotations))
#     )
#
#     return base_variables
