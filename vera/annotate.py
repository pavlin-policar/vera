from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from tqdm import tqdm

import vera.preprocessing as pp
from vera.region_annotation import RegionAnnotation
from vera.variables import Variable


def _is_uninformative(
    ra_group: list[RegionAnnotation], max_sample_coverage: float
) -> bool:
    """Determine whether a variable's region annotations describe nothing.

    A variable that splits the embedding into several regions is informative.
    A variable left with a single region can still be informative, provided it
    did not arrive at that single region by having all its indicators merged
    together — a discretized or one-hot encoded variable whose regions all
    merged is described by a rule spanning every one of its values, which holds
    for every sample and therefore explains nothing.

    Otherwise the region itself has to say something, which it fails to do when
    it singles out almost none of the data. Both the rule and the region are
    tested for this: the two come apart, since a region tight around a
    near-universal rule still describes the whole data set, while a rule
    matched by few samples scattered evenly across the embedding still yields a
    region spanning all of it.
    """
    if len(ra_group) > 1:
        return False

    ra = ra_group[0]
    if len(ra.contained_region_annotations) > 1:
        return True

    max_samples = max_sample_coverage * ra.region.embedding.X.shape[0]
    return (
        len(ra.all_members) >= max_samples
        or len(ra.contained_samples) >= max_samples
    )


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
    uninformative_max_sample_coverage: float = 0.95,
    indicator_columns: pp.IndicatorColumns = None,
    random_state: Any = None,
) -> list[list[RegionAnnotation]]:
    """
    Generate region annotations for variables in a features DataFrame and an
    embedding.

    This function samples the data if it exceeds a given sample size, expands
    each feature into indicator variables (via discretization or one-hot
    encoding), and generates  region annotations using either KDE contouring or
    rangeset triangulation methods. Optionally, overfragmented regions are
    iteratively merged, and uninformative variables can be filtered out.

    Parameters
    ----------
    features : pd.DataFrame
        Explanatory features. Continuous columns are discretized and
        categorical columns are one-hot encoded, unless the column is selected
        by `indicator_columns` or named by a
        :class:`~vera.variables.Variable`, which is used as-is.
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
        If True, variables that describe nothing are filtered out. These are
        variables whose indicators all merged into a single region annotation,
        and variables left with a single region annotation that singles out
        almost none of the data.
    uninformative_max_sample_coverage : float, default=0.95
        Fraction of the data that a single region annotation's rule may match,
        or that its region may contain, before its variable is considered
        uninformative; only used if `filter_uninformative=True`. This is the
        sole criterion for indicator columns, which are inherently described by
        one region annotation each.
    indicator_columns : str or iterable or dict, default=None
        Columns holding boolean indicators. Such a column is described by its
        positive case alone -- annotated with its own name, and silent about
        the samples it does not flag -- which is what a presence feature calls
        for: a region labelled "gene is absent" says little. Columns have to be
        of boolean dtype; a 0/1 or categorical column is rejected rather than
        converted. Missing values are supported through pandas' nullable
        ``boolean`` dtype: a sample with no measurement shapes no region and is
        left out of the variable's rates. Pass ``"all"`` for a table of nothing
        but indicators, a collection of column names to select them out of a
        mixed table, or a mapping from column name to the text annotating it.
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
        indicator_columns=indicator_columns,
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

    # Filter out annotation groups whose variable describes nothing
    if filter_uninformative:
        region_annotations = [
            ra_group
            for ra_group in region_annotations
            if not _is_uninformative(ra_group, uninformative_max_sample_coverage)
        ]

    return region_annotations
