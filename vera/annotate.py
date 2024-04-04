from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

import vera.preprocessing as pp
from vera.variables import Variable


def generate_explanatory_features(
    features: pd.DataFrame,
    embedding: np.ndarray,
    sample_size: int = 5000,
    filter_constant: bool = True,
    n_discretization_bins: int = 5,
    scale_factor: float = 1,
    kernel: str = "gaussian",
    contour_level: float = 0.25,
    merge_min_sample_overlap=0.8,
    merge_min_purity_gain=0.5,
    random_state: Any = None,
) -> list[Variable]:
    # Sample the data if necessary. Running on large data sets can be very slow
    random_state = check_random_state(random_state)
    if sample_size is not None:
        num_samples = min(sample_size, features.shape[0])
        sample_idx = random_state.choice(
            features.shape[0], size=num_samples, replace=False
        )
        features = features.iloc[sample_idx]
        embedding = embedding[sample_idx]

    # Convert the data frame so that it contains derived features
    df_expanded = pp.expand_to_indicator(
        features,
        n_discretization_bins=n_discretization_bins,
        filter_constant=filter_constant,
    )

    # Create explanatory variables from each of the derived features
    explanatory_features = pp.generate_explanatory_features(
        df_expanded,
        embedding,
        scale_factor=scale_factor,
        kernel=kernel,
        contour_level=contour_level,
    )

    # Perform iterative merging
    merged_explanatory_features = pp.merge_overfragmented(
        explanatory_features,
        min_sample_overlap=merge_min_sample_overlap,
        min_purity_gain=merge_min_purity_gain,
    )

    # Return the list of base variables, which contain the explanatory variables
    base_variables = set(v.base_variable for v in merged_explanatory_features)
    return list(base_variables)


def generate_indicator_explanatory_features(
    features: pd.DataFrame,
    embedding: np.ndarray,
    sample_size: int = 5000,
    filter_constant: bool = True,
    threshold: str | float = "auto",
    scale_factor: float = 1,
    kernel: str = "gaussian",
    contour_level: float = 0.25,
    merge_min_sample_overlap=0.8,
    merge_min_purity_gain=0.5,
    random_state: Any = None,
):
    # Sample the data if necessary. Running on large data sets can be very slow
    random_state = check_random_state(random_state)
    if sample_size is not None:
        num_samples = min(sample_size, features.shape[0])
        sample_idx = random_state.choice(
            features.shape[0], size=num_samples, replace=False
        )
        features = features.iloc[sample_idx]
        embedding = embedding[sample_idx]


    # Filter out features with identical values
    if filter_constant:
        df = df.loc[:, df.nunique(axis=0) > 1]

    df = pp.ingest(df)

    # Determine binary features
    if threshold == "auto":
        df = _one_hot(_discretize(ingest(df), n_bins=2))
    else:
        df_binary = df > threshold

    # Create explanatory variables from each of the derived features
    explanatory_features = pp.generate_explanatory_features(
        df_binary,
        embedding,
        scale_factor=scale_factor,
        kernel=kernel,
        contour_level=contour_level,
    )

    # Perform iterative merging
    merged_explanatory_features = pp.merge_overfragmented(
        explanatory_features,
        min_sample_overlap=merge_min_sample_overlap,
        min_purity_gain=merge_min_purity_gain,
    )

    # Return the list of base variables, which contain the explanatory variables
    base_variables = set(v.base_variable for v in merged_explanatory_features)
    return list(base_variables)

