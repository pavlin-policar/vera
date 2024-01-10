from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

import veca.preprocessing as pp
from veca.variables import Variable


def generate_explanatory_features(
    features: pd.DataFrame,
    embedding: np.ndarray,
    sample_size: int = 5000,
    n_discretization_bins: int = 5,
    scale_factor: float = 1,
    kernel: str = "gaussian",
    contour_level: float = 0.25,
    merge_min_purity_gain=0.5,
    merge_min_sample_overlap=0.5,
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
    df_derived = pp.generate_derived_features(
        features, n_discretization_bins=n_discretization_bins
    )

    # Create explanatory variables from each of the derived features
    explanatory_features = pp.convert_derived_features_to_explanatory(
        df_derived,
        embedding,
        scale_factor=scale_factor,
        kernel=kernel,
        contour_level=contour_level,
    )

    # Perform iterative merging
    merged_explanatory_features = pp.merge_overfragmented(
        explanatory_features,
        min_purity_gain=merge_min_purity_gain,
        min_sample_overlap=merge_min_sample_overlap,
    )

    # Return the list of base variables, which contain the explanatory variables
    base_variables = set(v.base_variable for v in merged_explanatory_features)
    return list(base_variables)
