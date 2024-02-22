from veca.variables import ExplanatoryVariable

from .contrastive import contrastive
from .descriptive import descriptive


def filter_explanatory_features(
    variables: list[ExplanatoryVariable],
    min_samples: int = 5,
    min_purity: float = 0.5,
    max_geary_index: float = 1,  # no filtering by Geary
):
    return [
        v
        for v in variables
        if v.purity >= min_purity
        and v.gearys_c <= max_geary_index
        and v.num_contained_samples >= min_samples
    ]
