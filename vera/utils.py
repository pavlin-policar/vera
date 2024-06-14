from collections import defaultdict

from vera.region_annotation import RegionAnnotation
from vera.variables import IndicatorVariable, RegionDescriptor


def flatten(xs):
    if not isinstance(xs, list):
        return [xs]
    return [xi for x in xs for xi in flatten(x)]


def group_by_descriptor(
    ras: list[RegionAnnotation],
    return_dict: bool = False,
) -> dict[tuple[RegionDescriptor], list[RegionAnnotation]]:
    result = defaultdict(list)
    for ra in ras:
        result[ra.descriptor.contained_variables].append(ra)

    if return_dict:
        return dict(result)
    else:
        return list(result.values())


def group_by_base_var(region_annotations: list[RegionAnnotation], return_dict: bool = False):
    result = defaultdict(list)
    for ra in region_annotations:
        if not isinstance(ra.descriptor, IndicatorVariable):
            raise TypeError(
                f"Can only group instances of `{IndicatorVariable.__class__.__name__}`"
            )
        result[ra.descriptor.base_variable].append(ra)

    result = {base_var: sorted(ra_list) for base_var, ra_list in result.items()}

    if return_dict:
        return result
    else:
        return list(result.values())
