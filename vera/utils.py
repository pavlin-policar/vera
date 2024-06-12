from collections import defaultdict

from vera.region_annotation import RegionAnnotation


def flatten(xs):
    if not isinstance(xs, list):
        return [xs]
    return [xi for x in xs for xi in flatten(x)]


def group_by_base_var(region_annotations: list[RegionAnnotation], return_dict: bool = False):
    result = defaultdict(list)
    for ra in region_annotations:
        result[ra.descriptor.base_variable].append(ra)

    result = {base_var: sorted(ra_list) for base_var, ra_list in result.items()}

    if return_dict:
        return result
    else:
        return list(result.values())
