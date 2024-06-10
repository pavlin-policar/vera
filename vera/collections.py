from collections import defaultdict

from vera.region_annotations import RegionAnnotation
from vera.variables import Variable


class RACollection:
    def __init__(self, region_annotations: list[RegionAnnotation]):
        self.region_annotations = list(region_annotations)

    @property
    def variable_dict(self) -> dict[Variable, list[RegionAnnotation]]:
        result = defaultdict(list)
        for ra in self.region_annotations:
            result[ra.variable.base_variable].append(ra)
        return dict(result)

    @property
    def variables(self) -> list[Variable]:
        """Get list of base variables."""
        return list(self.variable_dict)

    def __getitem__(self, item) -> RegionAnnotation | list[RegionAnnotation]:
        """Indexing object returns a list of assocaited region annotations."""
        if item in self.variable_dict:
            return self.variable_dict[item]
        if isinstance(item, str):
            var_mapping = {v.name: v for v in self.variable_dict}
            if item in var_mapping:
                return self.variable_dict[var_mapping[item]]
        if isinstance(item, int):
            return self.region_annotations[item]
        raise KeyError(item)
