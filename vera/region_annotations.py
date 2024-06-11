import itertools
import operator
from functools import reduce, cached_property

import numpy as np

from vera.region import Region
from vera.rules import IncompatibleRuleError
from vera.variables import IndicatorVariable, MergeError, VariableGroup


class RegionAnnotation:
    def __init__(
        self,
        variable: IndicatorVariable,
        region: Region,
    ):
        self.variable = variable
        self.region = region

    def can_merge_with(self, other: "RegionAnnotation") -> bool:
        if not isinstance(other, RegionAnnotation):
            return False

        # The variables must match on their base variable
        if self.variable.base_variable != other.variable.base_variable:
            return False

        if not self.variable.rule.can_merge_with(other.variable.rule):
            return False

        # The embedding has to be the same
        if not np.allclose(self.region.embedding.X, other.region.embedding.X):
            return False

        return True

    def merge_with(self, other: "RegionAnnotation") -> "ExplanatoryRegion":
        if not self.can_merge_with(other):
            raise MergeError(f"Cannot merge `{self}` and {other}!")
        return CompositeRegionAnnotation([self, other])

    def __repr__(self):
        attrs = [
            ("variable", repr(self.variable)),
            ("region", repr(self.region)),
        ]
        attrs_str = ", ".join(f"{k}={v}" for k, v in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    def split(self) -> list["RegionAnnotation"]:
        """If a variable comprises multiple regions, split each region into its
        own object."""
        return [
            RegionAnnotation(self.variable, region)
            for region in self.region.split_into_parts()
        ]

    @property
    def name(self):
        return self.variable.name

    @property
    def contained_region_annotations(self):
        return [self]

    @property
    def contained_samples(self):
        """Return the indices of all data points inside the region."""
        return self.region.contained_samples

    @property
    def all_members(self):
        """Return the indices of all data points that fulfill the rule."""
        return set(np.argwhere(self.variable.values).ravel())

    @property
    def contained_members(self):
        """Return the indices of all data points that fulfill the rule inside the region."""
        return self.contained_samples & self.all_members

    def __hash__(self):
        return hash((self.__class__.__name__, self.variable, self.region))

    def __eq__(self, other):
        if not isinstance(other, RegionAnnotation):
            return False
        return self.variable == other.variable and self.region == other.region

    def __lt__(self, other: "RegionAnnotation"):
        return self.variable < other.variable


class CompositeRegionAnnotation(RegionAnnotation):
    """Composite region annotations require all sub-region annotations to
    originate from the same base variable."""
    def __new__(cls, region_annotations: list[RegionAnnotation]):
        # If the list of region annotations contains a single entry, use that
        if len(region_annotations) == 1:
            return region_annotations[0]
        return super().__new__(cls)

    def __init__(self, region_annotations: list[RegionAnnotation]):
        self.base_region_annotations = region_annotations

        ra0 = region_annotations[0]
        base_variable = ra0.variable.base_variable
        if not all(ra.variable.base_variable == base_variable for ra in region_annotations[1:]):
            raise RuntimeError(
                "Cannot merge Region Annotations which do not share the "
                "same base variable!"
            )

        try:
            # Sort region annotations, so we will be able to merge them properly
            ra_order = sorted(region_annotations)
            new_rule = ra_order[0].variable.rule
            for ra in ra_order[1:]:
                new_rule = new_rule.merge_with(ra.variable.rule)
        except IncompatibleRuleError as e:
            raise MergeError(
                f"CompositeRegionAnnotation cannot be created because the "
                f"variables given were incompatible:\n{e.message}\n"
            )

        # Merge the values of the underlying samples, use OR/ANY relation
        values = np.vstack([ra.variable.values for ra in region_annotations])
        merged_values = np.max(values, axis=0)
        merged_variable = IndicatorVariable(base_variable, new_rule, merged_values)

        merged_region = Region.merge_regions([ra.region for ra in region_annotations])

        super().__init__(variable=merged_variable, region=merged_region)

    @property
    def contained_region_annotations(self):
        return sorted(
            reduce(
                operator.add,
                [v.contained_region_annotations for v in self.base_region_annotations]
            )
        )

    @cached_property
    def contained_samples(self):
        """Checking the region for contained samples is slow."""
        return reduce(operator.or_, (v.contained_samples for v in self.base_region_annotations))


class RegionAnnotationGroup:
    def __init__(self, region_annotations: list[RegionAnnotation], name: str = None):
        self.name = name

        # In case the variables contain multiple instances of the same variable,
        # join them up together into a composite variable.
        # TODO: This will fail if we try to join up two variables with
        #  non-compatible rules
        # TODO: This should also subsequently remove the CRA if the CRA now
        #  contains all sub-RAs for that particular variable
        region_annotations_ = []

        # itertools.groupby expects consecutive keys, but we accept them in any order
        region_annotations = sorted(region_annotations)
        for key, ra_group in itertools.groupby(region_annotations, lambda ra: ra.variable.base_variable):
            ra_group = sorted(list(ra_group))
            if len(ra_group) > 1:
                try:
                    ra_group = [CompositeRegionAnnotation(ra_group)]
                except MergeError:
                    raise
                    pass  # If merging failed, keep the variables unmerged
            region_annotations_ += ra_group

        self.region_annotations: list[RegionAnnotation] = region_annotations_

        contained_variables = [ra.variable for ra in self.region_annotations]
        # Take MIN/AND: if plotted together, we expect each point to fulfill all
        # the rules in the feature group
        values = np.vstack([ra.variable.values for ra in self.region_annotations])
        merged_values = np.min(values, axis=0)
        self.variable = VariableGroup(variables=contained_variables, values=merged_values)

        self.region = Region.merge_regions([ra.region for ra in self.region_annotations])

    def __repr__(self):
        attrs = [
            ("name", repr(self.name)),
            ("variable", repr(self.variable)),
            ("region", repr(self.region)),
        ]
        attrs_str = ", ".join(f"{k}={v}" for k, v in attrs)
        return f"{self.__class__.__name__}({attrs_str})"

    @property
    def contained_region_annotations(self):
        return sorted(self.region_annotations)

    @cached_property
    def contained_samples(self):
        """Checking the region for contained samples is slow."""
        return reduce(operator.or_, (ra.contained_samples for ra in self.region_annotations))

    @property
    def all_members(self):
        """Return the indices of all data points that fulfill the rule."""
        return set(np.argwhere(self.variable.values).ravel())

    @property
    def contained_members(self):
        """Return the indices of all data points that fulfill the rule inside the region."""
        return self.contained_samples & self.all_members
