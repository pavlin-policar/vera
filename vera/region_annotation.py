import copy

import numpy as np

from vera import metrics
from vera.region import Region
from vera.variables import IndicatorVariableGroup, RegionDescriptor


class RegionAnnotation:
    """Pairs a region of the embedding with the descriptor that explains it.

    A group descriptor's variables are stored ranked: most characteristic of
    this region first. The ranking is a property of the descriptor-region
    pairing, so the same group held by different region annotations (e.g. the
    parts of a split) may be ordered differently. Pass ``rank_descriptor=False``
    to keep the group's own alphabetical order instead — contrastive
    explanations do this so that a variable group reads identically in every
    region it annotates.
    """
    def __init__(
        self,
        region: Region,
        descriptor: RegionDescriptor,
        source_region_annotations: list["RegionAnnotation"] = None,
        rank_descriptor: bool = True,
    ):
        self.descriptor = descriptor
        self.region = region
        self.source_region_annotations = source_region_annotations

        if rank_descriptor and isinstance(descriptor, IndicatorVariableGroup):
            self.descriptor = self._ranked_descriptor(descriptor)

    def _ranked_descriptor(
        self, group: IndicatorVariableGroup
    ) -> IndicatorVariableGroup:
        """A copy of the group with its variables ranked for this region.

        Descriptor objects can be shared between region annotations, so the
        group is copied rather than reordered in place. Ties are broken by the
        variables' natural order, which makes the ranking deterministic. A NaN
        score would corrupt the sort, so it is treated as the lowest possible
        score.
        """
        scores = metrics.descriptor_scores(self)

        def sort_key(v):
            return -np.inf if np.isnan(scores[v]) else scores[v]

        ranked = sorted(sorted(scores), key=sort_key, reverse=True)

        ranked_group = copy.copy(group)
        ranked_group.variables = ranked
        return ranked_group

    def can_merge_with(self, other: "RegionAnnotation") -> bool:
        """Region annotations can be merged if their regions and descriptors are
        compatible. The exact method in which they are merged is left to the
        caller."""
        if not isinstance(other, RegionAnnotation):
            return False

        # The descriptors need to be compatible
        if not self.descriptor.can_merge_with(other.descriptor):
            return False

        # The embedding has to be the same
        if not np.allclose(self.region.embedding.X, other.region.embedding.X):
            return False

        return True

    @classmethod
    def merge(
        cls,
        region_annotations: list["RegionAnnotation"],
        rank_descriptor: bool = True,
    ) -> "RegionAnnotation":
        if len(region_annotations) == 1:
            return region_annotations[0]

        merged_descriptor = RegionDescriptor.merge(
            [ra.descriptor for ra in region_annotations]
        )
        merged_region = Region.merge([ra.region for ra in region_annotations])

        return RegionAnnotation(
            region=merged_region,
            descriptor=merged_descriptor,
            source_region_annotations=region_annotations,
            rank_descriptor=rank_descriptor,
        )

    def split(self) -> list["RegionAnnotation"]:
        """If a variable comprises multiple regions, split each region into its
        own object."""
        region_parts = self.region.split_into_parts()

        # If there is only a single part, no need to do anything
        if len(region_parts) == 1:
            return [self]

        return [
            RegionAnnotation(region, self.descriptor, source_region_annotations=[self])
            for region in region_parts
        ]

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def contained_region_annotations(self) -> list["RegionAnnotation"]:
        if self.source_region_annotations is None:
            return [self]
        result = []
        for ra in self.source_region_annotations:
            result.extend(ra.contained_region_annotations)
        return sorted(result)

    @property
    def contained_samples(self) -> set[int]:
        """Return the indices of all data points inside the region."""
        return self.region.contained_samples

    @property
    def all_members(self) -> set[int]:
        """Return the indices of all data points that fulfill the rule."""
        return set(np.argwhere(self.descriptor.values).ravel())

    @property
    def contained_members(self) -> set[int]:
        """Return the indices of all data points that fulfill the rule inside the region."""
        return self.contained_samples & self.all_members

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.descriptor})"

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.descriptor, self.region))

    def __eq__(self, other: "RegionAnnotation") -> bool:
        if not isinstance(other, RegionAnnotation):
            return False
        return self.descriptor == other.descriptor and self.region == other.region

    def __lt__(self, other: "RegionAnnotation"):
        return self.descriptor < other.descriptor
