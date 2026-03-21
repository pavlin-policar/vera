import operator
from functools import cached_property, reduce

import contourpy
import numpy as np
from shapely import geometry as geom
from shapely.ops import triangulate, unary_union

from vera.embedding import Embedding


def _max_edge_length(polygon: geom.Polygon) -> float:
    """Return the length of the longest edge of a polygon."""
    coords = polygon.boundary.coords
    return max(
        geom.LineString([p, q]).length
        for p, q in zip(coords[:-1], coords[1:])
    )


class Density:
    def __init__(self, grid: np.ndarray, values: np.ndarray):
        self.grid = grid
        self.values = values / values.sum()
        self.values_scaled = values / values.max()

    def _get_xyz(
        self, scaled: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_grid_points = int(np.sqrt(self.grid.shape[0]))  # always a square grid
        x, y = np.unique(self.grid[:, 0]), np.unique(self.grid[:, 1])
        vals = [self.values, self.values_scaled][scaled]
        z = vals.reshape(n_grid_points, n_grid_points).T
        return x, y, z

    def get_polygons_at(self, level: float) -> geom.MultiPolygon:
        x, y, z = self._get_xyz(scaled=True)

        contour_generator = contourpy.contour_generator(
            x,
            y,
            z,
            corner_mask=False,
            chunk_size=0,
            fill_type=contourpy.FillType.OuterOffset,
        )
        contours, chunks = contour_generator.filled(level, 1.01)  # up to >1

        polygons = []
        for contour, parts in zip(contours, chunks):
            geoms = [contour[i:j] for i, j in zip(parts, parts[1:])]
            # The first chunk is the contour, and should always be present, and
            # the remaining chunks are the holes
            polygon, *holes = geoms

            polygon = geom.Polygon(polygon, holes=holes)
            polygons.append(polygon)

        return geom.MultiPolygon(polygons)


class Region:
    def __init__(self, embedding: Embedding, polygon: geom.MultiPolygon):
        self.embedding = embedding
        self.polygon = self._ensure_multipolygon(polygon)

    @property
    def region_parts(self) -> list[geom.Polygon]:
        return self.polygon.geoms

    @property
    def num_parts(self) -> int:
        return len(self.region_parts)

    def split_into_parts(self):
        return [Region(self.embedding, p) for p in self.region_parts]

    @staticmethod
    def _ensure_multipolygon(polygon) -> geom.MultiPolygon:
        if not isinstance(polygon, geom.MultiPolygon):
            polygon = geom.MultiPolygon([polygon])
        return polygon

    @cached_property
    def contained_samples(self) -> set[int]:
        contained_indices = set()
        for i in range(len(self.embedding.points)):
            if self.polygon.covers(self.embedding.points[i]):
                contained_indices.add(i)

        return contained_indices

    def __eq__(self, other: "Region"):
        if not isinstance(other, Region):
            return False
        return self.polygon == other.polygon

    def __hash__(self):
        return hash((self.__class__.__name__, self.polygon))

    def __repr__(self):
        n = self.num_parts
        return f"{self.__class__.__name__}: {n} part{'s'[:n^1]}"

    @classmethod
    def from_density(cls, embedding: Embedding, density: Density, level: float = 0.25):
        polygon = cls._ensure_multipolygon(density.get_polygons_at(level))
        return cls(embedding, polygon)

    @classmethod
    def from_triangulation(
        cls,
        embedding: Embedding,
        member_mask: np.ndarray,
    ):
        """Construct a region via Delaunay triangulation with edge-length
        pruning, following the "rangeset" approach from:

            J.-T. Sohns, M. Schmitt, F. Jirasek, H. Hasse, and H. Leitte,
            "Attribute-based Explanation of Non-Linear Embeddings of
            High-Dimensional Data," IEEE VIS, 2021.

        Adapted from https://github.com/leitte/NoLiES

        Triangles whose longest edge exceeds ``2 * embedding.scale`` are
        pruned.  Use ``scale_factor`` on the :class:`Embedding` to control
        this threshold.

        Parameters
        ----------
        embedding : Embedding
        member_mask : np.ndarray
            Binary array indicating which points belong to this group.
        """
        active_idx = np.flatnonzero(member_mask)

        if len(active_idx) < 3:
            return cls(embedding, geom.MultiPolygon())

        threshold = embedding.scale * 2

        points = geom.MultiPoint(embedding.X[active_idx])
        triangles = triangulate(points)

        surviving = [
            t for t in triangles
            if _max_edge_length(t) < threshold
        ]

        if not surviving:
            return cls(embedding, geom.MultiPolygon())

        polygon = unary_union(surviving)
        return cls(embedding, polygon)

    @classmethod
    def merge(cls, regions: list["Region"], merge_method: str = "union"):
        # Ensure that all regions belong to the same embedding, otherwise
        # merging them makes little sense
        embedding = regions[0].embedding
        if not all(r.embedding == embedding for r in regions[1:]):
            raise RuntimeError(
                "Merged regions must originate belong to the same embedding!"
            )

        if merge_method == "union":
            merged_polygon = reduce(operator.or_, [r.polygon for r in regions])
        elif merge_method == "intersection":
            merged_polygon = reduce(operator.and_, [r.polygon for r in regions])
        else:
            raise ValueError(
                "Unsupported `merge_method`. Supported methods are `union` and "
                "`interscection`."
            )

        return cls(embedding=embedding, polygon=merged_polygon)
