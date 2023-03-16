import operator
from functools import reduce

import contourpy
import numpy as np
from shapely import geometry as geom


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

    @classmethod
    def from_embedding(
        cls,
        embedding: np.ndarray,
        values: np.ndarray,
        n_grid_points: int = 100,
        kernel: str = "gaussian",
        bw: float = 1,
    ):
        from KDEpy import FFTKDE
        kde = FFTKDE(kernel=kernel, bw=bw).fit(embedding, weights=values)
        grid, points = kde.evaluate(n_grid_points)

        return cls(grid, points)


class CompositeDensity(Density):
    def __init__(self, densities: list[Density]):
        self.base_densities = densities
        joint_density = np.sum(np.vstack([d.values for d in densities]), axis=0)
        grid = densities[0].grid
        super().__init__(grid, joint_density)
        if not all(np.allclose(d.grid, self.grid) for d in densities):
            raise RuntimeError(
                "All densities must have the same grid when constructing "
                "composite density!"
            )


class Region:
    def __init__(self, density: Density, polygon: geom.MultiPolygon):
        self.density = density
        self.polygon = self._ensure_multipolygon(polygon)

    @property
    def region_parts(self) -> list[geom.Polygon]:
        return self.polygon.geoms

    @property
    def num_parts(self) -> int:
        return len(self.region_parts)

    # @property
    # def plot_label(self) -> str:
    #     """The main label to be shown in a plot."""
    #     return str(self.feature)
    #
    # @property
    # def plot_detail(self) -> str:
    #     """Region details to be shown in a plot."""
    #     return None

    @staticmethod
    def _ensure_multipolygon(polygon) -> geom.MultiPolygon:
        if not isinstance(polygon, geom.MultiPolygon):
            polygon = geom.MultiPolygon([polygon])
        return polygon

    def get_contained_samples(self, embedding: np.ndarray) -> set[int]:
        """Get the indices of the contained samples within the region.s"""
        embedding_points = [geom.Point(p) for p in embedding]

        contained_indices = set()
        for i in range(len(embedding_points)):
            if self.polygon.contains(embedding_points[i]):
                contained_indices.add(i)

        return contained_indices

    def __add__(self, other: "Region") -> "CompositeRegion":
        if not isinstance(other, Region):
            raise NotImplementedError()
        return CompositeRegion([self, other])

    def __repr__(self):
        n = self.num_parts
        return f"{self.__class__.__name__}: {n} part{'s'[:n^1]}"

    @classmethod
    def from_density(cls, density: Density, level: float = 0.25):
        polygon = cls._ensure_multipolygon(density.get_polygons_at(level))
        return cls(density, polygon)


class CompositeRegion(Region):
    def __init__(self, regions: list[Region]):
        self.density = CompositeDensity([r.density for r in regions])

        self.base_regions = regions

        polygon = reduce(operator.or_, [r.polygon for r in regions])
        self.polygon = self._ensure_multipolygon(polygon)

    # @property
    # def plot_label(self) -> str:
    #     return str(self.feature)
    #
    # @property
    # def plot_detail(self) -> str:
    #     return "\n".join(str(f) for f in self.contained_features)
