import operator
from functools import reduce
from typing import Any

import contourpy
import numpy as np
from shapely import geometry as geom

from embedding_annotation.data import Variable


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

    def get_contours_at(self, level: float) -> list[np.ndarray]:
        x, y, z = self._get_xyz(scaled=True)

        contour_generator = contourpy.contour_generator(
            x, y, z, corner_mask=False, chunk_size=0
        )
        return contour_generator.lines(level)

    def get_polygons_at(self, level: float) -> geom.MultiPolygon:
        polygons = [geom.Polygon(c) for c in self.get_contours_at(level)]
        # Ensure the proper handling of holes
        # TODO: This probably doesn't work
        result = polygons[0]
        for p in polygons[1:]:
            if result.contains(p):
                result -= p
            else:
                result |= p

        return result

    def __add__(self, other: "Density") -> "CompositeDensity":
        if not isinstance(other, Density):
            raise ValueError(
                f"Cannot merge `{self.__class__.__name__}` with object of type "
                f"`{other.__class__.__name__}`"
            )
        return CompositeDensity([self, other])


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
    def __init__(self, feature: Variable, density: Density, level: float = 0.25):
        self.feature = feature
        self.level = level
        self.density = density

        self.polygon = self._ensure_multipolygon(density.get_polygons_at(level))

    @property
    def region_parts(self):
        return self.polygon.geoms

    @property
    def num_parts(self):
        return len(self.region_parts)

    @property
    def plot_label(self) -> str:
        """The main label to be shown in a plot."""
        return str(self.feature)

    @property
    def plot_detail(self) -> str:
        """Region details to be shown in a plot."""
        return None

    @staticmethod
    def _ensure_multipolygon(polygon):
        if not isinstance(polygon, geom.MultiPolygon):
            polygon = geom.MultiPolygon([polygon])
        return polygon

    def __add__(self, other: "Region"):
        if not np.allclose(self.density.grid, other.density.grid):
            raise RuntimeError("Grids must match when adding two density objects")

        return CompositeRegion(
            [self.feature, other.feature], [self.density, other.density]
        )

    def __repr__(self):
        n = self.num_parts
        return f"Region: `{str(self.feature)}`, {n} part{'s'[:n^1]}"

    def __eq__(self, other: "Region") -> bool:
        """We will check for equality only on the basis of the variable."""
        if not isinstance(other, Region):
            return False
        return self.feature == other.feature

    def __hash__(self):
        """Hashing only on the basis of the variable."""
        return hash(self.feature)

    @property
    def contained_features(self) -> list[Variable]:
        """Return all the features contained within this region"""
        return [self.feature]


class CompositeRegion(Region):
    def __init__(self, feature: str | list[Any], regions: list[Region]):
        self.feature = feature or " + ".join(str(r.feature) for r in regions)
        self.level = regions[0].level
        if not all(r.level == self.level for r in regions):
            raise RuntimeError(
                "All regions must have the same level when constructing "
                "composite region!"
            )
        self.density = CompositeDensity([r.density for r in regions])

        self.base_regions = regions

        polygon = reduce(operator.or_, [r.polygon for r in regions])
        self.polygon = self._ensure_multipolygon(polygon)

    @property
    def plot_label(self) -> str:
        return str(self.feature)

    @property
    def plot_detail(self) -> str:
        return "\n".join(str(f) for f in self.contained_features)

    @property
    def contained_features(self) -> list[Variable]:
        return reduce(operator.add, [r.contained_features for r in self.base_regions])
