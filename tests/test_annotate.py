import unittest

from embedding_annotation.annotate import AnnotationMap
import numpy as np


class TestAnnotate(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = np.vstack(list(map(np.ravel, np.mgrid[:5, :5]))).T
        self.embedding = np.random.randn(10, 2)
        self.densities = np.random.randn(5, self.grid.shape[0])

    def test_adding_and_removing_from_annmap(self):
        # Add several densities
        ann0 = AnnotationMap(self.grid, self.embedding)
        self.assertEquals(0, len(ann0))

        ann1 = ann0.add("Density 1", self.densities[0])
        self.assertEquals(0, len(ann0))
        self.assertEquals(1, len(ann1))

        ann2 = ann1.add("Density 2", self.densities[1])
        self.assertEquals(0, len(ann0))
        self.assertEquals(1, len(ann1))
        self.assertEquals(2, len(ann2))

        ann3 = ann1.add("Density 3", self.densities[2]).add("Density 4", self.densities[3])
        self.assertEquals(0, len(ann0))
        self.assertEquals(1, len(ann1))
        self.assertEquals(2, len(ann2))
        self.assertEquals(3, len(ann3))

        # Remove densities
        ann4 = ann1.remove("Density 1")
        self.assertEquals(0, len(ann4))

        ann5 = ann3.remove("Density 1").remove("Density 3")
        self.assertEquals(1, len(ann5))

    def test_immutable_annmap_objects(self):
        ann0 = AnnotationMap(self.grid, self.embedding)
        ann1 = ann0.add("Density 1", self.densities[0])
        ann2 = ann1.add("Density 2", self.densities[1])

        self.assertIsNot(ann0, ann1)
        self.assertIsNot(ann1, ann2)

        ann3 = ann2.remove("Density 1")
        ann4 = ann3.remove("Density 2")
        self.assertIsNot(ann2, ann3)
        self.assertIsNot(ann3, ann4)

    def test_immutability_of_grid(self):
        ann0 = AnnotationMap(self.grid, self.embedding)
        with self.assertRaises(ValueError):
            ann0.grid[0, 0] = 5

        orig_value = self.grid[0, 0]
        self.grid[0, 0] = 5
        self.assertNotEqual(ann0.grid[0, 0], 5)
        self.assertEqual(ann0.grid[0, 0], orig_value)

        ann1 = ann0.add("Density 1", self.densities[0])
        # Ensure grid isn't copied
        self.assertIs(ann0.grid, ann1.grid)

    def test_immutability_of_embedding(self):
        ann0 = AnnotationMap(self.grid, self.embedding)
        with self.assertRaises(ValueError):
            ann0.embedding[0, 0] = 5

        orig_value = self.embedding[0, 0]
        self.embedding[0, 0] = 5
        self.assertNotEqual(ann0.embedding[0, 0], 5)
        self.assertEqual(ann0.embedding[0, 0], orig_value)

        ann1 = ann0.add("Density 1", self.densities[0])
        # Ensure embedding isn't copied
        self.assertIs(ann0.embedding, ann1.embedding)

    def test_immutability_of_densities(self):
        densities = {f"Density {i}": self.densities[i] for i in range(self.densities.shape[0])}
        ann = AnnotationMap(self.grid, self.embedding, densities)
        with self.assertRaises(TypeError):
            ann.densities["foo"] = "bar"

        with self.assertRaises(TypeError):
            ann.densities["Density 0"] = "bar"

        with self.assertRaises(ValueError):
            ann.densities["Density 0"][0, :] = 5

        # Ensure densities aren't copied
        ann0 = AnnotationMap(self.grid, self.embedding, densities)
        ann1 = ann0.add("Density X", np.random.randn(self.grid.shape[0]))

        k0 = list(ann.densities.keys())[0]
        self.assertIs(ann0.densities[k0], ann1.densities[k0])
