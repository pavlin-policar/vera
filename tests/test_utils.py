import unittest

from vera.utils import flatten


class TestFlatten(unittest.TestCase):
    def test_flatten_1(self):
        xs = [1, 2, 3, 4, 5]
        flattened = flatten(xs)
        print(flattened)
        self.assertEqual(
            xs,
            flattened,
        )

    def test_flatten_2(self):
        xs = [[1], 2, [[3, 4]], 5]
        flattened = flatten(xs)
        print(flattened)
        self.assertEqual(
            [1, 2, 3, 4, 5],
            flattened,
        )
