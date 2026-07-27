import json
import os
import subprocess
import sys
import unittest

import pandas as pd

import vera

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "iris")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_iris():
    """Load the vendored iris fixture: features and a precomputed 2D embedding."""
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    embedding = pd.read_csv(
        os.path.join(DATA_DIR, "embedding.csv"), header=None
    ).values
    return features, embedding


class ExplainTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features, cls.embedding = load_iris()
        cls.region_annotations = vera.an.generate_region_annotations(
            cls.features,
            cls.embedding,
            n_discretization_bins=5,
            scale_factor=1,
            sample_size=5000,
            contour_level=0.25,
            merge_min_sample_overlap=0.5,
            random_state=0,
        )


class TestContrastiveRanking(ExplainTestBase):
    def test_1(self):
        layouts = vera.explain.contrastive(self.region_annotations, max_panels=2)

        self.assertEqual(2, len(layouts))


class TestDescriptiveRanking(ExplainTestBase):
    def test_1(self):
        layouts = vera.explain.descriptive(self.region_annotations, max_panels=2)

        self.assertEqual(2, len(layouts))


def _run_descriptive_pipeline(features, embedding):
    region_annotations = vera.an.generate_region_annotations(
        features,
        embedding,
        n_discretization_bins=5,
        scale_factor=1,
        sample_size=5000,
        contour_level=0.25,
        merge_min_sample_overlap=0.5,
        random_state=0,
    )
    layout = vera.explain.descriptive(region_annotations, max_panels=2)
    return [[repr(ra) for ra in panel] for panel in layout]


class TestDescriptiveLayoutDeterminism(unittest.TestCase):
    """Layouts must not depend on object hashes, which vary with object memory
    addresses (between runs in one process) and with PYTHONHASHSEED (between
    processes)."""

    def test_identical_layouts_for_equal_inputs_in_one_process(self):
        features, embedding = load_iris()

        layouts = [
            _run_descriptive_pipeline(features.copy(), embedding.copy())
            for _ in range(2)
        ]

        self.assertEqual(layouts[0], layouts[1])

    def test_identical_layouts_across_processes_with_different_hash_seeds(self):
        # Runs the pipeline on the iris fixture, printing the layout as JSON
        # on the last line of stdout
        script = """
import json
import vera
from tests.test_explain import load_iris, _run_descriptive_pipeline

features, embedding = load_iris()
print(json.dumps(_run_descriptive_pipeline(features, embedding)))
"""

        python_path = os.pathsep.join(
            p for p in [PROJECT_ROOT, os.environ.get("PYTHONPATH")] if p
        )

        layouts = []
        for hash_seed in ["0", "1", "4242"]:
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=python_path)
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                0, result.returncode, f"Subprocess failed:\n{result.stderr}"
            )
            layouts.append(json.loads(result.stdout.strip().splitlines()[-1]))

        self.assertEqual(layouts[0], layouts[1])
        self.assertEqual(layouts[0], layouts[2])
