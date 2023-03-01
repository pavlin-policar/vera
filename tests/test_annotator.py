import os.path
import shutil
import unittest
from dataclasses import dataclass
from os import path

import numpy as np
import openTSNE
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import KBinsDiscretizer

import embedding_annotation as annotate

DATA_DIR = "data"


@dataclass
class TestDataset:
    name: str
    embedding: np.ndarray
    features: pd.DataFrame

    @property
    def data_dir(self):
        return path.join(DATA_DIR, self.name)

    @classmethod
    def load(cls, name):
        embedding_fname = path.join(DATA_DIR, name, "embedding.csv")
        features_fname = path.join(DATA_DIR, name, "features.csv")
        if not path.exists(embedding_fname):
            raise FileNotFoundError(embedding_fname)
        if not path.exists(features_fname):
            raise FileNotFoundError(features_fname)

        embedding = pd.read_csv(embedding_fname, header=None).categories
        features = pd.read_csv(features_fname)

        return cls(name, embedding, features)

    def save(self, force=False):
        if os.path.exists(self.data_dir):
            if force:
                shutil.rmtree(self.data_dir)
            else:
                raise FileExistsError(self.data_dir)

        os.mkdir(self.data_dir)

        pd.DataFrame(self.embedding).to_csv(
            path.join(self.data_dir, "embedding.csv"), header=False, index=False
        )
        self.features.to_csv(path.join(self.data_dir, "features.csv"), index=False)


def prepare_iris():
    iris = datasets.load_iris()
    x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    features = annotate.d.generate_explanatory_variables(x)

    embedding = openTSNE.TSNE(metric="cosine", perplexity=30).fit(x.values)

    return TestDataset("iris", embedding, features)


class TestAnnotator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.iris = TestDataset.load("iris")
        except FileNotFoundError:
            cls.iris = prepare_iris()
            cls.iris.save(force=True)

    def test_bla(self):
        embedding, features = self.iris.embedding, self.iris.features
        candidates = annotate.fs.morans_i(embedding, features)
        feature_densities = annotate.an.estimate_feature_densities(
            candidates["feature"].tolist(),
            embedding,
            features,
        )

        clusters, cluster_densities = annotate.an.group_similar_features(
            candidates["feature"].tolist(),
            feature_densities,
            threshold=0.85,
            method="connected-components",
        )

        annotate.pl.plot_feature_densities(
            list(clusters),
            cluster_densities,
            embedding=embedding,
            levels=4,
            skip_first=False,
            per_row=1,
            figwidth=4,
        )
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.show()


