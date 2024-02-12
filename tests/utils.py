import numpy as np
import pandas as pd


def generate_clusters(means: list, stds: list, n_samples: int):
    assert len(means) == len(stds)
    datas, clusters = [], []
    for idx, (mu, sigma) in enumerate(zip(means, stds)):
        x0 = np.random.normal(mu, sigma, size=(n_samples, 2))
        y0 = idx * np.ones(n_samples, dtype=int)
        datas.append(x0), clusters.append(y0)

    x = np.vstack(datas)
    features = pd.DataFrame()
    features["cluster"] = np.hstack(clusters)
    features["cluster"] = features["cluster"].astype("category")

    assert x.shape[0] == features.shape[0]  # sanity check

    return x, features
