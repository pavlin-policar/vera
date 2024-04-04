# Visual Explanations via Region Annotation (VERA)

[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Installation

`vera` can be easily installed through pip using

```
pip install vera-explain
```

## A hello world example

Getting started with `vera` is very simple. First, we'll load up some data using scikit-learn.

```python
from sklearn import datasets

iris = datasets.load_iris()
x = iris["data"]
```

Next, we have to generate an embedding of the data. We'll use openTSNE here, but any embedding method will do.

```python
import openTSNE

embedding = openTSNE.TSNE().fit(x)
```

Then, we'll import and run the following commands to explain the embedding.

```python
import vera

features = vera.get_features(x)
contrastive_explanations = vera.explain.contrastive(features)
descriptive_explanations = vera.explain.descriptive(features)

vera.pl.plot_annotations(contrastive_explanations)
vera.pl.plot_annotations(descriptive_explanations)
```

## Citation

If you make use of `vera` for your work we would appreciate it if you would cite the paper

```
\article{Poli{\v c}ar2023
}
```
