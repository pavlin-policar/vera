# Embedding Annotator

[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Installation

`embedding_annotator` can be easily installed through pip using

```
pip install embedding_annotator
```

## A hello world example

Getting started with `embedding_annotator` is very simple. First, we'll load up some data using scikit-learn

```python
from sklearn import datasets

iris = datasets.load_iris()
x = iris["data"]
```

then we'll import and run

```python
import embedding_annotation as annotate

TODO
annotate.an.run(x)
```

## Citation

If you make use of `embedding_annotation` for your work we would appreciate it if you would cite the paper

```
\article{Poli{\v c}ar2023
}
```
