# Visual Explanations and Contrastive Analysis (VECA)

[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Installation

`veca` can be easily installed through pip using

```
pip install veca
```

## A hello world example

Getting started with `veca` is very simple. First, we'll load up some data using scikit-learn

```python
from sklearn import datasets

iris = datasets.load_iris()
x = iris["data"]
```

then we'll import and run

```python
import veca

TODO
veca.run(x)
```

## Citation

If you make use of `veca` for your work we would appreciate it if you would cite the paper

```
\article{Poli{\v c}ar2023
}
```
