# Visual Explanations via Region Annotation (VERA)

[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

> [!warning]
> This library is under active development. While the core algorithm and code are mostly stable, the plotting utilities will certainly change. I still haven't found a satisfactory way to do label placement, so this will definitely change.

VERA automatically generates visual explanations of two-dimensional embeddings. When dealing with high-dimensional data, a common first step is to generate a two-dimensional embedding using your dimensionality reduction algorithm of choice (t-SNE, UMAP, PCA) and visualize it in a scatter plot. However, once we have a visualization, the next step typically involves figuring out what each particular region or cluster in the embedding corresponds to. This is often tedious and requires us to tinker around in notebooks or interactive tools to figure out what's what.

The aim of VERA is to automate this process and automatically generate a series of potentially interesting explanatory visualizations, each of which explain different regions of the embedding in terms of the original data features. This way, users can quickly get a big-picture overview of the two-dimensional embedding without hassle.

- [Documentation]() (TODO)
- [User Guide and Tutorial]() (TODO)
- [Preprint](https://arxiv.org/abs/2406.04808)

![](docs/source/images/main-example.png)

A few caveats and limitations:
- VERA generates explanations using a provided set of features, which should be interpretable to the user.
- Currently, VERA is limited when dealing with large numbers of features. This can be both slow to run, and can generate very long region descriptions, which oftentimes end up unreadable.

While the API and core algorithm is stable, VERA is under **active development** and some its behaviour may change slightly in subsequent versions.
The following things are still on my todo list:
- better label placement
- better scalability
- label formatting and summarization

## Installation

`vera` can be easily installed through pip using

```
pip install vera-explain
```

[PyPI package](https://pypi.org/project/vera-explain/0.1.0/)

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

region_annotations = vera.an.generate_region_annotations(x, embedding)
contrastive_explanations = vera.explain.contrastive(region_annotations)
descriptive_explanations = vera.explain.descriptive(region_annotations)

vera.pl.plot_annotations(contrastive_explanations)
vera.pl.plot_annotations(descriptive_explanations)
```

## Citation

If you make use of `vera` for your work we would appreciate it if you would cite the [paper](https://arxiv.org/abs/2406.04808):

```
\article{Policar2024
  title={VERA: Generating Visual Explanations of Two-Dimensional Embeddings via Region Annotation}, 
  author={Pavlin G. Poličar and Blaž Zupan},
  year={2024},
  eprint={2406.04808},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
