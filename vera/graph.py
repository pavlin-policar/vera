from collections import defaultdict
from itertools import permutations
from typing import NewType, TypeVar, Union

import networkx as nx
import scipy.sparse as sp

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.utils import check_random_state

T = TypeVar("T")
Graph = NewType("Graph", dict[T, set[T]])
UndirectedGraph = NewType("UndirectedGraph", Graph)
NodeList = NewType("NodeList", list[T])
EdgeList = NewType("EdgeList", list[tuple[T, T]])


def matrix_to_graph(mtx: sp.spmatrix) -> Graph:
    g = defaultdict(set)

    knng = mtx.tocsr()
    for i in range(len(mtx.indptr) - 1):
        g[i]  # access to ensure empty list
        for j in mtx.indices[knng.indptr[i] : mtx.indptr[i + 1]]:
            g[i].add(j)

    return Graph(dict(g))


def graph_to_edgelist(g: Graph) -> EdgeList:
    adjlist = []
    for i, edges in g.items():
        for j in edges:
            adjlist.append((i, j))
    return EdgeList(adjlist)


def edgelist_to_graph(v: NodeList, e: EdgeList) -> Graph:
    g = {}
    for i in v:
        g[i] = set()
    for i, j in e:
        g[i].add(j)
    return Graph(g)


def similarities_to_graph(similarities: np.ndarray, threshold: float = 0.9) -> Graph:
    """Create a graph connecting nodes where their similarity is above a given threshold."""
    if similarities.ndim == 1:
        similarities = squareform(similarities)
    elif similarities.ndim != 2:
        raise ValueError("`dists` can only be 1- or 2-dimensional!")

    similarities_sp = sp.csr_matrix(similarities > threshold)
    return matrix_to_graph(similarities_sp)


def distances_to_graph(dists: np.ndarray, threshold: float = 0.1) -> Graph:
    """Create a graph connecting nodes where their distance is below a given threshold."""
    if dists.ndim == 1:
        dists = squareform(dists)
    elif dists.ndim != 2:
        raise ValueError("`dists` can only be 1- or 2-dimensional!")

    dists_sp = sp.csr_matrix(dists < threshold)
    return matrix_to_graph(dists_sp)


def label_nodes(graph: Graph, labels: dict[T, T]) -> Graph:
    return {labels[i]: {labels[j] for j in graph[i]} for i in graph}


def nodes(g: Graph) -> NodeList:
    return list(g.keys())


def graph_complement(g: Graph) -> Graph:
    v = list(g.keys())
    e = graph_to_edgelist(g)
    e_inv = list(set(permutations(v, 2)) - set(e))
    return edgelist_to_graph(v, e_inv)


def to_undirected(g: Graph) -> UndirectedGraph:
    g_undirected = defaultdict(set)
    for i, v in g.items():
        g_undirected[i]  # access to ensure empty list
        for j in v:
            g_undirected[i].add(j)
            g_undirected[j].add(i)
    return UndirectedGraph(dict(g_undirected))


def degrees(g: Graph) -> dict[T, int]:
    return {v: len(e) for v, e in g.items()}


def merge_nodes(graph: Graph, i: T, j: T, new: T) -> Graph:
    node_mapping = {i: new, j: new}
    new_graph = defaultdict(set)
    for k, v in graph.items():
        new_graph[node_mapping.get(k, k)] |= {node_mapping.get(i, i) for i in v}
    return remove_self_loops(dict(new_graph))


def remove_self_loops(graph: Graph) -> Graph:
    new_graph = defaultdict(set)
    for k, v in graph.items():
        new_graph[k] |= {i for i in v if i != k}
    return dict(new_graph)


def configuration_graph(g: Graph, random_state) -> Graph:
    random_state = check_random_state(random_state)

    edges = graph_to_edgelist(g)
    stems = [i for sublist in edges for i in sublist]
    random_state.shuffle(stems)
    edges = list(zip(stems[::2], stems[1::2]))

    return edgelist_to_graph(nodes(g), edges)


def connected_components(g: Graph) -> list[Graph]:
    components = []
    remaining_nodes = set(g.keys())
    while len(remaining_nodes):
        v = next(iter(remaining_nodes))
        visited = set()
        stack = [v]
        while len(stack):
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                stack.extend(g[v])
        remaining_nodes -= visited
        components.append(visited)

    component_graphs = [
        Graph({k: v for k, v in g.items() if k in c}) for c in components
    ]
    component_graphs = sorted(component_graphs, key=len, reverse=True)

    return component_graphs


def max_cliques(g: Graph) -> list[Graph]:
    def _bron_kerbosch(g: dict, r: set, p: set, x: set):
        if len(p) == 0 and len(x) == 0:
            return [r]

        result = []
        pivot = max(p | x, key=lambda u: len(g[u] & p))
        for v in p - g[pivot]:
            result.extend(_bron_kerbosch(g, r | {v}, p & g[v], x & g[v]))
            p = p - {v}
            x = x | {v}

        return result

    cliques = _bron_kerbosch(g, set(), set(g.keys()), set())

    cliques = list(map(NodeList, map(list, cliques)))
    cliques = sorted(cliques, key=len, reverse=True)

    clique_graphs = [
        Graph({k: {vi for vi in v if vi in c} for k, v in g.items() if k in c})
        for c in cliques
    ]
    clique_graphs = sorted(clique_graphs, key=len, reverse=True)

    return clique_graphs


def max_cliques_nx(g: Graph) -> list[Graph]:
    g_nx = nx.from_dict_of_lists(g)
    cliques = list(nx.algorithms.clique.find_cliques(g_nx))

    cliques = list(map(NodeList, map(list, cliques)))
    cliques = sorted(cliques, key=len, reverse=True)

    clique_graphs = [
        Graph({k: {vi for vi in v if vi in c} for k, v in g.items() if k in c})
        for c in cliques
    ]
    clique_graphs = sorted(clique_graphs, key=len, reverse=True)

    return clique_graphs


def independent_sets(g: Graph) -> list[NodeList]:
    return list(map(nodes, max_cliques_nx(graph_complement(g))))


def graph_coloring_greedy(
    g: Graph, node_ordering: Union[NodeList, str] = "degree"
) -> dict[T, int]:
    def _next_available_color(g, colors, v):
        neighboring_colors = {colors[u] for u in g[v]}
        i = 0
        while i in neighboring_colors:
            i += 1
        return i

    if isinstance(node_ordering, str) and node_ordering == "degree":
        g_degrees = degrees(g)
        node_ordering = sorted(g_degrees, key=g_degrees.get)

    colors = {v: None for v in node_ordering}
    for v in g:
        colors[v] = _next_available_color(g, colors, v)

    return colors


def graph_coloring_greedy_nx(g: Graph, strategy: str = "largest_first"):
    g_nx = nx.from_edgelist(graph_to_edgelist(g))
    colors = nx.coloring.greedy_color(g_nx, strategy=strategy)
    return colors


def plot_graph(coords, e: EdgeList, vc="tab:blue", edge_alpha=0.25, ax=None):
    import matplotlib.pyplot as plt
    import matplotlib.collections as mcollections

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if isinstance(coords, dict):
        x_ = np.vstack(list(coords.values()))
        ax.scatter(x_[:, 0], x_[:, 1], zorder=2, s=48, c=vc)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], zorder=2, s=48, c=vc)

    lines = []
    for i, j in e:
        lines.append([coords[i], coords[j]])
    line_collection = mcollections.LineCollection(
        lines, zorder=1, color="k", alpha=edge_alpha
    )
    ax.add_collection(line_collection)

    return ax


def knng_k(x: np.ndarray, k_neighbors=5) -> UndirectedGraph:
    from sklearn import neighbors

    nn = neighbors.NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(x)
    mtx = nn.kneighbors_graph()
    mtx = mtx + mtx.T
    return UndirectedGraph(matrix_to_graph(mtx))


def knng_radius(x: np.ndarray, radius=1) -> UndirectedGraph:
    from sklearn import neighbors

    nn = neighbors.NearestNeighbors()
    nn.fit(x)
    indices = nn.radius_neighbors(radius=radius)[1]

    g: dict[int, set[int]] = defaultdict(set)
    for i in range(indices.shape[0]):
        g[i]  # access to ensure empty list
        for j in indices[i]:
            g[i].add(j)
            g[j].add(i)
    return UndirectedGraph(dict(g))
