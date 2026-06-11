import unittest

import vera.graph as g


class IdentityHashedNode:
    """Mimics RegionAnnotation-like graph nodes whose hash depends on the
    object's memory address, which varies between runs and processes."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name


# The Petersen graph is vertex-transitive and 3-regular, so the algorithms
# under test have no structural grounds to prefer one node over another. Every
# tie is then broken by iteration order alone, making hash-order dependence
# observable as a difference in output order.
PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
]


def build_petersen_graph():
    """Build the Petersen graph on fresh identity-hashed node objects.

    Each call allocates new node objects at new memory addresses, but the
    graph's insertion order is always the same, so any two builds are
    logically equal inputs."""
    nodes = [IdentityHashedNode(f"n{i}") for i in range(10)]
    edges = [(nodes[i], nodes[j]) for i, j in PETERSEN_EDGES]
    return g.to_undirected(g.edgelist_to_graph(nodes, edges))


class TestOutputOrderDeterminism(unittest.TestCase):
    """Graph algorithm output order must not depend on node object hashes.
    Downstream panel layout selection breaks ties by candidate order, so the
    order these return is user-visible."""

    def names(self, node_lists):
        return [[n.name for n in nodes] for nodes in node_lists]

    def assertSameOutputOrder(self, func):
        results = [self.names(func(build_petersen_graph())) for _ in range(5)]
        for result in results[1:]:
            self.assertEqual(results[0], result)

    def test_max_cliques(self):
        self.assertSameOutputOrder(lambda graph: map(g.nodes, g.max_cliques(graph)))

    def test_max_cliques_nx(self):
        self.assertSameOutputOrder(lambda graph: map(g.nodes, g.max_cliques_nx(graph)))

    def test_independent_sets(self):
        self.assertSameOutputOrder(g.independent_sets)

    def test_connected_components(self):
        self.assertSameOutputOrder(
            lambda graph: map(g.nodes, g.connected_components(graph))
        )

    def test_graph_complement(self):
        # Adjacencies are sets, so only node order and logical edges are
        # deterministic; consumers relabel before iterating adjacency sets
        def complement_adjacency(graph):
            complement = g.graph_complement(graph)
            return [
                (k.name, sorted(n.name for n in v)) for k, v in complement.items()
            ]

        results = [complement_adjacency(build_petersen_graph()) for _ in range(5)]
        for result in results[1:]:
            self.assertEqual(results[0], result)

    def test_graph_coloring_greedy_nx(self):
        def color_groups(graph):
            colors = g.graph_coloring_greedy_nx(graph)
            groups = {}
            for node, color in colors.items():
                groups.setdefault(color, []).append(node)
            return groups.values()

        self.assertSameOutputOrder(color_groups)


class TestMergeNodes(unittest.TestCase):
    def test_merge_nodes(self):
        graph = [(0, 1), (1, 2), (2, 3), (3, 4)]
        graph = g.edgelist_to_graph(range(5), graph)
        merged_graph = g.merge_nodes(graph, 1, 2, new=99)
        self.assertEqual({0: {99}, 99: {3}, 3: {4}, 4: set()}, merged_graph)


class TestMaxCliquesApprox(unittest.TestCase):
    def test_1(self):
        graph = [(0, 1), (1, 2), (2, 3), (3, 4)]
        graph = g.edgelist_to_graph(range(5), graph)
        g.max_cliques_nx(graph)

        tmp = g.graph_coloring_greedy_nx(graph)
        from collections import defaultdict
        groups = defaultdict(set)
        for k, v in tmp.items():
            groups[v].add(k)
        return list(groups.values())

