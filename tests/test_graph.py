import unittest

import vera.graph as g


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

