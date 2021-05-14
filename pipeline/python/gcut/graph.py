from collections import defaultdict
import numpy as np
from pipeline_util import timer
from .topology import TopologicalGraph


class BFS:
    def __init__(self):
        self._source_soma_id = None
        # tuple(topo_id0, topo_id1), where topo_id0 < topo_id1
        self.connected_topo_ids = set()
        self.connected_neurites = set()
        # topological nodes along major path connecting two somas, empty
        # if only one soma in the bfs graph
        self.freeway_topo_ids = set()
        self.freeway_neurites = set()
        # ramp neurite: {set of unordered neurites rooted at a ramp neurite
        #                off freeway, including the ramp neurite}
        # all post ramp nodes must have same soma assignment as the
        # ramp node. the orientation is forced to be
        # freeway parent node -> ramp node
        self.ramps = defaultdict(set)
        self._parent = {}
        self._soma_targets = set()

    def clear(self):
        self._source_soma_id = None
        self.connected_topo_ids.clear()
        self.connected_neurites.clear()
        self.freeway_topo_ids.clear()
        self.freeway_neurites.clear()
        self.ramps.clear()
        self._parent.clear()
        self._soma_targets.clear()

    # (ramp entrance node, freeway node) neurites
    def ramp_neurites(self):
        return set([ramp_neurite for ramp_neurite in self.ramps.keys()])

    # ramp neurites and free way neurites. these neurites participate in
    # the linear programming problem
    def decision_neurites(self):
        return self.freeway_neurites | self.ramp_neurites()

    # if a soma is encountered, the graph search is trapped by it. no edge can
    # leave the non source soma
    def bfs(self, source_id, topo_graph):
        assert isinstance(topo_graph, TopologicalGraph)
        assert topo_graph.is_topological_node(source_id)
        # first in first out. append in
        frontier = [source_id]
        self.connected_topo_ids.add(source_id)
        while len(frontier) > 0:
            active_id = frontier.pop(0)
            # add topo neighbors reacheable by neurites leaving the active node
            for out_topo_id in topo_graph.graph[active_id].out_neurites:
                if out_topo_id in self.connected_topo_ids:
                    continue
                if not topo_graph.is_soma_node(out_topo_id):
                    frontier.append(out_topo_id)
                else:
                    self._soma_targets.add(out_topo_id)
                self._parent[out_topo_id] = active_id
                self.connected_topo_ids.add(out_topo_id)
                self.connected_neurites.add(TopologicalGraph.undirected_hash(active_id, out_topo_id))

    def build_freeways(self, topo_graph):
        if len(self._soma_targets) == 0:
            print('no freeway paths')
            return
        # trace back from target soma for highway nodes
        for soma_target in self._soma_targets:
            self.freeway_topo_ids.add(soma_target)
            parent = self._parent[soma_target]
            while parent is not None:
                self.freeway_topo_ids.add(parent)
                self.freeway_neurites.add(TopologicalGraph.undirected_hash(soma_target, parent))
                soma_target = parent
                parent = self._parent[soma_target]
        # ramp entrance nodes:
        # they are not on the high way but their parents are
        entrance_candidates = self.connected_topo_ids - self.freeway_topo_ids
        entrance_ids = set()
        for entrance_candidate in entrance_candidates:
            if self._parent[entrance_candidate] in self.freeway_topo_ids:
                entrance_ids.add(entrance_candidate)
                ramp = TopologicalGraph.undirected_hash(entrance_candidate,
                                                        self._parent[entrance_candidate])
                self.ramps[ramp] = {ramp}
        # run some smaller bfs to find off ramp neurites rooted at each
        # ramp entrance node
        for ramp in self.ramps:
            ramp_entrance_id = ramp[0] if ramp[0] in entrance_ids else ramp[1]
            self.bfs_off_freeway(ramp, ramp_entrance_id, topo_graph)

        # sanity check
        assert len(self.ramp_neurites() & self.freeway_neurites) == 0
        off_freeway_neurites = set()
        for ramp_neurites in self.ramps.values():
            assert len(ramp_neurites & off_freeway_neurites) == 0
            off_freeway_neurites |= ramp_neurites
        assert len(off_freeway_neurites & self.freeway_neurites) == 0
        assert len(self._soma_targets - self.freeway_topo_ids) == 0
        assert self.connected_neurites == off_freeway_neurites | self.freeway_neurites

    def bfs_off_freeway(self, ramp, ramp_entrance_id, topo_graph):
        off_freeway_nodes = set()
        off_freeway_neurites = set()
        assert isinstance(topo_graph, TopologicalGraph)
        assert topo_graph.is_topological_node(ramp_entrance_id)
        # first in first out. append in
        frontier = [ramp_entrance_id]
        off_freeway_nodes.add(ramp_entrance_id)
        while len(frontier) > 0:
            active_id = frontier.pop(0)
            # add topo neighbors reacheable by neurites leaving the active node
            for out_topo_id in topo_graph.graph[active_id].out_neurites:
                if not out_topo_id in off_freeway_nodes and \
                        not out_topo_id in self.freeway_topo_ids:
                    frontier.append(out_topo_id)
                    off_freeway_nodes.add(out_topo_id)
                    off_freeway_neurites.add(TopologicalGraph.undirected_hash(active_id, out_topo_id))
        self.ramps[ramp].update(off_freeway_neurites)

    # topological nodes reachable from the neurite.
    # at least one of the neurite end node must be a soma
    @timer
    def bfs_neurite(self, neurite, topo_graph):
        assert isinstance(neurite, tuple)
        # if the two nodes are (soma, soma) or (soma, leaf), result is obtained
        # without running the graph search
        n_somas = int(topo_graph.is_soma_node(neurite[0])) + int(topo_graph.is_soma_node(neurite[1]))
        assert n_somas >= 1
        n_leaf = int(topo_graph.is_leaf_node(neurite[0])) + int(topo_graph.is_leaf_node(neurite[1]))
        if n_somas == 2 or n_leaf > 0:
            self.connected_topo_ids.add(neurite[0])
            self.connected_topo_ids.add(neurite[1])
            self.connected_neurites.add(neurite)
            if n_somas == 2:
                self.freeway_topo_ids.update(neurite)
                self.freeway_neurites.add(TopologicalGraph.undirected_hash(neurite[0], neurite[1]))
        # otherwise, perform bfs from the edge soma->branch
        else:
            self._source_soma_id = neurite[0] if topo_graph.is_soma_node(neurite[0]) else neurite[1]
            branch_id = neurite[0] if topo_graph.is_branch_node(neurite[0]) else neurite[1]
            self.connected_topo_ids.add(self._source_soma_id)
            self.connected_neurites.add(TopologicalGraph.undirected_hash(self._source_soma_id, branch_id))
            self._parent[self._source_soma_id] = None
            self._parent[branch_id] = self._source_soma_id
            self.freeway_topo_ids.add(self._source_soma_id)
            # start bfs from branch node, so other neurites originating from
            # the source soma will not be reached
            self.bfs(branch_id, topo_graph)
            self.build_freeways(topo_graph)


class Dijkstra:
    class Node:
        def __init__(self, node_id, d=None, u=None, n_edges=None, parent_id=None):
            self.node_id = node_id
            assert self.node_id >= 0
            # distance from source to node_id
            self.d = np.inf if d is None else d
            # edge cost from parent to node_id
            self.u = np.inf if u is None else u
            assert self.d >= 0 and self.u >= 0
            # number of edges from source
            self.n_edges = np.inf if n_edges is None else n_edges
            self.parent_id = parent_id
            if self.parent_id is not None:
                assert self.parent_id >= 0

        def __lt__(self, other):
            assert isinstance(other, Dijkstra.Node)
            return self.d < other.d

        def __gt__(self, other):
            assert isinstance(other, Dijkstra.Node)
            return self.d > other.d

        def __le__(self, other):
            assert isinstance(other, Dijkstra.Node)
            return self.d <= other.d

        def __ge__(self, other):
            assert isinstance(other, Dijkstra.Node)
            return self.d >= other.d

    # min heap. support push, pop and element updates.
    # no heapify of non heap array is implemented
    # 1 indexing. self.queue[0] = None.
    # heap element number = len(self.queue) - 1
    # heap invariant: heap[k] <= heap[2k] and heap[k] <= heap[2k+1]
    class Heap:
        def __init__(self):
            self.queue = [None]
            # node_id: queue position
            self.queue_positions = {}

        def empty(self):
            return len(self.queue) == 1

        def clear(self):
            self.queue = [None]
            self.queue_positions.clear()

        def n_elements(self):
            return len(self.queue) - 1

        def push(self, node):
            assert isinstance(node, Dijkstra.Node)
            self.queue.append(node)
            self.queue_positions[node.node_id] = len(self.queue) - 1
            if self.n_elements() == 1:
                return
            self._swim_up(node)

        def pop(self):
            if self.empty():
                return None
            root_id = self._root_id()
            root_node = self.queue[1]
            self._swap(self.queue[1], self.queue[-1])
            self.queue.pop(-1)
            self.queue_positions.pop(root_id)
            if self.n_elements() > 1:
                self._swim_down(self.queue[1])
            return root_node

        # if node is not in in heap, make a push call
        # if node is already in heap, update its d value and swim
        def update(self, node):
            assert isinstance(node, Dijkstra.Node)
            if node.node_id not in self.queue_positions:
                self.push(node)
            else:
                node_pos = self.queue_positions[node.node_id]
                if self.queue[node_pos].d > node.d:
                    self.queue[node_pos] = node
                    self._swim_up(self.queue[node_pos])

        def _root_id(self):
            if self.empty():
                return None
            return self.queue[1].node_id

        def _swap(self, node0, node1):
            assert isinstance(node0, Dijkstra.Node) and isinstance(node1, Dijkstra.Node)
            # swap queue elements
            node0_pos, node1_pos = self.queue_positions[node0.node_id], self.queue_positions[node1.node_id]
            node0_id, node1_id = node0.node_id, node1.node_id
            self.queue[node0_pos], self.queue[node1_pos] = \
                self.queue[node1_pos], self.queue[node0_pos]
            # swap queue_positions entries
            self.queue_positions[node0_id], self.queue_positions[node1_id] = \
                self.queue_positions[node1_id], self.queue_positions[node0_id]

        def _parent(self, node):
            node_pos = self.queue_positions[node.node_id]
            parent_pos = node_pos // 2
            if parent_pos == 0:
                return None
            return self.queue[parent_pos]

        def _swim_up(self, node):
            parent_node = self._parent(node)
            while parent_node is not None and node < parent_node:
                self._swap(node, parent_node)
                parent_node = self._parent(node)

        def _smaller_child(self, node):
            node_pos = self.queue_positions[node.node_id]
            child0_pos, child1_pos = 2 * node_pos, 2 * node_pos + 1
            if child0_pos > self.n_elements():
                return None
            elif child1_pos > self.n_elements():
                return self.queue[child0_pos]
            else:
                return min(self.queue[child0_pos], self.queue[child1_pos])

        def _swim_down(self, node):
            child_node = self._smaller_child(node)
            while child_node is not None and not node < child_node:
                self._swap(node, child_node)
                child_node = self._smaller_child(node)

        def verify(self):
            for i in range(2, self.n_elements() + 1):
                assert self.queue[i // 2] <= self.queue[i]

    def __init__(self):
        self.current_id, self.source_id = None, None
        self.target_ids = None
        # id in topo_graph: Dijkstra.Node instance
        self.nodes = {}
        # target_id: [target_id <- paren(target_id) <- ... <- source_id]
        self._shortest_paths = defaultdict(list)
        self._heap = Dijkstra.Heap()
        self._assigned = set()

    def clear(self):
        self.current_id, self.source_id = None, None
        self.nodes.clear()
        self._shortest_paths.clear()
        self._heap.clear()
        self._assigned.clear()

    def init(self, src_id, target_ids, topo_graph):
        assert isinstance(topo_graph, TopologicalGraph)
        assert topo_graph.is_soma_node(src_id)
        for target_id in target_ids:
            assert topo_graph.is_topological_node(target_id)
        self.source_id = src_id
        self.nodes[self.source_id] = Dijkstra.Node(src_id, d=0, u=0, n_edges=0, parent_id=src_id)
        self._heap.update(self.nodes[self.source_id])
        self.target_ids = set(target_ids)
        # remove source node from target node set
        self.target_ids.remove(self.source_id)
        for target_id in self.target_ids:
            self.nodes[target_id] = Dijkstra.Node(target_id)

    @timer
    def dijkstra(self, src_id, target_ids, topo_graph):
        self.init(src_id, target_ids, topo_graph)
        # continue till all target nodes are visited
        while not self._heap.empty():
            self.current_id = self._heap.pop().node_id
            for neighbor_id, edge_id in topo_graph.graph[self.current_id].out_neurites.items():
                # exclude nodes not within target set
                if neighbor_id not in target_ids:
                    continue
                if neighbor_id in self._assigned:
                    continue
                neighbor_order = self.nodes[self.current_id].n_edges + 1
                edge_cost = topo_graph.neurites[edge_id].gof_cost(src_id, neighbor_order)
                if self.nodes[self.current_id].d + edge_cost < self.nodes[neighbor_id].d:
                    self.nodes[neighbor_id].d = self.nodes[self.current_id].d + edge_cost
                    self.nodes[neighbor_id].u = edge_cost
                    self.nodes[neighbor_id].parent_id = self.current_id
                    self.nodes[neighbor_id].n_edges = self.nodes[self.current_id].n_edges + 1
                self._heap.update(self.nodes[neighbor_id])
            self._assigned.add(self.current_id)

    def construct_shortest_paths(self):
        for target_id, target_node in self.nodes.items():
            path_id = target_id
            while path_id != self.source_id:
                self._shortest_paths[target_id].append(path_id)
                path_id = self.nodes[path_id].parent_id
            self._shortest_paths[target_id].append(self.source_id)

    def shortest_paths(self):
        if len(self._shortest_paths) == 0:
            self.construct_shortest_paths()
        return self._shortest_paths

    # the set of (child_neurite, parent_neurite) pair constructed
    # from self._shortest_paths. each (child_neurite, parent_neurite) pair
    # consumes 3 topological nodes
    # if neurite_subset is not None, only include neurites from the set
    def neurite_lineages(self, ramps, neurite_subset=None):
        neurite_pair_orderings = set()
        for target_id, target_src_path in self.shortest_paths().items():
            # do nothing for paths with less than 3 nodes.
            # these include path from source node to itself, or path
            # corresponding to a single neurite
            if len(target_src_path) < 3:
                assert target_src_path[-1] == self.source_id
                continue
            for i in range(len(target_src_path) - 2):
                neurite0 = TopologicalGraph.undirected_hash(target_src_path[i], target_src_path[i + 1])
                neurite1 = TopologicalGraph.undirected_hash(target_src_path[i + 1], target_src_path[i + 2])
                if neurite_subset is None:
                    neurite_pair_orderings.add((neurite0, neurite1))
                else:
                    in_set = int(neurite0 in neurite_subset) + int(neurite1 in neurite_subset)
                    in_ramp = int(neurite0 in ramps) + int(neurite1 in ramps)
                    assert in_ramp == 0 or in_ramp == 1
                    assert in_set == 0 or in_set == 2 or (in_set == 1 and in_ramp == 1)
                    # only add ordering information if both neurites are in
                    # decision set
                    if in_set == 2:
                        neurite_pair_orderings.add((neurite0, neurite1))
        return neurite_pair_orderings

    # cost of neurite (child, parent) is edge cost from parent to child
    def neurite_costs(self):
        edge_costs = {}
        for target_src_path in self.shortest_paths().values():
            for i in range(len(target_src_path) - 1):
                child_id, parent_id = target_src_path[i], target_src_path[i + 1]
                edge_hash = TopologicalGraph.undirected_hash(child_id, parent_id)
                edge_costs[edge_hash] = self.nodes[child_id].u
        return edge_costs

    # cost of freeway and ramp neurites. ramp neurite costs are summed across
    # all off freeway neurites rooted at it
    def decision_neurite_costs(self, ramps):
        decision_edge_costs = self.neurite_costs()
        for ramp_neurite, off_freeway_neurites in ramps.items():
            ramp_neurite_cost = sum([decision_edge_costs[off_freeway_neurite]
                                     for off_freeway_neurite in off_freeway_neurites])
            decision_edge_costs[ramp_neurite] = ramp_neurite_cost
            for off_freeway_neurite in off_freeway_neurites:
                if off_freeway_neurite != ramp_neurite:
                    decision_edge_costs.pop(off_freeway_neurite)
        return decision_edge_costs

