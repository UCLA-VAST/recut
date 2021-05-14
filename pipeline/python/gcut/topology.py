from copy import deepcopy
from collections import defaultdict
from .neuron_tree import NeuronTree
from .neurite import DirectedNeurite


class TopologicalGraph:
    class TopologicalNode(object):
        def __init__(self, node_id, neighbor_ids):
            # direction of the neurite: node_id[0] -> node_id[-1]
            self.node_id = node_id
            # {topo node id arriving from: via neurite_id}
            self.in_neurites = {}
            # {topo node id going to: via neurite_id}
            self.out_neurites = {}
            # id of unfollowed nodes immediately connected to the
            # topological node, either topological or non topological
            self._unexplored_neighbor_node_ids = neighbor_ids

        def remove_unexplored_neighbor(self, neighbor_id):
            self._unexplored_neighbor_node_ids.remove(neighbor_id)

        def topo_neighbor_ids(self):
            return set([in_topo_id for in_topo_id in self.in_neurites])\
                .union(set([out_topo_id for out_topo_id in self.out_neurites]))

        def not_explored(self, neighbor_id):
            return neighbor_id in self._unexplored_neighbor_node_ids

        def n_unexplored_neighbors(self):
            return len(self._unexplored_neighbor_node_ids)

    class SomaNode(TopologicalNode):
        def __init__(self, node_id, neighbor_ids):
            super(TopologicalGraph.SomaNode, self).__init__(node_id, neighbor_ids)

    class BranchNode(TopologicalNode):
        def __init__(self, node_id, neighbor_ids):
            super(TopologicalGraph.BranchNode, self).__init__(node_id, neighbor_ids)

    class LeafNode(TopologicalNode):
        def __init__(self, node_id, neighbor_ids):
            super(TopologicalGraph.LeafNode, self).__init__(node_id, neighbor_ids)

    def __init__(self):
        # {node_id: TopologicalNode}
        self.graph = {}
        # neurite_id: Neurite
        self.neurites = defaultdict(DirectedNeurite)
        # node id: neurite id. nodes in the dict are non topological nodes
        self.node_to_neurite_id = {}
        # neurite_ids with both ends being soma nodes.
        self.unresolvable = set()
        self._current_neurite_id = 0
        # set of node ids with at least one 1 unexplored neighbor
        self._active = set()
        self._soma_ids = set()
        # set of undirected hash of neurites (soma, soma's topological node id)
        self._stems = set()

    def init_topo_nodes(self, neuron_tree, soma_ids):
        assert isinstance(neuron_tree, NeuronTree)
        for soma_id in soma_ids:
            self.graph[soma_id] = TopologicalGraph.SomaNode(soma_id, neuron_tree.neighbor_ids(soma_id))
            self._soma_ids.add(soma_id)
        for branch_id in neuron_tree.branch_nodes():
            if branch_id not in soma_ids:
                self.graph[branch_id] = TopologicalGraph.BranchNode(branch_id, neuron_tree.neighbor_ids(branch_id))
        for leaf_id in neuron_tree.leaves():
            if leaf_id not in soma_ids:
                self.graph[leaf_id] = TopologicalGraph.LeafNode(leaf_id, neuron_tree.neighbor_ids(leaf_id))
        self._active = set([node_id for node_id in self.graph.keys()])

    def clear(self):
        self.graph.clear()
        self._active.clear()
        self._soma_ids.clear()
        self.clear_neurites()

    def clear_neurites(self):
        self._current_neurite_id = 0
        self.neurites.clear()
        self.node_to_neurite_id.clear()
        self.unresolvable.clear()

    def decrement_unexplored_neighbors(self, node_id, neighbor_id):
        assert node_id in self.graph
        self.graph[node_id].remove_unexplored_neighbor(neighbor_id)
        if self.graph[node_id].n_unexplored_neighbors() == 0:
            # node_id can be removed from self.active in split_neurites
            # (from pop() call)
            self._active.discard(node_id)

    def is_topological_node(self, node_id):
        return node_id in self.graph

    def is_soma_node(self, node_id):
        return self.is_topological_node(node_id) and \
               isinstance(self.graph[node_id], TopologicalGraph.SomaNode)

    def is_branch_node(self, node_id):
        return self.is_topological_node(node_id) and \
               isinstance(self.graph[node_id], TopologicalGraph.BranchNode)

    def is_leaf_node(self, node_id):
        return self.is_topological_node(node_id) and \
               isinstance(self.graph[node_id], TopologicalGraph.LeafNode)

    def has_unexplored_neurites(self):
        return len(self._active) > 0

    # create directed graph, where the nodes are topological nodes, and edges
    # are directed neurites between topological nodes. edge weights equal
    # gof of each neurite
    def construct_directed_graph(self, tree_nodes):
        assert isinstance(tree_nodes, NeuronTree)
        if tree_nodes.empty():
            print('empty neuron tree. do nothing')
            return
        while self.has_unexplored_neurites():
            # get a periphery node
            active_node_id = self._active.pop()
            parent_id = tree_nodes.parent_id(active_node_id)
            # if parent_id = NeuronTree.ROOT_PARENT_ID, skip
            if parent_id != NeuronTree.ROOT_PARENT_ID:
                self._construct_neurite(tree_nodes, active_node_id, parent_id,
                                        is_parent=True)
            children_ids = tree_nodes.children_ids(active_node_id)
            for child_id in children_ids:
                self._construct_neurite(tree_nodes, active_node_id, child_id,
                                        is_parent=False)

    # merge master to slave
    def merge_nodes(self, master_id, slave_id):
        pass

    # set of undirected hash of neurites (soma, soma's topological node id)
    def stems(self):
        if len(self._stems) > 0:
            return self._stems
        assert not self.has_unexplored_neurites()
        for soma_id in self._soma_ids:
            self._stems.update([self.neurites[neurite_id].hash(directed=False)
                                for neurite_id in self.graph[soma_id].out_neurites.values()])
        return self._stems

    # tuple(topo_id0, topo_id1), where topo_id0 < topo_id1
    def undirected_hashes(self):
        return set([neurite.hash(directed=False)
                    for neurite in self.neurites.values()])

    def _process_neurite_end_nodes(self, neurite01):
        assert isinstance(neurite01, DirectedNeurite)
        # if first node in node_ids is a leaf node, reverse node_ids, since
        # neurite direction is node_ids[0] -> node_ids[-1]
        if self.is_leaf_node(neurite01.node_ids[0]):
            neurite01.node_ids.reverse()

        node_id0, node_id1 = neurite01.node_ids[0], neurite01.node_ids[-1]
        neighbor0, neighbor1 = neurite01.node_ids[1], neurite01.node_ids[-2]
        assert self.is_topological_node(node_id0) and \
               self.is_topological_node(node_id1)
        # marker down unexplored neighbor nodes of end nodes of the neurite
        self.decrement_unexplored_neighbors(node_id0, neighbor0)
        self.decrement_unexplored_neighbors(node_id1, neighbor1)
        # add the neurite id to the topological nodes' in and out neurites
        neurite01_id = self._current_neurite_id
        self.graph[node_id0].out_neurites[node_id1] = neurite01_id
        self.graph[node_id1].in_neurites[node_id0] = neurite01_id
        self.neurites[neurite01_id] = neurite01
        # if neurite has no leaf node, a neurite with opposite direction is
        # created
        neurite10_id = None
        if not self.is_leaf_node(node_id1):
            neurite10_id = neurite01_id + 1
            neurite10 = DirectedNeurite(neurite10_id)
            neurite10.node_ids = deepcopy(neurite01.node_ids)
            neurite10.node_ids.reverse()
            self.graph[node_id0].in_neurites[node_id1] = neurite10_id
            self.graph[node_id1].out_neurites[node_id0] = neurite10_id
            self.neurites[neurite10_id] = neurite10
            self._current_neurite_id += 2
        else:
            self._current_neurite_id += 1
        # if the neurite direction is unresolvable, add it to self.unresolvable
        if self.is_soma_node(node_id0) and self.is_soma_node(node_id1):
            self.unresolvable.add(neurite01_id)
            self.unresolvable.add(neurite10_id)

    # construct neurite with first two nodes in Neurite.node_ids sequence being
    # [topo_node_id, neighbor_node_id], until another topological node is
    # encountered in the sequence
    # call to process_neurite_end_nodes marks explored edges of self.topo_graph,
    # and update explored edges of self.topo_graph
    def _construct_neurite(self, tree_nodes, topo_node_id, neighbor_node_id,
                           is_parent=True):
        assert isinstance(tree_nodes, NeuronTree)
        assert self.is_topological_node(topo_node_id)
        assert neighbor_node_id in tree_nodes.tree
        assert neighbor_node_id == tree_nodes.parent_id(topo_node_id) or \
               neighbor_node_id in tree_nodes.children_ids(topo_node_id)

        # if the neurite has been previously constructed from the other end
        # node, nothing needs to be done
        if not self.graph[topo_node_id].not_explored(neighbor_node_id):
            return
        node_finder = tree_nodes.parent_id \
            if is_parent else tree_nodes.children_ids

        # extend the neurite to the next topological
        assert self._current_neurite_id not in self.neurites
        neurite = DirectedNeurite(self._current_neurite_id)
        neurite.node_ids.extend([topo_node_id, neighbor_node_id])
        # construct neurite until another topological node is encountered
        if self.is_topological_node(neighbor_node_id):
            self._process_neurite_end_nodes(neurite)
        else:
            self.node_to_neurite_id[neighbor_node_id] = neurite.neurite_id
            while True:
                neighbor_node_id = node_finder(neighbor_node_id)
                if isinstance(neighbor_node_id, set):
                    assert len(neighbor_node_id) == 1
                    neighbor_node_id = neighbor_node_id.pop()
                # it is possible for the tree root node (with parent = -1) to be
                # classified as none of soma node, branch node or leaf node
                # (since NeuronTree.leaves() does not return tree root node),
                #  but in reality be a leaf node
                if neighbor_node_id == NeuronTree.ROOT_PARENT_ID:
                    break
                neurite.node_ids.append(neighbor_node_id)
                if self.is_topological_node(neighbor_node_id):
                    self._process_neurite_end_nodes(neurite)
                    break
                self.node_to_neurite_id[neighbor_node_id] = neurite.neurite_id

    @staticmethod
    def undirected_hash(node_id0, node_id1):
        return min(node_id0, node_id1), max(node_id0, node_id1)

    @staticmethod
    def sort_undirected_neurites(undirected_neurites):
        # use two stable sorts by first and then second values in the tuple
        neurite_list = list(undirected_neurites)
        neurite_list.sort(key=lambda neurite: neurite[1])
        neurite_list.sort(key=lambda neurite: neurite[0])
        return neurite_list

    # given the set of undirected neurites, in the form of tuple(node0, node1),
    # retrieve the directed neurites involving the same set of topological nodes
    def directed_neurite_ids(self, undirected_neurites):
        neurite_ids = set()
        for undirected_neurite in undirected_neurites:
            node0, node1 = undirected_neurite
            if node1 in self.graph[node0].out_neurites:
                neurite_ids.add(self.graph[node0].out_neurites[node1])
            if node0 in self.graph[node1].out_neurites:
                neurite_ids.add(self.graph[node1].out_neurites[node0])
        return neurite_ids
