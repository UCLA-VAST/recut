import numpy as np
from .neuron_tree import NeuronTree
from .distribution import Distribution


# the start and end node of a neurite is one of the 3 types of topological nodes
class DirectedNeurite:
    def __init__(self, neurite_id):
        self.neurite_id = neurite_id
        self.node_ids = []
        # {soma_id: [0, pi]}
        self._gof = {}
        self._gof_probability = {}
        self._path_length, self._mean_radius = None, None

    def empty(self):
        return len(self.node_ids) == 0

    def start_node_id(self):
        if self.empty():
            return None
        assert len(self.node_ids) >= 2
        return self.node_ids[0]

    def end_node_id(self):
        if self.empty():
            return None
        assert len(self.node_ids) >= 2
        return self.node_ids[-1]

    def hash(self, directed=False):
        if not directed:
            return (min(self.start_node_id(), self.end_node_id()),
                    max(self.start_node_id(), self.end_node_id()))
        else:
            return self.start_node_id(), self.end_node_id()

    def n_nodes(self):
        return len(self.node_ids)

    # compute gof(self, soma_id) for soma_id in soma_ids and its probability
    # also compute neurite length and mean radius in an intermediate step
    def compute_gof(self, soma_ids, neuron_tree):
        if self.n_nodes() < 2:
            return
        assert isinstance(neuron_tree, NeuronTree)
        epsilon = 1e-10
        # compute unit tangent vectors along the neurite
        neurite_xyzs = np.zeros((self.n_nodes(), 3))
        neurite_radius = np.zeros((self.n_nodes(),))
        for i, node_id in enumerate(self.node_ids):
            neurite_xyzs[i, 0] = neuron_tree.tree[node_id].x
            neurite_xyzs[i, 1] = neuron_tree.tree[node_id].y
            neurite_xyzs[i, 2] = neuron_tree.tree[node_id].z
            neurite_radius[i] = neuron_tree.tree[node_id].radius
        self._mean_radius = np.mean(neurite_radius)
        # matrix with -1 1 repeated in columns, at row postions
        m = -np.eye(self.n_nodes(), M=self.n_nodes() - 1) + \
            np.eye(self.n_nodes(), M=self.n_nodes() - 1, k=-1)
        # neurite tangents
        # row vector = [x[j + 1] - x[j], y[j + 1] - y[j], z[j + 1] - z[j]]
        tangent = np.zeros((self.n_nodes() - 1, 3))
        tangent[:, 0] = np.matmul(np.expand_dims(neurite_xyzs[:, 0], axis=0), m)
        tangent[:, 1] = np.matmul(np.expand_dims(neurite_xyzs[:, 1], axis=0), m)
        tangent[:, 2] = np.matmul(np.expand_dims(neurite_xyzs[:, 2], axis=0), m)
        # keepdims argument leaves shape as (n, 1),
        # which broadcast correctly with original array
        tangent_norm = np.linalg.norm(tangent, ord=2, axis=1, keepdims=True)
        self._path_length = np.sum(tangent_norm)
        tangent /= (tangent_norm + epsilon)
        # gof value for each soma
        for soma_id in soma_ids:
            if soma_id in self._gof:
                continue
            assert soma_id in neuron_tree.tree

            soma_xyz = np.array([neuron_tree.tree[soma_id].x,
                                 neuron_tree.tree[soma_id].y,
                                 neuron_tree.tree[soma_id].z])
            # neurite - soma
            ps = neurite_xyzs[0: self.n_nodes() - 1, :] - soma_xyz
            ps_norm = np.linalg.norm(ps, ord=2, axis=1, keepdims=True)
            ps /= (ps_norm + epsilon)
            # vector projection
            projection = np.clip(np.sum(tangent * ps, axis=1), -1, 1)
            # average of theta at each node in radians (exclude end node)
            self._gof[soma_id] = np.mean(np.arccos(projection))

    def compute_gof_probability(self, distribution):
        assert isinstance(distribution, Distribution)
        for soma_id, gof in self._gof.items():
            self._gof_probability[soma_id] = distribution.probability(gof)

    def gof(self, soma_id, return_angle=False):
        if self.n_nodes() < 2:
            raise ValueError('empty or invalid neurite: has less than 2 nodes')
        if soma_id not in self._gof:
            raise ValueError('no gof computed for soma_id = {}'.format(soma_id))
        convert = 1 if not return_angle else 180 / np.pi
        return self._gof[soma_id] * convert

    def _weight(self, branch_order=1):
        if self._path_length is None or self._mean_radius is None:
            return None
        return np.log(1 + self._path_length)

    def gof_fitness(self, soma_id, branch_order=1):
        assert soma_id in self._gof_probability
        modifier = np.log(1 + 1 / branch_order)
        return self._gof_probability[soma_id] * modifier

    def gof_cost(self, soma_id, branch_order=1):
        cost = 1 - self.gof_fitness(soma_id)
        return cost * self._weight(branch_order=branch_order)
