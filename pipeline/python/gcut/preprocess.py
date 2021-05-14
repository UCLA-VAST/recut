import os
from copy import deepcopy
import numpy as np
from scipy.spatial.distance import cdist
from .neuron_tree import NeuronTree


class Preprocess:
    def __init__(self, swc_path, soma_file_path, scale_z=1):
        assert os.path.isfile(swc_path)
        assert os.path.isfile(soma_file_path)
        self.swc_path = swc_path
        self.scale_z = scale_z
        self.tree_nodes = NeuronTree()
        self.tree_nodes.build_tree_from_swc(self.swc_path)
        self.soma_file_path = soma_file_path
        self.soma_ids = set()
        self.soma_xyzs = {}

    def process(self, standardize_tree=False):
        if standardize_tree:
            id_mapping = self.tree_nodes.standardize(return_id_mapping=True)
        else:
            id_mapping = {}
        if not self.scale_z == 1:
            self.tree_nodes.scale_coordinates(scale_z=self.scale_z)
        # if input swc is not standard, and soma ids are provided directly
        # instead of soma xyz coordinates, the input soma id may no longer
        # exist in the standardized swc and needs to be remapped
        self.extract_soma_ids(id_mapping, size_threshold=None)
        self.soma_sinks()
        self.tree_nodes.prune_short_leaves(min_leaf_branch_length=2,
                                           stable_nodes=self.soma_ids)

    def results(self):
        return self.tree_nodes, self.soma_ids

    def write_somas(self, out_path):
        with open(out_path, 'w') as f:
            for soma_id in self.soma_ids:
                soma_node = self.tree_nodes.tree[soma_id]
                f.write('{} {} {} {} {} {} {}\n'.format(soma_node.node_id,
                                                        soma_node.node_type,
                                                        soma_node.x, soma_node.y,
                                                        soma_node.z,
                                                        30, -1))
        with open(os.path.join(os.path.dirname(out_path), 'xyz_to_node'), 'w') as f:
            for (x, y, z), (cid, cx, cy, cz, cr) in self.soma_xyzs.items():
                f.write('{},{},{},{},{},{},{},{}\n'.format(x, y, z, cid, cx, cy, cz, cr))

    def write_swc_file(self, out_path):
        self.tree_nodes.write_swc_file(out_path)

    def extract_soma_ids(self, id_mapping, size_threshold=None):
        soma_xyzs = []
        with open(self.soma_file_path, 'r') as f:
            for l in f:
                if len(l) == 0:
                    continue
                tokens = l.strip().split()
                if len(tokens) == 1:
                    soma_id = int(tokens[0])
                    if soma_id in id_mapping:
                        print('standardized soma id {} to {}'.format(soma_id, id_mapping[soma_id]))
                        soma_id = id_mapping[soma_id]
                    self.soma_ids.add(soma_id)
                else:
                    assert len(tokens) == 3
                    soma_xyzs.append(np.array([float(tokens[0]), float(tokens[1]), float(tokens[2])]).reshape(1, 3))
        if len(soma_xyzs) > 0:
            soma_xyzs = np.concatenate(soma_xyzs, axis=0)
            # the soma z coordinate is calculated without scaling along z
            soma_xyzs[:, 2] *= self.scale_z
            self.extract_soma_ids_from_xyz(soma_xyzs)

        # size filer
        if size_threshold is not None:
            filtered = set()
            for soma_id in self.soma_ids:
                if self.tree_nodes.tree[soma_id].radius < size_threshold:
                    filtered.add(soma_id)
            print('filtered ', filtered)
            self.soma_ids -= filtered

    def extract_soma_ids_from_xyz(self, soma_xyzs):
        tree_matrix = self.tree_nodes.export_matrix()
        for i in range(soma_xyzs.shape[0]):
            d = cdist(soma_xyzs[i, :].reshape(1, 3), tree_matrix[:, 2:5])
            candidate_index = np.argsort(d.ravel())[0]
            candidate_id = tree_matrix[candidate_index, 0].astype(np.int)
            # app2 reconstruction step will only pass somas present in the cluster
            self.soma_ids.add(candidate_id)
            cx, cy, cz, cr = self.tree_nodes.tree[int(candidate_id)].x, \
                             self.tree_nodes.tree[int(candidate_id)].y, \
                             self.tree_nodes.tree[int(candidate_id)].z, \
                             self.tree_nodes.tree[int(candidate_id)].radius
            self.soma_xyzs[(soma_xyzs[i][0], soma_xyzs[i][1], soma_xyzs[i][2])] = (candidate_id, cx, cy, cz, cr)

    def all_sinks(self):
        self.tree_nodes.find_children(rescan=False)
        node_ids = deepcopy(set([node_id for node_id in self.tree_nodes.tree.keys()]))
        while len(node_ids) > 0:
            node_id = node_ids.pop()
            if node_id not in self.tree_nodes.tree:
                continue
            self.sink_neighbor(node_id)

    def soma_sinks(self):
        self.tree_nodes.find_children(rescan=False)
        soma_ids = deepcopy(self.soma_ids)
        while len(soma_ids) > 0:
            soma_id = soma_ids.pop()
            self.sink_neighbor(soma_id)

    def sink_neighbor(self, node_id):
        if node_id not in self.tree_nodes.tree:
            print('node {} not in neuron tree. possibly has been '
                  'swallowed earlier'.format(node_id))
            return
        # note that ancestors should be swallowed first, due to the possibility
        # of one of the soma ids being swapped with swc root node id
        node_id = self.sink_ancestors(node_id)
        self.sink_descendants(node_id)
        self.tree_nodes.log_operation()

    def _combine_radius(self, node_id0, node_id1):
        v = self.tree_nodes.pair_volume(node_id0, node_id1)
        return np.power(3 * v / (4 * np.pi), 1 / 3)

    # does not merge soma id
    def sink_descendants(self, node_id):
        descendant_ids = list(self.tree_nodes.children[node_id])
        while len(descendant_ids) > 0:
            descendant_id = descendant_ids.pop(0)
            if descendant_id in self.soma_ids:
                continue
            d = self.tree_nodes.pair_distance(node_id, descendant_id)
            if d > self.tree_nodes.tree[node_id].radius + self.tree_nodes.tree[descendant_id].radius:
                continue
            # update soma node radius
            self.tree_nodes.tree[node_id].radius = self._combine_radius(node_id, descendant_id)
            # add children of descendant to queue
            descendant_ids.extend(self.tree_nodes.children[descendant_id])
            # remove descendant
            self.tree_nodes.remove_node(descendant_id)

    # when swallowing ancestors, it's possible to encounter root node of the
    # swc file, which may not be returned by euclidean distance based soma ids
    # if this occurs, replace the soma_id with swc root node and return
    # does not merge two soma nodes
    def sink_ancestors(self, node_id):
        if node_id == self.tree_nodes.root_id():
            return
        parent_id = self.tree_nodes.tree[node_id].parent_id
        while True:
            if parent_id == self.tree_nodes.root_id():
                if node_id in self.soma_ids:
                    self.soma_ids.add(parent_id)
                    self.soma_ids.remove(node_id)
                node_id = self.tree_nodes.root_id()
                break
            if parent_id in self.soma_ids:
                break
            d = self.tree_nodes.pair_distance(node_id, parent_id)
            if d > self.tree_nodes.tree[node_id].radius + self.tree_nodes.tree[parent_id].radius:
                break
            # update soma node radius
            self.tree_nodes.tree[node_id].radius = self._combine_radius(node_id, parent_id)
            # record grand parent id. remove parent node from tree
            grantparent_id = self.tree_nodes.tree[parent_id].parent_id
            self.tree_nodes.remove_node(parent_id)
            # assign grand parent id as new parent id
            parent_id = grantparent_id
        return node_id

