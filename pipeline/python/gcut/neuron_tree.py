import os
from copy import deepcopy
import argparse
import json
import inspect
from collections import defaultdict
import numpy as np


class NeuronNode:
    def __init__(self, node_id, node_type, x, y, z, radius, parent_id):
        self.node_id = node_id
        self.node_type = node_type
        self.x, self.y, self.z, self.radius = x, y, z, radius
        self.parent_id = parent_id

    @staticmethod
    def read_swc_meta_lines(swc_path):
        assert os.path.isfile(swc_path)
        meta_lines = []
        with open(swc_path, 'r') as f:
            for l in f:
                if len(l.strip()) == 0:
                    continue
                elif l.strip()[0] == '#':
                    meta_lines.append(l)
                else:
                    break
        return meta_lines

    @staticmethod
    def read_swc_node_lines(swc_path):
        assert os.path.isfile(swc_path)
        with open(swc_path, 'r') as f:
            for l in f:
                if len(l.strip()) == 0:
                    continue
                if l.strip()[0] == '#':
                    continue
                delim = ' ' if l.find(',') < 0 else ','
                node_id_str, node_type_str, x_str, y_str, z_str, \
                radius_str, parent_id_str = l.strip().split(delim)
                yield [int(node_id_str), int(node_type_str),
                       float(x_str), float(y_str), float(z_str),
                       float(radius_str), int(parent_id_str)]

    @staticmethod
    def write_swc_file_line(f, node):
        assert isinstance(node, NeuronNode)
        f.write('{} {} {} {} {} {} {}\n'.format(node.node_id, node.node_type,
                                                node.x, node.y, node.z,
                                                node.radius, node.parent_id))


class NeuronTree:
    ROOT_PARENT_ID = -1
    MAX_BRANCH_ORDER = 99999

    def __init__(self, meta_lines=''):
        self.meta_lines = meta_lines
        # node_id: NeuronNode
        self.tree = {}
        # node_id: {ids of children of node_id}
        self.children = defaultdict(set)
        # log operations to neuron tree sequentially
        self._operations = []

    def build_tree_from_swc(self, swc_path):
        self.clear()
        # place original metaline in self.meta_lines
        self.write_to_meta_lines('\n'.join(['# original metaline: ' + meta_line.strip()
                                            for meta_line in NeuronNode.read_swc_meta_lines(swc_path)]))
        find_root = False
        for node_id, node_type, x, y, z, radius, parent_id in \
                NeuronNode.read_swc_node_lines(swc_path):
            if parent_id == NeuronTree.ROOT_PARENT_ID:
                if find_root:
                    raise ValueError('multiple root node found in file {}, '
                                     'use NeuronTrees class instead'.format(swc_path))
                else:
                    find_root = True
            node = NeuronNode(node_id, node_type, x, y, z, radius, parent_id)
            self.add_node(node)
        if self.empty():
            return
        self.find_children()
        self.log_operation()

    def write_to_meta_lines(self, contents, append=True):
        assert isinstance(contents, str)
        if append:
            self.meta_lines += '{}\n'.format(contents)
        else:
            self.meta_lines = '{}\n'.format(contents)

    def size(self):
        return len(self.tree)

    def empty(self):
        return self.size() == 0

    def clear(self):
        self.meta_lines = ''
        self.tree.clear()
        self.children.clear()
        self._operations.clear()

    def root_id(self):
        self.find_children(rescan=False)
        assert len(self.children[NeuronTree.ROOT_PARENT_ID]) == 1
        return next(iter(self.children[NeuronTree.ROOT_PARENT_ID]))

    def parent_id(self, node_id):
        if node_id not in self.tree:
            return None
        return self.tree[node_id].parent_id

    def children_ids(self, node_id):
        if node_id not in self.tree:
            return None
        self.find_children(rescan=False)
        return self.children[node_id]

    def neighbor_ids(self, node_id):
        if node_id not in self.tree:
            return None
        neighbors = set()
        neighbors.add(self.parent_id(node_id))
        neighbors |= self.children_ids(node_id)
        neighbors.discard(NeuronTree.ROOT_PARENT_ID)
        return neighbors

    def find_children(self, rescan=True):
        if not rescan and len(self.children) > 0:
            return
        self.children.clear()
        if self.empty():
            return
        for node_id, node in self.tree.items():
            parent_id = node.parent_id
            self.children[parent_id].add(node_id)
            # make sure leaf nodes with no children have entries
            if node_id not in self.children:
                self.children[node_id] = set()

    def leaves(self):
        leaf_nodes = set()
        if self.empty():
            return leaf_nodes
        self.find_children(rescan=False)
        for node_id in self.tree:
            if self.n_children(node_id) == 0:
                # root can not be leaf
                if node_id != self.root_id():
                    leaf_nodes.add(node_id)
        return leaf_nodes

    def branch_orders(self):
        orders = {}
        if self.empty():
            return orders
        self.find_children(rescan=False)
        children = [self.root_id()]
        while len(children) > 0:
            node_id = children.pop(-1)
            children.extend(self.children[node_id])
            parent_id = self.tree[node_id].parent_id
            if parent_id == NeuronTree.ROOT_PARENT_ID:
                orders[node_id] = 0
                continue
            if parent_id == self.root_id():
                orders[node_id] = 0
                continue
            parent_branch_order = self.branch_orders[parent_id]
            orders[node_id] = parent_branch_order + \
                              int(self.n_children(parent_id) > 1)
        return orders

    def branch_nodes(self):
        self.find_children(rescan=False)
        return set([branch_node_id for branch_node_id in self.children
                    if self.n_children(branch_node_id) > 1])

    def node_volume(self, node_id):
        if node_id not in self.tree:
            return -1
        return (4 / 3) * np.pi * np.power(self.tree[node_id].radius, 3)

    def _node_xyz(self, node_id):
        if node_id not in self.tree:
            return None
        return np.array([self.tree[node_id].x,
                         self.tree[node_id].y,
                         self.tree[node_id].z])

    def pair_distance(self, node_id0, node_id1):
        if node_id0 not in self.tree or node_id1 not in self.tree:
            return -1
        if node_id0 == node_id1:
            return 0
        return np.linalg.norm(self._node_xyz(node_id0) -
                              self._node_xyz(node_id1), 2)

    # if node0 contains node1, return node_id0. otherwise return -1
    def pair_containment(self, node_id0, node_id1):
        if node_id0 not in self.tree or node_id1 not in self.tree:
            return -1
        d = self.pair_distance(node_id0, node_id1)
        r0 = self.tree[node_id0].radius
        r1 = self.tree[node_id1].radius
        if d + r0 < r1:
            return node_id1
        elif d + r1 < r0:
            return node_id0
        else:
            return -1

    def intersect(self, node_id0, node_id1):
        if node_id0 not in self.tree or node_id1 not in self.tree:
            return False
        d = self.pair_distance(node_id0, node_id1)
        r0 = self.tree[node_id0].radius
        r1 = self.tree[node_id1].radius
        return r0 + r1 > d

    # return true if node with index node_id contains (x, y, z)
    def node_contains(self, node_id, x, y, z, tol=0):
        if node_id not in self.tree:
            return False
        d = np.linalg.norm(self._node_xyz(node_id) - np.array([x, y, z]), 2)
        return d < self.tree[node_id].radius + tol

    # volume of the intersection region between two nodes
    def intersection_volume(self, node_id0, node_id1):
        if node_id0 not in self.tree or node_id1 not in self.tree:
            return -1
        if node_id0 == node_id1:
            return self.node_volume(node_id0)
        d = self.pair_distance(node_id0, node_id1)
        r0 = self.tree[node_id0].radius
        r1 = self.tree[node_id1].radius
        if r0 + r1 <= d:
            return 0
        # one node entirely contained by the other
        containing_node_id = self.pair_containment(node_id0, node_id1)
        if containing_node_id >= 0:
            contained_node_id = node_id0 \
                if containing_node_id == node_id1 else node_id1
            return self.node_volume(contained_node_id)

        # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        v = np.pi * np.power(r0 + r1 - d, 2) * \
            (np.power(d, 2) + 2 * d *(r0 + r1) -
             3 * (np.power(r0, 2) + np.power(r1, 2)) + 6 * r0 * r1) / (12 * d)
        return v

    # total volume of two nodes. intersection is accounted
    def pair_volume(self, node_id0, node_id1):
        if node_id0 not in self.tree or node_id1 not in self.tree:
            return -1
        r0 = self.tree[node_id0].radius
        r1 = self.tree[node_id1].radius
        return (4 / 3) * np.pi * (np.power(r0, 3) + np.power(r1, 3)) - \
               self.intersection_volume(node_id0, node_id1)

    # logs calling function's name. description if present is expeted as
    # the function's parameter dictionary string decodable by json.loads
    # example '{{\"name\":value={}}}'.format(v)
    # this is meant to reproduce the results from original input
    def log_operation(self):
        caller_frame_info = inspect.stack()[1]
        # retrive calling function's name
        caller_name = caller_frame_info.function
        caller_frame = caller_frame_info.frame
        arg_names, _, __, locals_dict = inspect.getargvalues(caller_frame)
        arg_names = set(arg_names)
        if 'self' in arg_names:
            caller_prefix = locals_dict['self'].__class__
        else:
            caller_prefix = caller_frame_info.filename
        arg_dict = {k: str(v) for k, v in locals_dict.items() if k in arg_names}
        self._operations.append('# operation log: {}.{}: {}'.format(caller_prefix, caller_name, json.dumps(arg_dict)))

    def add_node(self, node):
        assert isinstance(node, NeuronNode)
        self.tree[node.node_id] = node

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    # remove the node with node_id, connect its parent and children
    # modify self.tree and self.children accordingly
    def remove_node(self, node_id):
        if node_id not in self.tree:
            print('node_id = {} not in NeuronTree'.format(node_id))
            return
        parent_id = self.tree[node_id].parent_id
        self.find_children(rescan=False)
        if node_id == self.root_id() and self.n_children(node_id) > 1:
            print('can not remove root node with more than one children')
            return
        # connect node's parent and children
        children_ids = self.children[node_id]
        self.children[parent_id].remove(node_id)
        self.children[parent_id].update(children_ids)
        for child_id in children_ids:
            self.tree[child_id].parent_id = parent_id
        # remove node
        self.tree.pop(node_id)
        self.children.pop(node_id)
        self.log_operation()

    def n_children(self, node_id):
        if node_id not in self.children:
            print('node_id = {} not in NeuronTree.children'.format(node_id))
            return -1
        self.find_children(rescan=False)
        return len(self.children[node_id])

    def node_degree(self, node_id):
        if node_id not in self.tree:
            print('node_id = {} not in NeuronTree'.format(node_id))
            return -1
        return self.n_children(node_id) + int(node_id != self.root_id())

    # export the swc data matrix
    def export_matrix(self):
        if self.empty():
            return None
        matrix = np.zeros((len(self.tree), 7))
        for i, node_id in enumerate(self.tree):
            matrix[i, 0] = node_id
            matrix[i, 1] = self.tree[node_id].node_type
            matrix[i, 2] = self.tree[node_id].x
            matrix[i, 3] = self.tree[node_id].y
            matrix[i, 4] = self.tree[node_id].z
            matrix[i, 5] = self.tree[node_id].radius
            matrix[i, 6] = self.tree[node_id].parent_id
        return matrix

    def offset_node_ids(self, offset_value):
        assert offset_value >= 0
        if offset_value == 0:
            return
        new_tree = {}
        for node_id in self.tree:
            new_tree[node_id + offset_value] = self.tree[node_id]
            new_tree[node_id + offset_value].node_id += offset_value
            if new_tree[node_id + offset_value].parent_id != NeuronTree.ROOT_PARENT_ID:
                new_tree[node_id + offset_value].parent_id += offset_value
        self.tree = new_tree
        self.find_children(rescan=True)
        self.log_operation()

    def prune_branch_order(self, max_branch_order=MAX_BRANCH_ORDER):
        """
        the bifurcation node is grouped with the branch of its parent
        instead of children
        :param swc_tree: input swc tree to be pruned
        :param max_branch_order: branches with branch order greater than
        max_branch_order are discarded
        :return:
        """
        assert max_branch_order >= 0
        branch_orders = self.branch_orders()
        for node_id, node_branch_order in branch_orders.items():
            if node_branch_order > max_branch_order:
                self.tree.pop(node_id)
        self.find_children(rescan=True)
        self.log_operation()

    # if a leaf branch has less than min_leaf_branch_length nodes
    # (excluding the branch point it originates from), the branch is removed
    # recursively remove all such branches
    # stable_nodes is a set of node ids that can not be removed
    def prune_short_leaves(self, min_leaf_branch_length=2, stable_nodes=None):
        if stable_nodes is None:
            stable_nodes = {self.root_id()}
        info = 'iteratively prune leaf ' \
               'branches with <= {} nodes'.format(min_leaf_branch_length)
        while self._prune_short_leaves(min_leaf_branch_length=min_leaf_branch_length, stable_nodes=stable_nodes):
            print(info)
        self.log_operation()

    def _prune_short_leaves(self, min_leaf_branch_length=2, stable_nodes=None):
        if stable_nodes is None:
            stable_nodes = set()
        has_short_leaves = False
        self.find_children(rescan=True)
        leaf_ids = self.leaves()
        for leaf_id in leaf_ids:
            # stable node enountered, go to next leaf
            if leaf_id in stable_nodes:
                continue
            neurite_ids = [leaf_id]
            neurite_id = leaf_id
            while True:
                parent_id = self.tree[neurite_id].parent_id
                # stable node enountered, neurite ends
                if parent_id in stable_nodes:
                    break
                # if branch node enounterd, neurite ends
                if self.n_children(parent_id) > 1:
                    break
                neurite_id = parent_id
                neurite_ids.append(neurite_id)
            if len(neurite_ids) < min_leaf_branch_length:
                has_short_leaves = True
                for neurite_id in neurite_ids:
                    self.children[self.tree[neurite_id].parent_id].remove(neurite_id)
                    self.children.pop(neurite_id)
                    self.tree.pop(neurite_id)
            neurite_ids.clear()
        return has_short_leaves

    def standardize(self, return_id_mapping=False):
        id_mapping = self.remap_to_standard_ids()
        self.find_children(rescan=True)
        self.log_operation()
        if return_id_mapping:
            return id_mapping

    def translate(self, delta_x=0, delta_y=0, delta_z=0):
        for node_id in self.tree:
            self.tree[node_id].x += delta_x
            self.tree[node_id].y += delta_y
            self.tree[node_id].z += delta_z
        self.log_operation()

    """
    x axis pointing right, y axis pointing down, theta denotes angle of rotation
    in xy plane. positive theta is clock wise rotation
    the 3d rotation matrices assumes x axis right, y axis down, z axis away
    """
    def rotate(self, theta=0.0, gamma=0.0, unit_is_angle=True):
        if unit_is_angle:
            theta *= (np.pi / 180)
            gamma *= (np.pi / 180)
        assert 0.0 <= theta < 2 * np.pi
        assert 0.0 <= gamma < 2 * np.pi
        # 2d rotation in xy plane
        cosine_theta = np.cos(theta)
        sine_theta = np.sin(theta)
        if gamma == 0.0:
            rotation_m2d = np.matrix([[cosine_theta, -sine_theta],
                                      [sine_theta, cosine_theta]])
            for node in self.tree.values():
                new_xy = np.matmul(rotation_m2d, np.array([node.x, node.y]))
                new_xy = np.squeeze(np.asarray(new_xy))
                newx, newy = new_xy
                node.x, node.y = newx, newy
        else:
            rotation_m3dx = np.matrix([[1, 0, 0],
                                       [0, cosine_theta, -sine_theta],
                                       [0, sine_theta, cosine_theta]])
            rotation_m3dz = np.matrix([[cosine_theta, -sine_theta, 0],
                                       [sine_theta, cosine_theta, 0],
                                       [0, 0, 1]])
            rotation_m3d = np.matmul(rotation_m3dx, rotation_m3dz)
            rotation_m3d = np.squeeze(np.asarray(rotation_m3d))
            for node in self.tree.values():
                newx, newy, newz = rotation_m3d
                node.x, node.y, node.z = newx, newy, newz
        self.log_operation()

    def scale_coordinates(self, scale_x=1, scale_y=1, scale_z=1):
        assert scale_x > 0 and scale_y > 0 and scale_z > 0
        for node_id in self.tree:
            self.tree[node_id].x *= scale_x
            self.tree[node_id].y *= scale_y
            self.tree[node_id].z *= scale_z
        self.log_operation()

    def scale_radius(self, scale=1):
        assert scale > 0
        for node_id in self.tree:
            self.tree[node_id].radius *= scale
        self.log_operation()

    # node type = 0 for non soma nodes, node type = 1 for soma node
    def aivianize(self):
        for node in self.tree.values():
            node.node_type = 1if node.parent_id == NeuronTree.ROOT_PARENT_ID else 0
        self.log_operation()

    def remap_to_standard_ids(self):
        """
        remap node ids from 1 to n
        :return:
        """
        if self._ids_are_standard():
            return
        # create node id mapping
        node_id_mapping = {}
        start_root_id = self.root_id()
        node_ids = [key for key in self.tree.keys()]
        next_id = 2
        for node_id in node_ids:
            if node_id == start_root_id:
                node_id_mapping[node_id] = 1
            else:
                node_id_mapping[node_id] = next_id
                next_id += 1
        node_id_mapping[NeuronTree.ROOT_PARENT_ID] = NeuronTree.ROOT_PARENT_ID
        # change node ids according to mapping
        new_tree = {}
        for old_id, new_id in node_id_mapping.items():
            if old_id == NeuronTree.ROOT_PARENT_ID:
                continue
            new_tree[new_id] = self.tree[old_id]
            new_tree[new_id].node_id = new_id
            old_parent_id = new_tree[new_id].parent_id
            new_tree[new_id].parent_id = node_id_mapping[old_parent_id]
        self.tree = new_tree
        return node_id_mapping

    def write_swc_file(self, output_path):
        with open(output_path, 'w') as f:
            node_ids = [key for key in self.tree.keys()]
            node_ids.sort()
            self.write_to_meta_lines('\n'.join(self._operations))
            f.write(self.meta_lines)
            for node_id in node_ids:
                NeuronNode.write_swc_file_line(f, self.tree[node_id])

    def _ids_are_standard(self):
        if len(self.tree) == 0:
            return True
        self.find_children(rescan=False)
        root_id = next(iter(self.children[NeuronTree.ROOT_PARENT_ID]))
        if root_id != 1:
            return False
        return self.tree.keys() == range(1, len(self.tree) + 1)


class NeuronTrees:
    def __init__(self, meta_lines=''):
        self.meta_lines = meta_lines
        # root_node_id: NeuronTree
        self.trees = {}

    def clear(self):
        self.meta_lines = ''
        for tree in self.trees:
            tree.clear()

    def n_trees(self):
        return len(self.trees)

    def n_nodes(self):
        return sum([tree.size() for tree in self.trees.values()])

    def max_node_id(self):
        if self.n_nodes() == 0:
            return 0
        return max([max([node_id for node_id in tree.tree.keys()]) for tree in self.trees.values()])

    # for single swc file that contain multiple neuron trees. each neuron_id in
    # the file must be unique
    def build_trees_from_swc(self, swc_path):
        self.clear()
        root_ids = set()
        nodes = {}
        self.meta_lines += NeuronNode.read_swc_meta_lines(swc_path)
        for node_id, node_type, x, y, z, radius, parent_id in NeuronNode.read_swc_node_lines():
            if self._node_id_exists(node_id):
                raise ValueError('node id {} already exists'.format(node_id))
            if parent_id == NeuronTree.ROOT_PARENT_ID:
                root_ids.add(node_id)
            nodes[node_id] = NeuronNode(node_id, node_type, x, y, z, radius, parent_id)
        children = defaultdict(list)
        for node in nodes:
            children[node.parent_id].append(node.node_id)
        for root_id in root_ids:
            self._build_tree_from_root(root_id, nodes, children)

    # for a list of individual swc files. will offset node ids to make them
    # unique
    def build_trees_from_swcs(self, swc_paths, clear_existing=True):
        if clear_existing:
            self.clear()
        self.meta_lines += '# including files \n{}'\
                           .format('\n'.join(['# {}'.format(swc_path)
                                              for swc_path in swc_paths]))
        tree_list = []
        for swc_path in swc_paths:
            tree = NeuronTree()
            tree.build_tree_from_swc(swc_path)
            tree_list.append(tree)
        offset = max(self.n_nodes(), self.max_node_id())
        for tree in tree_list:
            tree.offset_node_ids(offset)
            tree_root_id = tree.root_id()
            self.trees[tree_root_id] = tree
            offset += max(self.trees[tree_root_id].size(),
                          max([node_id for node_id in self.trees[tree_root_id].tree.keys()]))

    def write_swc_file(self, output_path):
        with open(output_path, 'w') as f:
            f.write('{}\n'.format(self.meta_lines))
            root_ids = [root_id for root_id in self.trees.keys()]
            root_ids.sort()
            for root_id in root_ids:
                node_ids = [node_id for node_id in self.trees[root_id].tree.keys()]
                node_ids.sort()
                for node_id in node_ids:
                    NeuronNode.write_swc_file_line(f, self.trees[root_id].tree[node_id])

    # writes individual swc files
    def write_swc_files(self, output_paths, standardize=True, indices=None):
        n_files = len(output_paths) if indices is None else len(indices)
        assert n_files <= len(self.trees)
        root_ids = [root_id for root_id in self.trees.keys()]
        root_ids.sort()
        if indices is None:
            indices = range(len(output_paths))
        for i in indices:
            copy_tree = deepcopy(self.trees[root_ids[i]])
            if standardize:
                copy_tree.standardize()
            copy_tree.write_swc_files(output_paths[i])
            print('save {}'.format(output_paths[i]))

    def _node_id_exists(self, node_id):
        for tree in self.trees:
            if node_id in tree:
                return True
        return False

    def _build_tree_from_root(self, root_id, nodes, children):
        self.trees[root_id] = NeuronTree()
        root_children = children[root_id]
        while len(root_children) > 0:
            child_id = root_children.pop(0)
            self.trees[root_id].add_node(nodes[child_id])
            root_children.extend(children[child_id])


def test():
    swc_path = '/home/muyezhu/mcp3d/python/misc/SW190111-01R_g1_1.swc'
    tree = NeuronTree()
    tree.build_tree_from_swc(swc_path)
    pruned_tree = tree.prune_branch_order(max_branch_order=1)
    pruned_tree.standardize(output_path='/home/muyezhu/mcp3d/python/misc/SW190111-01R_g1_1_branch_order=1.swc')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--standardize', action='store_true',
                        help='standardize swc file')
    parser.add_argument('-t', '--translate', action='store_true',
                        help='translate swc file by dx, dy, dz')
    parser.add_argument('-r', '--rotate', nargs='*',
                        help='rotate swc in xy plane(ccw) and/or around z '
                             'axis(ccw) by given angles')
    parser.add_argument('-dx', type=int, default=0,
                        help='amount of translation on x axis')
    parser.add_argument('-dy', type=int, default=0,
                        help='amount of translation on y axis')
    parser.add_argument('-dz', type=int, default=0,
                        help='amount of translation on z axis')
    parser.add_argument('-prune_order', default=-1,
                        help='prune neuron tree to the given branch order '
                             '(higher order branches discarded)')
    parser.add_argument('-prune_leaves', default=-1,
                        help='prune leaves no greater than a minimum number of '
                             'edges away from its branch point')
    parser.add_argument('-prune_distance', default=1000,
                        help='discard nodes with distance to soma '
                             'greater than given value')
    parser.add_argument('input_path', help='path to input swc file')
    parser.add_argument('output_path', help='path to output swc file')
    args = parser.parse_args()
    neuron_tree = NeuronTree()
    neuron_tree.build_tree_from_swc(args.input_path)
    if args.translate:
        print('translate {}'.format(args.input_path))
        neuron_tree.translate(delta_x=args.dx, delta_y=args.dy, delta_z=args.dz)
    if len(args.rotate) > 0:
        print(args.rotate)
        theta = float(args.rotate[0])
        gamma = float(args.rotate[1]) if len(args.rotate) > 1 else 0.0
        neuron_tree.rotate(theta=theta, gamma=gamma)
    if args.prune_leaves > 0:
        print('prune leavels no greater than {} edges away from '
              'its branch point'.format(args.prune_leaves))
        neuron_tree.prune_short_leaves(min_leaf_branch_length=args.prune_leaves)
    if args.prune_order > 0:
        print('prune neuron tree with max branch order = {}'
              .format(args.prune_order))
        neuron_tree.prune_branch_order(max_branch_order=int(args.prune_order))
    if args.standardize:
        print('standardize {}'.format(args.input_path))
        neuron_tree.standardize()
    neuron_tree.write_swc_file(args.output_path)


if __name__ == '__main__':
    main()
