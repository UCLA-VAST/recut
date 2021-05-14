from collections import defaultdict
import numpy as np
import scipy.ndimage as ndi
# writing import mcp3d_clib does not work. must import all elements
from mcp3d_clib import *


class Point3D:
    def __init__(self, tup, zyx_order=False):
        if zyx_order:
            self.z = tup[0]
            self.y = tup[1]
            self.x = tup[2]
        else:
            self.x = tup[0]
            self.y = tup[1]
            self.z = tup[2]

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __mul__(self, n):
        new_x = self.x * n
        new_y = self.y * n
        new_z = self.z * n
        return Point3D((new_x, new_y, new_z))

    def __rmul__(self, n):
        new_x = self.x * n
        new_y = self.y * n
        new_z = self.z * n
        return Point3D((new_x, new_y, new_z))

    def __add__(self, other):
        new_x = self.x + other.x
        new_y = self.y + other.y
        new_z = self.z + other.z
        return Point3D((new_x, new_y, new_z))

    def __sub__(self, other):
        new_x = self.x - other.x
        new_y = self.y - other.y
        new_z = self.z - other.z
        return Point3D((new_x, new_y, new_z))

    def __truediv__(self, other):
        if isinstance(other, np.int64):
            other += 1e-9
            new_x = self.x / other
            new_y = self.y / other
            new_z = self.z / other
            return Point3D((new_x, new_y, new_z))
        elif isinstance(other, Point3D):
            new_x = self.x / (other.x + 1e-9)
            new_y = self.y / (other.y + 1e-9)
            new_z = self.z / (other.z + 1e-9)
            return Point3D((new_x, new_y, new_z))

    def __str__(self):
        return "{0},{1},{2}".format(self.x, self.y, self.z)


class Objects3D:
    def __init__(self, image_dir, chunk_size=15):
        assert os.path.isdir(image_dir)
        self.image_dir = image_dir
        self.chunk_size = chunk_size
        self.image = pymcp3d.MImage(self.image_dir)
        self.image.ReadImageInfo(0)
        self.image_dims = self.image.xyz_dims()
        print('image dimensions = {}'.format(self.image_dims))
        self.chunk_dims = [chunk_size, self.image_dims[1], self.image_dims[2]]
        # mapping can be one to many, many to one or many to many
        # point from later chunk to earlier chunk
        self.label_map = defaultdict(set)

    def read_chunk(self, z_start):
        # load stack
        block = pymcp3d.MImageBlock([z_start, 0, 0], self.chunk_dims)
        self.image.SelectView(block, 0)
        return self.image.ReadData()

    def _map_chunk_labels(self, current_start_level, previous_end_level):
        # establish mapping between previous chunk end level and
        # labels representing same object and will be merged
        # note that many to one or many to many mapping is possible
        if previous_end_level is not None and np.sum(previous_end_level) > 0:
            assert np.sum(previous_end_level == 0) == np.sum(current_start_level == 0)
            # label the overlap level in 2d
            overlap_label_2d, n = ndi.label(previous_end_level,
                                            np.ones((3, 3), dtype=np.uint8))
            labels_2d, positions = np.unique(overlap_label_2d, return_index=True)
            for label_2d, position in zip(labels_2d, positions):
                if label_2d == 0:
                    continue
                previous_end_label = previous_end_level.ravel()[position]
                current_start_label = current_start_level.ravel()[position]
                self.label_map[current_start_label].add(previous_end_label)

    def _merge_chunk_labels(self):
        # self.label_map: l -> L
        # reverse_label_map:
        # L -> set of l such that set self.label_map[l] contains L
        reverse_label_map = defaultdict(set)
        for label, mapped_labels in self.label_map.items():
            for mapped_label in mapped_labels:
                reverse_label_map[mapped_label].add(label)
        # total set of labels to resolve
        total_labels = set()
        for label, mapped_labels in self.label_map.items():
            total_labels |= mapped_labels
            total_labels.add(label)
        # extract connected labels
        resolved_labels = set()
        # root_label: set {labels connected to root_label}
        roots = defaultdict(set)
        labels = set(self.label_map.keys())
        while len(resolved_labels) < len(total_labels):
            root = Objects3D._get_next_root(labels, resolved_labels)
            assert root is not None
            resolved_labels.add(root)
            roots[root] = Objects3D._find_connected_labels(root, self.label_map, reverse_label_map)
            resolved_labels.update(roots[root])
        # merge connected clusters
        for root, connected_labels in roots.items():
            for connected_label in connected_labels:
                self._merge_component_pair(root, connected_label)
        self._remove_redundant_records(roots)

    def _remove_redundant_records(self, component_roots):
        pass

    def _merge_component_pair(self, component_id0, component_id1):
        pass

    @staticmethod
    def _get_next_root(labels, resolved_labels):
        while len(labels) > 0:
            label = labels.pop()
            if label not in resolved_labels:
                return label
        return None

    @staticmethod
    def _find_connected_labels(root, label_map, reverse_label_map):
        assert root in label_map
        connected = {root}
        frontier = label_map[root] | reverse_label_map[root]
        while len(frontier) > 0:
            label = frontier.pop()
            connected.add(label)
            # if already in connected, do not add to frontier
            frontier.update(
                label_map[label] | reverse_label_map[label] - connected)
        return connected
