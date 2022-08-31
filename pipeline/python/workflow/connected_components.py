import os
import math
from collections import defaultdict
import numpy as np
import scipy.ndimage as ndi
from misc.bounding_box import BoundingBox
from misc.objects_3d import Point3D, Objects3D
from trained_models_catalogue import model_classes_intensity
from pipeline_output_layout import PipelineOutputLayout
from pipeline_arguments import PipelineArguments


class Blob:
    def __init__(self, point3d, mass):
        assert isinstance(point3d, Point3D)
        self.point3d = point3d
        assert mass >= 0
        self.mass = mass


class Somas(Objects3D):
    def __init__(self, pipeline_arguments, pipeline_output_layout, chunk_size=15):
        assert isinstance(pipeline_arguments, PipelineArguments)
        assert isinstance(pipeline_output_layout, PipelineOutputLayout)
        super(Somas, self).__init__(pipeline_output_layout.neural_net_segmentation_dir, chunk_size=chunk_size)
        self.pipeline_arguments = pipeline_arguments
        self.pipeline_output_layout = pipeline_output_layout
        self.blobs = {}

    def find_component_centers(self):
        end_overlap_level = None
        N = 0

        soma_intensity = model_classes_intensity[self.pipeline_arguments.model_classes]['soma']
        for i in range(0, self.image_dims[0], self.chunk_dims[0] - 1):
            image = self.read_chunk(i)
            image[image != soma_intensity] = 0
            image_label, n = ndi.measurements.label(image, np.ones((3, 3, 3), dtype=np.uint8))
            # increment by total number of objects counted so far to guarantee
            # unique label index throughout whole stack
            start_overlap_level = image_label[0, :, :].copy()
            start_overlap_level[start_overlap_level > 0] += N
            self._map_chunk_labels(start_overlap_level, end_overlap_level)
            self._find_chunk_component_centers(image_label, N, i)
            end_overlap_level = image_label[-1, :, :].copy()
            end_overlap_level[end_overlap_level > 0] += N
            N += n
        self._merge_chunk_labels()

    def write_components(self, mass_threshold=None):
        os.makedirs(self.pipeline_output_layout.tracing_dir, exist_ok=True)
        if mass_threshold is None:
            mass_threshold = 0
        prefix = self.pipeline_output_layout.image_prefix
        csv_name = '{}_soma_centers_mass_ge_{}.csv'.format(prefix, mass_threshold)
        swc_name = '{}_soma_centers_mass_ge_{}.swc'.format(prefix, mass_threshold)

        with open(os.path.join(self.pipeline_output_layout.tracing_dir, csv_name), 'w') as csv:
            for k in self.blobs:
                if not isinstance(self.blobs[k], Blob):
                    raise ValueError("non Blob instance encountered")
                if self.blobs[k].mass < mass_threshold:
                    continue
                csv.write('{},{}\n'.format(self.blobs[k].point3d, self.blobs[k].mass))
        print("soma centers saved to {}".format(os.path.join(self.pipeline_output_layout.tracing_dir, csv_name)))

        with open(os.path.join(self.pipeline_output_layout.tracing_dir, swc_name), 'w') as swc:
            for i, k in enumerate(self.blobs):
                if self.blobs[k].mass < mass_threshold:
                    continue
                l = '{} {} {} {} {} {} {}\n'.format(i, 274,
                                                    self.blobs[k].point3d.x,
                                                    self.blobs[k].point3d.y,
                                                    self.blobs[k].point3d.z,
                                                    np.power(0.75 * self.blobs[k].mass / np.pi, 1 / 3), -1)
                swc.write(l)
        print("soma centers saved to {}".format(os.path.join(self.pipeline_output_layout.tracing_dir, swc_name)))

    def _find_chunk_component_centers(self, image_label, n, zoffset):
        # find objects
        component_slices = ndi.find_objects(image_label)

        for component_label, component_slice in enumerate(component_slices, start=1):
            # discard portion in first level of the current self.image_data
            if component_slice[0].start == 0 and zoffset > 0:
                component_slice = (slice(component_slice[0].start + 1,
                                         component_slice[0].stop, None),
                                   component_slice[1], component_slice[2])
            # note that local origin z value also need to account
            # for current chunk's position along whole stack z axis
            local_origin = Point3D((component_slice[0].start + zoffset,
                                    component_slice[1].start,
                                    component_slice[2].start), zyx_order=True)
            component_mass = np.sum(image_label[component_slice] == component_label)

            # for components only in first level new added mass is zero
            # entire mass for these components were included in previous chunk
            if component_mass > 0:
                # remove first level from object bounding box
                component_center = ndi.center_of_mass(image_label[component_slice],
                                                      labels=image_label[component_slice],
                                                      index=component_label)
                component_center = Point3D(component_center, zyx_order=True)
            else:
                component_center = Point3D((0, 0, 0), zyx_order=True)
            component_center += local_origin
            # component_label is within the chunk, indexed from 1
            # component_label + n gives global index across all chunks
            self.blobs[component_label + n] = Blob(component_center, component_mass)

    def _merge_component_pair(self, root, label):
        self.blobs[root].point3d = (self.blobs[label].mass * self.blobs[label].point3d +
                                    self.blobs[root].mass * self.blobs[root].point3d) \
                                    / (self.blobs[label].mass + self.blobs[root].mass)
        self.blobs[root].mass += self.blobs[label].mass

    def _remove_redundant_records(self, component_roots):
        # for each component, remove labels connected to component root
        for component_root, connected_labels in component_roots.items():
            for connected_label in connected_labels:
                if connected_label != component_root:
                    self.blobs.pop(connected_label)


class NeuralCluster:
    def __init__(self, bounding_box, somas=None, center_of_mass=None, mass=0):
        assert isinstance(bounding_box, BoundingBox)
        self.bounding_box = bounding_box
        assert somas is None or isinstance(somas, dict)
        # (z, y, x): mass
        self.somas = somas if somas is not None else {}
        # in zyx order
        assert center_of_mass is None or isinstance(center_of_mass, Point3D)
        self.center_of_mass = None
        self.mass = mass


class NeuralClusters(Objects3D):
    """
    this class finds the 3d bounding box for connected components containing somas
    if model_classes = neurite+soma, foreground_percent is ignored. all positive voxels are foreground (determined by neural network)
    if model_classes = soma, foreground_percent is used
    """
    def __init__(self, pipeline_arguments, pipeline_output_layout, chunk_size=15, neighbors=26):
        assert isinstance(pipeline_arguments, PipelineArguments)
        assert isinstance(pipeline_output_layout, PipelineOutputLayout)
        super(NeuralClusters, self).__init__(pipeline_output_layout.tracing_dir, chunk_size=chunk_size)
        self.pipeline_output_layout = pipeline_output_layout
        self.pipeline_arguments = pipeline_arguments
        if not os.path.isfile(self.pipeline_output_layout.soma_csv_path):
            raise ValueError('required file not found for neural cluster analysis: {}'.format(self.pipeline_output_layout.soma_csv_path))
        self.somas = {}
        self.parse_soma_file()
        self.chunk_dims = [chunk_size, self.image_dims[1], self.image_dims[2]]
        self.neighbors = neighbors
        if self.neighbors not in [8, 18, 26]:
            self.neighbors = 26
        # label: NeuralCluster
        self.neural_clusters = {}

    # x,y,z,mass, as output csv from FindObject3D
    def parse_soma_file(self):
        with open(self.pipeline_output_layout.soma_csv_path, 'r') as f:
            for l in f:
                x, y, z, mass = [int(math.floor(float(token))) for token in l.strip().split(',')]
                self.somas[(z, y, x)] = mass

    def find_component_bounds(self):
        end_overlap_level = None
        N = 0
        # ensure all levels are processed. end of stack pad images are all black
        # and doesn't affect result
        for i in range(0, self.image_dims[0], self.chunk_dims[0] - 1):
            N, end_overlap_level = self.find_chunk_component_bounds(i, N, end_overlap_level)
        self._merge_chunk_labels()

    def write_components(self, write_orphans=False):
        os.makedirs(self.pipeline_output_layout.tracing_dir, exist_ok=True)
        with open(self.pipeline_output_layout.cluster_csv_path, 'w') as f:
            if self.pipeline_arguments.model_classes == 'soma':
                f.write('# foreground percent = {}\n'.format(self.pipeline_arguments.fg_percent))
            f.write('# neighbors = {}\n'.format(self.neighbors))
            f.write('# chunk size = {}\n'.format(self.chunk_size))
            f.write('# zmin, zmax, ymin, ymax, xmin, xmax, soma_number, [soma_z, soma_y, soma_x, soma_mass]...\n')
            for neural_cluster in self.neural_clusters.values():
                line = '{},{},{},{},{},{},{}'.format(neural_cluster.bounding_box.zmin,
                                                     neural_cluster.bounding_box.zmax,
                                                     neural_cluster.bounding_box.ymin,
                                                     neural_cluster.bounding_box.ymax,
                                                     neural_cluster.bounding_box.xmin,
                                                     neural_cluster.bounding_box.xmax,
                                                     len(neural_cluster.somas))
                if len(neural_cluster.somas) == 0:
                    if write_orphans:
                        f.write('{}\n'.format(line))
                    continue
                print(neural_cluster.somas)
                for zyx, mass in neural_cluster.somas.items():
                    line += ',{},{},{},{}'.format(zyx[0], zyx[1], zyx[2], mass)
                f.write('{}\n'.format(line))

    def find_chunk_component_bounds(self, i, N, end_overlap_level):
        image_label, n = self.label_chunk(i)
        print('n = {}'.format(n))
        if n == 0:
            end_overlap_level = image_label[-1, :, :].copy()
        else:
            # increment by total number of objects counted so far to guarantee
            # unique label index throughout whole stack
            start_overlap_level = image_label[0, :, :].copy()
            start_overlap_level[start_overlap_level > 0] += N
            self._map_chunk_labels(start_overlap_level, end_overlap_level)
            self._find_chunk_component_bounds(image_label, N, i)
            end_overlap_level = image_label[-1, :, :].copy()
            end_overlap_level[end_overlap_level > 0] += N
            N += n
        return N, end_overlap_level

    def label_chunk(self, i):
        image = self._threshold_chunk(self.read_chunk(i))
        if self.neighbors == 26:
            struct = np.ones((3, 3, 3), dtype=np.uint8)
        elif self.neighbors == 18:
            struct = np.ones((3, 3, 3), dtype=np.uint8)
            struct[0, 0, 0] = 0
            struct[0, 0, 2] = 0
            struct[0, 2, 0] = 0
            struct[0, 2, 2] = 0
            struct[2, 0, 0] = 0
            struct[2, 0, 2] = 0
            struct[2, 2, 0] = 0
            struct[2, 2, 2] = 0
        else:  # neighbors = 6
            struct = np.zeros((3, 3, 3), dtype=np.uint8)
            struct[0, 1, 1] = 1
            struct[2, 1, 1] = 1
            struct[1, 1, 0] = 1
            struct[1, 1, 2] = 1
            struct[1, 0, 1] = 1
            struct[1, 2, 1] = 1
        return ndi.measurements.label(image, struct)

    def _threshold_chunk(self, image):
        assert self.chunk_dims[0] == image.shape[0]
        if self.pipeline_arguments.model_classes == 'soma':
            for i in range(self.chunk_dims[0]):
                threshold = np.percentile(image[i, :, :], (1 - self.pipeline_arguments.fg_percent) * 100)
                image[i, :, :][image[i, :, :] < threshold] = 0
                image[i, :, :][image[i, :, :] > 0] = 1
        else:
            image[image > 0] = 1
        return image

    def _find_chunk_component_bounds(self, image_label, N, zoffset):
        # obtain labels that needs to be looked at: has soma, or cross bounds
        # label: [soma zyx]
        chunk_components = defaultdict(list)
        # first soma labels
        for soma_zyx in self.somas:
            z, y, x = soma_zyx
            if zoffset <= z < zoffset + self.chunk_dims[0]:
                chunk_component_label = image_label[z - zoffset, y, x]
                if chunk_component_label > 0:
                    chunk_components[chunk_component_label].append(soma_zyx)
        # then labels present in first and last level
        # (to be joined with neighbor chunks)
        crossing_labels = set(np.unique(image_label[0, :, :]))
        crossing_labels.update(np.unique(image_label[-1, :, :]))
        # remove zero value
        crossing_labels.discard(0)
        for crossing_label in crossing_labels:
            if crossing_label not in chunk_components:
                # currently have no soma association
                chunk_components[crossing_label] = []

        # find object slices
        component_slices = ndi.find_objects(image_label)
        for chunk_component_label in chunk_components:
            # index cluster slice with chunk_component_label - 1
            component_slice = component_slices[chunk_component_label - 1]
            if component_slice is None:
                continue
            component_box = BoundingBox(component_slice[0].start + zoffset, component_slice[0].stop + zoffset,
                                        component_slice[1].start, component_slice[1].stop,
                                        component_slice[2].start, component_slice[2].stop)
            component_somas = {}
            for soma_zyx in chunk_components[chunk_component_label]:
                print('somas in component {} chunk zoffset {}: '.format(chunk_component_label, zoffset), soma_zyx)
                component_somas[soma_zyx] = self.somas[soma_zyx]
            # offset chunk_component_label by N to make it globally unique
            self.neural_clusters[chunk_component_label + N] = NeuralCluster(component_box, somas=component_somas)

    def _remove_redundant_records(self, component_roots):
        # remove non root entries
        for label in [key for key in self.neural_clusters.keys()]:
            if label not in component_roots:
                self.neural_clusters.pop(label)

    def _merge_component_pair(self, root, label):
        # update bounding box
        self.neural_clusters[root].bounding_box = self.neural_clusters[root].bounding_box.enclose(self.neural_clusters[label].bounding_box)
        # update somas
        self.neural_clusters[root].somas.update(self.neural_clusters[label].somas)


def make_marker_files(pipeline_output_layout):
    assert isinstance(pipeline_output_layout, PipelineOutputLayout)
    os.makedirs(pipeline_output_layout.marker_file_dir, exist_ok=True)
    with open(pipeline_output_layout.soma_csv_path, 'r') as f:
        for l in f:
            tokens = l.strip().split(',')
            x, y, z, mass = [int(float(token)) for token in tokens]
            marker_name = 'marker_{}_{}_{}_{}'.format(x, y, z, mass)
            with open(os.path.join(pipeline_output_layout.marker_file_dir, marker_name), 'w') as g:
                g.write('# x,y,z\n')
                g.write('{},{},{}\n'.format(x, y, z))


def find_somas(pipeline_arguments, pipeline_output_layout):
    if not os.path.isdir(pipeline_output_layout.neural_net_segmentation_dir):
        raise ValueError('required input directory not found for soma connected component analysis: {}'
                         .format(pipeline_output_layout.neural_net_segmentation_dir))
    somas = Somas(pipeline_arguments, pipeline_output_layout)
    somas.find_component_centers()
    for mass in [1, 314]:
        somas.write_components(mass_threshold=mass)


def find_clusters(pipeline_arguments, pipeline_output_layout):
    finder = NeuralClusters(pipeline_arguments, pipeline_output_layout)
    finder.find_component_bounds()
    finder.write_components(write_orphans=False)

