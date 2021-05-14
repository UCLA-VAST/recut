from shutil import rmtree
from subprocess import run
import json
import math
import numpy as np
from scipy.spatial.distance import cdist
from mcp3d_clib import *
from mcp3d_cbin import app2_bin
from misc.bounding_box import BoundingBox
from gcut.neuron_tree import NeuronTree
from pipeline_arguments import PipelineArguments
from pipeline_output_layout import PipelineOutputLayout


class ClusterCsvParser:
    def __init__(self, cluster_csv_path, roi_bounds):
        if not os.path.isfile(cluster_csv_path):
            raise ValueError('cluster csv file not found: {}'.format(cluster_csv_path))
        self.cluster_csv_path = cluster_csv_path
        self.roi_bounds = roi_bounds

    def next_line(self):
        with open(self.cluster_csv_path, 'r') as f:
            for l in f:
                yield l

    @staticmethod
    def line_tokens(line):
        if line[0] == '#':
            return None
        tokens = [int(math.floor(float(token))) for token in line.strip().split(',')]
        assert len(tokens) >= 7 and (len(tokens) - 7) % 4 == 0
        return tokens

    @staticmethod
    def soma_locations(line):
        locations = []
        tokens = ClusterCsvParser.line_tokens(line)
        if not tokens:
            return locations
        n_somas = tokens[6]
        if n_somas == 0:
            return locations
        for position in range(7, len(tokens), 4):
            locations.append(tokens[position: position + 4])
        return locations

    @staticmethod
    def cluster_bounds(line):
        tokens = ClusterCsvParser.line_tokens(line)
        if not tokens:
            return BoundingBox()
        zmin, zmax, ymin, ymax, xmin, xmax = tokens[0:6]
        return BoundingBox(zmin=zmin, zmax=zmax, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)

    def in_bound_soma_locations(self, line):
        locations = ClusterCsvParser.soma_locations(line)
        if not locations:
            return locations
        cluster_bounds = ClusterCsvParser.cluster_bounds(line)
        bounds = cluster_bounds.intersect(self.roi_bounds)
        if bounds.empty():
            return locations
        in_bound_locations = {}
        for location in locations:
            somaz, somay, somax, mass = location
            if not bounds.zmin <= somaz < bounds.zmax:
                continue
            if not bounds.ymin <= somay < bounds.ymax:
                continue
            if not bounds.xmin <= somax < bounds.xmax:
                continue
            in_bound_locations[(somaz, somay, somax)] = mass
        return in_bound_locations


class App2Reconstruction:
    def __init__(self, pipeline_arguments, pipeline_output_layout):
        assert isinstance(pipeline_arguments, PipelineArguments)
        assert isinstance(pipeline_output_layout, PipelineOutputLayout)
        self.pipeline_arguments = pipeline_arguments
        self.pipeline_output_layout = pipeline_output_layout

        if os.path.isdir(self.pipeline_output_layout.reconstruction_roi_dir):
            print('removing existing reconstruction directory {}'.format(self.pipeline_output_layout.reconstruction_roi_dir))
            rmtree(self.pipeline_output_layout.reconstruction_roi_dir, ignore_errors=True)
        os.makedirs(self.pipeline_output_layout.reconstruction_roi_dir, exist_ok=True)
        self.command_logs = {}
        self.execution_id = 0
        self.log_path = os.path.join(self.pipeline_output_layout.reconstruction_roi_dir, 'app2_log')

        self.cnn_type = 2
        self.length_thresh = 100

        self.roi_bounds = PipelineOutputLayout.get_roi_bounds(self.pipeline_arguments.input_image_dims(),
                                                              self.pipeline_arguments.offset, self.pipeline_arguments.extent)

    def trace(self):
        if self.pipeline_arguments.app2_auto_soma:
            self.trace_app2_auto_soma()
        else:
            cluster_csv_parser = ClusterCsvParser(self.pipeline_output_layout.cluster_csv_path, self.roi_bounds)
            for line in cluster_csv_parser.next_line():
                # if the cluster is outside of roi, skip the cluster
                bounds = ClusterCsvParser.cluster_bounds(line).intersect(self.roi_bounds)
                if bounds.empty():
                    continue
                soma_locations = cluster_csv_parser.in_bound_soma_locations(line)
                if not soma_locations:
                    continue
                self.trace_cluster(bounds, soma_locations)
        self.write_log()

    def trace_app2_auto_soma(self):
        app2_command, swc_path = self._generate_app2_command(self.roi_bounds, marker_path=None)
        run(app2_command, shell=True, check=True)
        swc_resample_path = swc_path.replace('.swc', '_resample.swc')
        if os.path.isfile(swc_resample_path):
            # check if each soma in cluster is reached by the tracing
            swc_translate_path = self.map_swc_to_global_volume(swc_resample_path)
        self.log(app2_command, swc_path)

    def trace_cluster(self, bounds, target_locations):
        cluster_id = np.random.randint(0, high=10000)
        run_id = 0

        while len(target_locations) > 0:
            initial_target_locations = target_locations.copy()
            source_zyx, source_mass = target_locations.popitem()
            marker_path = self._marker_path(source_zyx, source_mass)
            app2_command, swc_path = self._generate_app2_command(bounds, marker_path=marker_path,
                                                                 cluster_id=cluster_id, run_id=run_id)
            run(app2_command, shell=True, check=True)
            # if swc_resample_path does not exist, record command as failed, execute next command
            swc_resample_path = swc_path.replace('.swc', '_resample.swc')
            reached_locations = [source_zyx]
            if os.path.isfile(swc_resample_path):
                # check if each soma in cluster is reached by the tracing
                swc_translate_path = self.map_swc_to_global_volume(swc_resample_path)
                if swc_translate_path:
                    tree = NeuronTree()
                    tree.build_tree_from_swc(swc_translate_path)
                    tree_matrix = tree.export_matrix()
                    check_locations = [key for key in target_locations.keys()]
                    for check_location in check_locations:
                        # remove reached soma from potential source pool
                        mass = target_locations[check_location]
                        r = np.power(3 * mass / (4 * np.pi), 1 / 3)
                        if App2Reconstruction.soma_reached(check_location, tree, tree_matrix, tol=r):
                            target_locations.pop(check_location)
                            reached_locations.append(check_location)
                    if len(reached_locations) > 1:
                        self.make_soma_index_file(swc_translate_path, reached_locations)
            run_id += 1
            self.log(app2_command, swc_path, source_zyx=source_zyx,
                     target_locations=initial_target_locations,
                     reached_locations=reached_locations)

    def log(self, app2_command, swc_path, source_zyx=None, target_locations=None, reached_locations=None):
        # create entry for command
        self.command_logs[self.execution_id] = {}
        self.command_logs[self.execution_id]['command'] = app2_command
        # swc path prior to resample
        self.command_logs[self.execution_id]['swc_path'] = swc_path
        # resample swc path
        swc_resample_path = swc_path.replace('.swc', '_resample.swc')
        # if resample swc path exists, the command was successful
        app2_success = os.path.isfile(swc_resample_path)
        self.command_logs[self.execution_id]['app2_success'] = app2_success
        if app2_success:
            self.command_logs[self.execution_id]['swc_resample_path'] = swc_resample_path
            swc_translate_path = swc_resample_path.replace('.swc', '_translate.swc')
            if os.path.isfile(swc_translate_path):
                self.command_logs[self.execution_id]['swc_translate_path'] = swc_translate_path
                if source_zyx is not None:
                    self.command_logs[self.execution_id]['source_zyx'] = source_zyx
                    self.command_logs[self.execution_id]['target_locations'] = [key for key in target_locations.keys()]
                    self.command_logs[self.execution_id]['reached_locations'] = reached_locations
        print(self.command_logs[self.execution_id])
        self.execution_id += 1

    def write_log(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.command_logs, f, indent=4)

    @staticmethod
    def soma_reached(soma_zyx, tree, tree_matrix, tol=0):
        d = cdist(np.array([soma_zyx[2], soma_zyx[1], soma_zyx[0]]).reshape(1, 3), tree_matrix[:, 2:5])
        candidate_index = np.argsort(d.ravel())[0]
        candidate_id = tree_matrix[candidate_index, 0].astype(np.int)
        # only add soma to system if the soma center of mass is contained by
        # a tree node (app2 reconstruction can fail to reach all soma in a cc)
        return tree.node_contains(candidate_id, soma_zyx[2], soma_zyx[1], soma_zyx[0], tol=tol)

    def _marker_path(self, source_zyx, source_mass):
        marker_name = 'marker_{}_{}_{}_{}'.format(source_zyx[2], source_zyx[1], source_zyx[0], source_mass)
        marker_dir = os.path.join(self.pipeline_output_layout.tracing_dir, 'marker_files')
        marker_path = os.path.join(marker_dir, marker_name)
        if not os.path.isfile(marker_path):
            raise ValueError('marker file not found: {}'.format(marker_path))
        return marker_path

    def _generate_app2_command(self, bounds, marker_path=None, cluster_id=0, run_id=0):
        swc_name = '{}_z{}_{}_y{}_{}_x{}_{}_cluster_{}_run_{}.swc'\
                   .format(self.pipeline_arguments.image_prefix, bounds.zmin, bounds.zmax, bounds.ymin,
                           bounds.ymax, bounds.xmin, bounds.xmax, cluster_id, run_id)
        swc_path = os.path.join(self.pipeline_output_layout.reconstruction_roi_dir, swc_name)

        resolution_level = 0
        n_bytes = (bounds.zmax - bounds.zmin) * (bounds.ymax - bounds.ymin) * \
                  (bounds.xmax - bounds.xmin) * 2  # assuming 2 bps / 16bit input
        n_gb = n_bytes / (1024 * 1024 * 1024)
        while n_gb > self.pipeline_arguments.gb_mem_limit:
            resolution_level += 1
            n_gb /= 4
        length_thresh = self.length_thresh / np.power(2, resolution_level)

        # if tracing performed on input volume of imaris format directly without neural network processing,
        # use self.pipeline_arguments.channel. otherwise
        tracing_channel = 0
        if self.pipeline_arguments.app2_auto_soma and self.pipeline_arguments.input_file_format() == pymcp3d.IMARIS:
            tracing_channel = self.pipeline_arguments.channel
        else:
            tracing_channel = 0

        app2_command = [app2_bin, '\"{}\"'.format(self.pipeline_output_layout.tracing_dir), str(tracing_channel),
                        '-io {} {} {}'.format(bounds.zmin, bounds.ymin, bounds.xmin),
                        '-ie {} {} {}'.format(bounds.zmax - bounds.zmin, bounds.ymax - bounds.ymin, bounds.xmax - bounds.xmin),
                        '-lt {}'.format(length_thresh),
                        '-ct {}'.format(self.cnn_type),
                        '-rl {}'.format(resolution_level),
                        '-os \"{}\"'.format(swc_path)]
        if marker_path is not None:
            app2_command.append('-im \"{}\"'.format(marker_path))
        # if model_classes = neurite+soma, all positive voxels are foreground
        if not self.pipeline_arguments.app2_auto_soma and self.pipeline_arguments.model_classes == 'neurite+soma':
            app2_command.append('-tv 1')
        else:
            app2_command.append('-fp {}'.format(self.pipeline_arguments.fg_percent))
            app2_command.append('-by-plane')
        app2_command = ' '.join(app2_command)
        return app2_command, swc_path

    def map_swc_to_global_volume(self, swc_resample_path):
        assert os.path.isfile(swc_resample_path) and swc_resample_path.endswith('resample.swc')

        z, y, x = None, None, None
        resolution_level = None
        with open(swc_resample_path, 'r') as f:
            for l in f:
                if l.find('# offsets (zyx) = [') >= 0:
                    l = l.strip()
                    l = l.replace('# offsets (zyx) = [', '')
                    l = l.replace(']', '')
                    z, y, x = [int(token) for token in l.split(', ')]
                elif l.find('# resolution level = ') >= 0:
                    l = l.strip()
                    l = l.replace('# resolution level = ', '')
                    resolution_level = int(l)
                elif z is not None and resolution_level is not None:
                    break
        if z is None or resolution_level is None:
            raise ValueError('can not parse z, y, x offsets and resolution level from metaline')
        swc_translate_path = swc_resample_path.replace('.swc', '_translate.swc')
        neuron_tree = NeuronTree()
        neuron_tree.build_tree_from_swc(swc_resample_path)
        if neuron_tree.empty():
            print('warning: empty tree {}'.format(os.path.basename(swc_resample_path)))
            return
        print('mapping to global coordinates: {}'.format(swc_resample_path))
        scale_x = pow(2, resolution_level)
        scale_y = pow(2, resolution_level)
        # adjust for resolution level
        neuron_tree.scale_coordinates(scale_x=scale_x, scale_y=scale_y)
        neuron_tree.scale_radius(pow(scale_x * scale_y, 1 / 3))
        # translate into global coordinate
        neuron_tree.translate(delta_x=x, delta_y=y, delta_z=z)
        neuron_tree.standardize()
        neuron_tree.write_swc_file(swc_translate_path)
        return swc_translate_path

    def make_soma_index_file(self, swc_path, reached_locations):
        assert os.path.isfile(swc_path)
        assert swc_path.endswith('translate.swc')

        soma_file_path = swc_path.replace('.swc', '_soma_ind.txt')
        print('soma index file path', soma_file_path)
        with open(soma_file_path, 'w') as f:
            for reached_location in reached_locations:
                somaz, somay, somax = reached_location
                f.write('{} {} {}\n'.format(somax, somay, somaz))


