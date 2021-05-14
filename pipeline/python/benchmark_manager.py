import os
import sys
import shutil
import argparse
import numpy as np
from tifffile import imwrite
from mcp3d_clib import *
from pipeline_version import time_stamp, PipelineVersion
from compatibility.compatibility import PipelineOutputLayoutV1
from pipeline_arguments import PipelineArguments, PipelineInvocationFileReader
from pipeline_output_layout import PipelineOutputLayout
from imaris_io import ims_n_channels, read_ims
from gcut.neuron_tree import NeuronTree


class BenchmarkManager:
    # use invocation file to try find soma csv.
    # if not found soma csv path should be given
    def __init__(self, args=None):
        self._benchmark_args= None
        self._pipeline_args, self._pipeline_version = None, None
        self._pipeline_output_layout = None

        # subvolume dimensions
        self._subvolume_dims = [None, 2048, 2048]
        # soma (z, y, x): swc_name for corresponding auto reconstructed neuron
        # tuple coordinates are casted to int from original node coordinates
        self._soma_swc_pairs = {}

        self.parse_benchmark_args(args=args)
        self.write_log()

    def __str__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self._benchmark_args))

    @property
    def benchmark_args(self):
        return self._benchmark_args

    # parse argument to BenchmarkManager
    def parse_benchmark_args(self, args=None):
        parser = argparse.ArgumentParser(description='3D Processing Pipeline Benchmark Manager')
        BenchmarkManager._add_args(parser)
        if args is None:
            args = sys.argv[1:]
        self._benchmark_args = parser.parse_args(args=args)
        # reconstruct PipelineArguments and PipelineOutputLayout from invocation file
        print('reconstructing pipeline arguments from invocation file {}'.format(self.benchmark_args.invocation_file_path))
        # reconstruct self._pipeline_args, self._pipeline_version
        self.reconstruct_pipeline_args()
        print('executed by pipeline_{}'.format(self._pipeline_version))
        print(self._pipeline_args)
        # reconstruct self._pipeline_output_layout
        self.reconstruct_pipeline_layout()

        # if length of self._benchmark_args.edge_extent less than 3, append 0
        while len(self._benchmark_args.edge_extent) < 3:
            self._benchmark_args.edge_extent.append(0)
        self._subvolume_dims[0] = self._benchmark_args.subvolume_z_levels
        self._validate_benchmark_args()

        print(self)
        print('benchmarking segmented swc files under {}'.format(self.segmented_swc_dir()))

    # validate self._benchmark_args
    def _validate_benchmark_args(self):
        if not os.path.isdir(self.segmented_swc_dir()):
            raise ValueError('segmented swc directory does not exist: {}'.format(self.segmented_swc_dir()))
        if self._benchmark_args.output_format not in {'multi_page_tiff', 'tiff_sequence'}:
            raise ValueError('output format not understood: {}'.format(self._benchmark_args.output_format))
        if self._pipeline_args is not None:
            assert 0 <= self._benchmark_args.edge_extent[0] <= self._pipeline_args.input_image_dims()[0] // 2
            assert 0 <= self._benchmark_args.edge_extent[1] <= self._pipeline_args.input_image_dims()[1] // 2
            assert 0 <= self._benchmark_args.edge_extent[2] <= self._pipeline_args.input_image_dims()[2] // 2
            assert 0 < self._benchmark_args.subvolume_z_levels < self._pipeline_args.input_image_dims()[0]
        else:
            print('pipeline arguments not reconstructed. not validating benchmark arguments edge_extent and subvolume_'
                  'z_levels against input image dimensions')
        assert 0 < self._benchmark_args.soma_selection_probability <= 1

    def segmented_swc_dir(self):
        if self._pipeline_version < PipelineVersion(version_str='v.1.0'):
            return self._pipeline_output_layout.segmented_reconstruction_dir
        else:
            return self._pipeline_output_layout.segmented_reconstruction_roi_dir

    # recover argument to pipeline from invocation file
    def reconstruct_pipeline_args(self):
        reader = PipelineInvocationFileReader(self._benchmark_args.invocation_file_path)
        self._pipeline_version, self._pipeline_args = reader.reconstruct_pipeline_arguments()

    # reconstruct PipelineOutputLayout
    def reconstruct_pipeline_layout(self):
        if self._pipeline_version < PipelineVersion(version_str='v.1.0'):
            self._pipeline_output_layout = self._pipeline_args.pipeline_layout
        elif self._pipeline_version < PipelineVersion(version_str='v.2.0'):
            self._pipeline_output_layout = PipelineOutputLayoutV1(self._pipeline_args, self._pipeline_version.version_str)
        else:
            self._pipeline_output_layout = PipelineOutputLayout(self._pipeline_args, self._pipeline_version)

    def _trim_somas(self):
        selection = {}
        for (z, y, x), swc_path in self._soma_swc_pairs.items():
            if self._benchmark_args.ignore_edge_neurons:
                edge_extent = self._benchmark_args.edge_extent
                # if soma is in edge regions, discard
                if not (edge_extent[0] <= z < self._pipeline_args.input_image_dims()[0] - edge_extent[0] and
                        edge_extent[1] <= y < self._pipeline_args.input_image_dims()[1] - edge_extent[1] and
                        edge_extent[2] <= x < self._pipeline_args.input_image_dims()[2] - edge_extent[2]):
                    continue
            # bernoulli test
            draw = np.random.binomial(1, self._benchmark_args.soma_selection_probability, size=1)
            if draw == 1:
                selection[(z, y, x)] = swc_path
        self._soma_swc_pairs = selection

    # retrieve NeuronTree from swc files in self._benchmark_args.segmented_swc_dir.
    # discard soma within specified edge region. draw with specified probability.
    def _get_soma_swc_pairs(self):
        for swc_name in os.listdir(self.segmented_swc_dir()):
            if not swc_name.endswith('.swc'):
                continue
            swc_path = os.path.join(self.segmented_swc_dir(), swc_name)
            tree = NeuronTree()
            tree.build_tree_from_swc(swc_path)
            root_node = tree.tree[tree.root_id()]
            self._soma_swc_pairs[(int(root_node.z), int(root_node.y), int(root_node.x))] = swc_path
        self._trim_somas()

    # centering at soma, 100 * 2048 * 2048
    def make_subvolumes(self):
        if os.path.isdir(self._pipeline_output_layout.benchmark_subvolumes_dir):
            if self._benchmark_args.remove_existing_subvolumes:
                print('removing existing benchmark directory {}'.format(self._pipeline_output_layout.benchmark_subvolumes_dir))
                shutil.rmtree(self._pipeline_output_layout.benchmark_subvolumes_dir)
            else:
                print('existing subvolumes found. do nothing. assign true to remove_existing_subvolumes argument if you want to '
                      'remove them and create new subvolumes')
                return
        os.makedirs(self._pipeline_output_layout.benchmark_subvolumes_dir)
        self._get_soma_swc_pairs()
        with open(self._pipeline_output_layout.benchmark_soma_swc_pairs_csv_path, 'w') as f:
            f.write('# total number of neurons reconstructed by pipeline = {}\n'.format(len([swc_name for swc_name in os.listdir(self.segmented_swc_dir()) if swc_name.endswith('.swc')])))
            f.write('# total number of subvolumes generated by benchmark manager = {}\n'.format(len(self._soma_swc_pairs)))
            f.write('manual_reconstruction_complete,int(z),int(y),int(x),manual_reconstruction_swc_path\n')
        for (soma_z, soma_y, soma_x), swc_path in self._soma_swc_pairs.items():
            self.make_subvolume(soma_z, soma_y, soma_x)

    def _get_offsets(self, soma_z, soma_y, soma_x):
        return max(0, soma_z - self._subvolume_dims[0] // 2), max(0, soma_y - self._subvolume_dims[1] // 2), max(0, soma_x - self._subvolume_dims[2] // 2)

    # z_delta is offset from current plane to subvolume first z plane
    def _get_plane_offsets_str(self, soma_z, delta_z, soma_y, soma_x):
        input_image_dims = self._pipeline_args.input_image_dims()
        z_offset, y_offset, x_offset = self._get_offsets(soma_z, soma_y, soma_x)
        return 'offsets_x{}_y{}_z{}'.format(str(x_offset).zfill(len(str(input_image_dims[2]))),
                                            str(y_offset).zfill(len(str(input_image_dims[1]))),
                                            str(z_offset + delta_z).zfill(len(str(input_image_dims[0]))))

    def _get_extents(self, soma_z, soma_y, soma_x, z_offset, y_offset, x_offset):
        input_image_dims = self._pipeline_args.input_image_dims()
        z_end = min(soma_z + self._subvolume_dims[0] // 2, input_image_dims[0])
        y_end = min(soma_y + self._subvolume_dims[1] // 2, input_image_dims[1])
        x_end = min(soma_x + self._subvolume_dims[2] // 2, input_image_dims[2])
        return z_end - z_offset, y_end - y_offset, x_end - x_offset

    def _convert_subvolume_to_8bit(self, data):
        if self._benchmark_args.convert_to_8bit:
            data_max = np.max(data)
            if data_max > 0:
                data = data.astype(np.float32) * 255 / data_max
            return data.astype(np.uint8)
        else:
            return data

    def _get_subvolume(self, soma_z, soma_y, soma_x, channel):
        input_image_dims = self._pipeline_args.input_image_dims()
        assert 0 <= soma_z < input_image_dims[0] and 0 <= soma_y < input_image_dims[1] and 0 <= soma_x < input_image_dims[2]
        z_offset, y_offset, x_offset = self._get_offsets(soma_z, soma_y, soma_x)
        z_extent, y_extent, x_extent = self._get_extents(soma_z, soma_y, soma_x, z_offset, y_offset, x_offset)
        #block = pymcp3d.MImageBlock(offsets=[z_offset, y_offset, x_offset], extents=[z_extent, y_extent, x_extent])
        #self._pipeline_args.input_image.SelectView(block, channel)
        #data = self._pipeline_args.input_image.ReadData()
        data = read_ims(self._pipeline_args.image_info.channel_info(0).ImagePath(0, 0), [z_offset, y_offset, x_offset],
                        zyx_extents=[z_extent, y_extent, x_extent], channels=(channel,))
        return self._convert_subvolume_to_8bit(data)

    def write_tiff_planes(self, subvolume_dir, data, soma_z, soma_y, soma_x):
        assert os.path.isdir(subvolume_dir)
        # save subvolume planes
        for i in range(0, data.shape[0]):
            plane_offsets_str = self._get_plane_offsets_str(soma_z, i, soma_y, soma_x)
            plane_path = os.path.join(subvolume_dir, '{}_{}.tif'.format(self._pipeline_output_layout.image_prefix, plane_offsets_str))
            imwrite(plane_path, data[i, :, :])

    def write_tiff_stack(self, subvolume_dir, data, soma_z, soma_y, soma_x):
        assert os.path.isdir(subvolume_dir)
        subvolume_offsets_str = self._get_plane_offsets_str(soma_z, 0, soma_y, soma_x)
        subvolume_path = os.path.join(subvolume_dir, '{}_{}.tif'.format(self._pipeline_output_layout.image_prefix, subvolume_offsets_str))
        imwrite(subvolume_path, data)

    def write_swc_files(self, subvolume_dir, soma_z, soma_y, soma_x):
        assert os.path.isdir(subvolume_dir)
        z_offset, y_offset, x_offset = self._get_offsets(soma_z, soma_y, soma_x)
        auto_swc_path = self._soma_swc_pairs[(soma_z, soma_y, soma_x)]
        auto_tree = NeuronTree()
        auto_tree.build_tree_from_swc(auto_swc_path)
        auto_tree.translate(delta_z=-z_offset, delta_y=-y_offset, delta_x=-x_offset)
        local_auto_swc_path = os.path.join(subvolume_dir, os.path.basename(auto_swc_path).replace('.swc', '_local.swc'))
        auto_tree.write_swc_file(local_auto_swc_path)
        manual_tree = NeuronTree()
        manual_tree.add_node(auto_tree.tree[auto_tree.root_id()])
        manual_tree.write_to_meta_lines('# manual reconstruction to validate {}'.format(auto_swc_path))
        manual_reconstruction_swc_path = os.path.join(subvolume_dir, os.path.basename(auto_swc_path).replace('.swc', '_manual.swc'))
        manual_tree.write_swc_file(manual_reconstruction_swc_path)
        with open(self._pipeline_output_layout.benchmark_soma_swc_pairs_csv_path, 'a+') as f:
            f.write('{},{},{},{},{}\n'.format('', soma_z, soma_y, soma_x, manual_reconstruction_swc_path))

    def make_subvolume(self, soma_z, soma_y, soma_x):
        print('making subvolume for soma at ({}, {}, {})'.format(soma_z, soma_y, soma_x))
        volume_offsets_str = self._get_plane_offsets_str(soma_z, 0, soma_y, soma_x)
        subvolume_dir = os.path.join(self._pipeline_output_layout.benchmark_subvolumes_dir,
                                     '{}_{}'.format(self._pipeline_output_layout.image_prefix, volume_offsets_str))
        if self._benchmark_args.include_all_channels and self._pipeline_args.input_file_format() == pymcp3d.IMARIS:
            channels = [i for i in range(ims_n_channels(self._pipeline_args.image_info.channel_info(0).ImagePath(0, 0)))]
        else:
            channels = [self._pipeline_args.channel]
        for channel in channels:
            try:
                data = self._get_subvolume(soma_z, soma_y, soma_x, channel)
                if channel == self._pipeline_args.channel:
                    out_dir = subvolume_dir
                else:
                    out_dir = os.path.join(subvolume_dir, 'ch{}'.format(channel))
                os.makedirs(out_dir)
                if self._benchmark_args.output_format == 'multi_page_tiff':
                    self.write_tiff_stack(out_dir, data, soma_z, soma_y, soma_x)
                else:
                    self.write_tiff_planes(out_dir, data, soma_z, soma_y, soma_x)
            except Exception as e:
                print('error writing subvolume for channel {}'.format(channel))
                print(e.args)
        self.write_swc_files(subvolume_dir, soma_z, soma_y, soma_x)

    def measure_accuracy(self):
        pass

    def write_log(self):
        os.makedirs(self._pipeline_output_layout.benchmark_dir, exist_ok=True)
        with open(os.path.join(self._pipeline_output_layout.benchmark_dir, 'benchmark_manager_log'), 'a+') as f:
            f.write('# {}: {}\n'.format(time_stamp(output_format='%Y-%m-%d-%H:%M:%S'), self))

    @staticmethod
    def _add_args(parser):
        parser.add_argument('--invocation_file_path', nargs='?', default='', const='',
                            help='path to pipeline invocation file. if the file is not found, segmented_swc_dir must be provided and exist')
        parser.add_argument('--make_subvolumes', type=PipelineArguments.str2bool, nargs='?', default=False, const=False,
                            help='if true and no subvolume exists, create subvolumes for manual tracing. if true and subvolumes already '
                                 'created in an ealiear execution, if remove_existing_subvolumes is false, leave the subvolumes unchanged. '
                                 'if remove_existing_subvolumes is true, all existing subvolumes are removed and new subvolumes will be '
                                 'generated')
        parser.add_argument('--include_all_channels', type=PipelineArguments.str2bool, nargs='?', default=False, const=False,
                            help='if true, create subvolumes for all channels in imaris file.')
        parser.add_argument('--remove_existing_subvolumes', type=PipelineArguments.str2bool, nargs='?', default=False, const=False)
        parser.add_argument('--subvolume_z_levels', type=int, nargs='?', default=100, const=100,
                            help='number of z planes to include in the subvolume')
        parser.add_argument('--ignore_edge_neurons', type=PipelineArguments.str2bool, nargs='?', default=False, const=False,
                            help='if true, ignore soma nearing the edge of the tissue. the')
        parser.add_argument('--edge_extent', nargs='*', type=int, default=[0, 0, 0],
                            help='the extent of tissue edge along zyx axes. if z edge extent is given as 50, somas within the first 50 and '
                                 'last 50 z planes will not have subvolumes created.')
        parser.add_argument('--soma_selection_probability', type=float, nargs='?', default=1.0, const=1.0,
                            help='the probability for a soma to be selected for subvolume generation (after edge regions are discarded)')
        parser.add_argument('--convert_to_8bit', type=PipelineArguments.str2bool, nargs='?', default=True, const=True,
                            help='if true, convert subvolumes to 8 bit')
        parser.add_argument('--output_format', nargs='?', default='multi_page_tiff', const='multi_page_tiff',
                            help='output format of subvolumes. can be multi_page_tiff or tiff_sequence')
        parser.add_argument('--measure', type=PipelineArguments.str2bool, nargs='?', default=False, const=False,
                            help='if true, measure distances between manual and auto tracing')


def main(args=None):
    manager = BenchmarkManager(args=args)
    if manager.benchmark_args.make_subvolumes:
        manager.make_subvolumes()


if __name__ == '__main__':
    arguments = None
    if len(sys.argv) == 1:
        # arguments to the benchmark manager
        arguments = [
            '--invocation_file_path', '/media/muyezhu/Mumu/morph_project/raw_images/CamK-MORF3_MSNs/Camk2-MORF3-D1Tom-D2GFP_TGME02-1_30x_Str_01F/pipeline_dev_2020-09-28-21:07:33/invocation_2020-09-28-21-07.sh',
            '--make_subvolumes', 'true',
            '--include_all_channels', 'true',
            '--remove_existing_subvolumes', 'true',
            '--subvolume_z_levels', '50',
            '--ignore_edge_neurons', 'true',
            '--edge_extent', '0', '0', '0',
            '--soma_selection_probability', '1.0',
            '--convert_to_8bit', 'true',
            '--output_format', 'multi_page_tiff'
        ]
    main(args=arguments)

