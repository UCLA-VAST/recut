import os
import sys
import argparse
import re
from mcp3d_clib import *
from compatibility.mcp3d_path_v1 import Mcp3dPathDeployV1
from neural_networks.util.model_utils import ModelUtils


# this class parse command line or string arguments to pipeline into self._args
# and validates the arguments. it also returns attributes of the input image
class PipelineArgumentsV1:
    def __init__(self):
        self._args, self._args_str_list = None, None
        self._input_image, self._image_info = None, None

    def __str__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self._args))

    # parse arguments for the pipeline from sys.argv or args and validate them
    # if minimal is True, only input_dir and channel is needed,
    # _update_args call will be omitted. minimal = True is useful when parsing
    # arguments in dev mode from invocation file created in deploy mode.
    def parse_args(self, args=None, minimal=False):
        parser = argparse.ArgumentParser(description='3D Processing Pipeline')
        PipelineArgumentsV1._add_args(parser)
        if args is None:
            args = sys.argv[1:]
        self._args_str_list = args
        self._args  = parser.parse_args(args=args)
        if not minimal:
            self._update_args()
        # set input image volume
        self._input_image = pymcp3d.MImage(self.args.input_dir)
        self._input_image.ReadImageInfo(self.args.channel)
        self._image_info = self.input_image.image_info()
        self._validate_args()

    @property
    def args(self):
        return self._args

    @property
    def args_str_list(self):
        return self._args_str_list

    @property
    def commands(self):
        return self._args.commands

    @property
    def input_dir(self):
        return self._args.input_dir

    @property
    def input_image(self):
        return self._input_image

    @property
    def image_info(self):
        return self._image_info

    @property
    def image_prefix(self):
        return self.args.image_prefix

    @property
    def channel(self):
        return self.args.channel

    @property
    def trained_model_name(self):
        return self.args.trained_model_name

    @property
    def trained_model_path(self):
        return self.args.trained_model_path

    @property
    def offset(self):
        return self.args.offset

    @property
    def extent(self):
        return self.args.extent

    @property
    def gb_mem_limit(self):
        return self.args.gb_mem_limit

    @property
    def fg_percent(self):
        return self.args.fg_percent

    @property
    def app2_auto_soma(self):
        return self.args.app2_auto_soma

    @property
    def deploy_pipeline(self):
        return self.args.deploy_pipeline

    def input_file_format(self):
        return self.image_info.channel_info(self.args.channel).file_format(0)

    def n_channels(self):
        if self.input_file_format() == pymcp3d.IMARIS:
            return self.input_image.n_volumes()
        # if input image is tiff, return 0
        else:
            return 0

    def input_image_dims(self):
        return self.input_image.xyz_dims(channel=self.args.channel)

    def imaris_path(self):
        if self.input_file_format() == pymcp3d.IMARIS:
            return self.image_info.channel_info(self.args.channel).ImagePath(0, 0)
        return None

    def imaris_name(self):
        if self.input_file_format() == pymcp3d.IMARIS:
            return os.path.splitext(os.path.basename(self.imaris_path()))[0]
        else:
            return None

    # this function returns the name of zth tiff image (with file extension removed)
    # from input image volume. if the image volume is imaris format, return None
    def tiff_name(self, z, remove_z_val_postfix=True):
        if self.input_file_format() == pymcp3d.IMARIS:
            return None
        else:
            name = self.image_info.channel_info(self.args.channel).ImagePath(0, z)
            if not remove_z_val_postfix:
                return re.sub('.tif$', '', name)
            else:
                return re.sub('_Z*[0-9]+.tif$', '', name)

    def _update_args(self):
        # if soma command given, override skip_soma switch
        if 'soma' in self._args.commands:
            if self._args.app2_auto_soma:
                print('soma command overrides --app2_auto_soma argument')
                self._args.app2_auto_soma = False
        # in deployed pipeline, model name is provided to select from available models
        # and load weights and settings from pipeline/trained_models directory
        if self._args.deploy_pipeline:
            use_default = False
            if not self._args.trained_model_name:
                use_default = True
                self._args.trained_model_name = Mcp3dPathDeployV1().default_model_name
            self._args.trained_model_path = Mcp3dPathDeployV1().trained_model_path(model_name=self._args.trained_model_name)
            print('use {} packaged model: {}'.format('default' if use_default else 'user specified', self._args.trained_model_name))
        # in dev pipeline, full path to model should be provided
        else:
            self._args.trained_model_name = ModelUtils.trained_instance_name(self._args.trained_model_path)
            print('using dev model: {}'.format(self._args.trained_model_name))
        if not os.path.isfile('{}.index'.format(self._args.trained_model_path)):
            raise ValueError('trained model not found: {}.index'.format(self._args.trained_model_path))
        # command ordering
        if any([command not in self._valid_commands() for command in self._args.commands]):
            raise ValueError('unknown command encountered: {}. must be from {}'
                             .format(self._args.commands, PipelineArgumentsV1._valid_commands()))
        self._order_commands()

    # validate arguments to pipeline. input_dir and channel is not
    # validated here (except enforcing single channel tiff volume)
    # because if they are not valid, ReadImageInfo call would
    # have failed prior to reaching this function
    def _validate_args(self):
        if not 0 < self._args.fg_percent < 1:
            raise ValueError('foreground percent should be in (0, 1)')
        input_image_dims = self.input_image.xyz_dims(channel=self._args.channel)
        if not 0 <= self._args.offset[0] < input_image_dims[0]:
            raise ValueError('offset along z axis out of bounds')
        if not 0 <= self._args.offset[1] < input_image_dims[1]:
            raise ValueError('offset along y axis out of bounds')
        if not 0 <= self._args.offset[2] < input_image_dims[2]:
            raise ValueError('offset along x axis out of bounds')
        file_format = self.input_image.image_info().channel_info(self._args.channel).file_format(0)
        if file_format != pymcp3d.IMARIS:
            if self._args.channel != 0:
                raise ValueError('only 1 channel tiff input data volumes are currently supported')

    @staticmethod
    def str2bool(arg):
        return arg.lower() == 'true'

    @staticmethod
    def _add_args(parser):
        parser.add_argument('--commands', default=['soma', 'connected_components', 'app2', 'gcut'],
                            choices=PipelineArgumentsV1._valid_commands(), nargs='+')
        parser.add_argument('--input_dir', help='Directory of volume to process')
        parser.add_argument('--image_prefix', nargs='?', default='output', const='output',
                            help='prefix of pipeline outputs. suggested prefix is name of the dataset')
        parser.add_argument('--channel', nargs='?', default=0, const=0, type=int, help='channel number, relevant only for imaris files')
        parser.add_argument('--trained_model_name', nargs='?', default=None, const=None,
                            help='used to select from available models in deployed pipeline')
        parser.add_argument('--trained_model_path', nargs='?', default=None, const=None,
                            help='used to specify full path to trained model in dev pipeline')
        parser.add_argument('--offset', nargs=3, default=[0, 0, 0], type=int,
                            help="Offset for start of subregion to process in zyx order")
        parser.add_argument('--extent', nargs=3, default=[-1, -1, -1], type=int,
                            help='Extent of subregion to process, use -1 for full extent, in zyx order')
        parser.add_argument('--gb_mem_limit', nargs='?', default=3.5, const=3.5, type=float,
                            help='Use progressively downsampled volumes until size < provided mem limit')
        parser.add_argument('--fg_percent', nargs='?', default=0.01, const=0.01, type=float,
                            help='percentage of voxels to be considered foreground in the image volume')
        parser.add_argument('--app2_auto_soma', type=PipelineArgumentsV1.str2bool, nargs='?', default=False, const=False,
                            help='if true, the neural network soma segmentation as well as connected '
                                 'component analysis are skipped. app2 is evoked on raw input data. '
                                 'this argument is overridden if soma command is given')
        parser.add_argument('--deploy_pipeline', action='store_true',
                            help='if thisname flag is given, the script will use settings for deployed pipeline. '
                                 'when this flag is on, trained_model_path is ignored')

    def _order_commands(self):
        ranks = {'soma': 0, 'connected_components': 1, 'app2': 2, 'gcut': 3}
        self._args.commands.sort(key=lambda c: ranks[c])

    @staticmethod
    def _valid_commands():
        return {'soma', 'connected_components', 'app2', 'gcut'}
