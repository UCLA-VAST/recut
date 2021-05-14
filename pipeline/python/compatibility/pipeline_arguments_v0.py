import sys
import os
import argparse
from mcp3d_clib import *
from .mcp3d_path_v0 import Mcp3dPathV0
from .pipeline_output_layout_v0 import PipelineOutputLayoutV0
from neural_networks.util.model_utils import ModelUtils


# this class is based on pipeline_2020_02_01 run_pipeline.ArgDispatch class. modification was made to maintain
# compatible api with other versions
class PipelineArgumentsV0(object):
    def __init__(self, args=None):
        parser = argparse.ArgumentParser(description='3D Processing Pipeline')
        self._add_commands(parser)
        self._add_params(parser)
        if args is not None:
            self.args, unknown = parser.parse_known_args(args)
        else:
            self.args, unknwon = parser.parse_known_args(sys.argv[1:])
        # hard code model instance name shared by dated pipeline pre v.1.0 versions: e.g. 2020-02-01 to 2020-03-18
        model_instance_name = 'ContextualUNetV1_200129_1051'
        # self.pipeline_layout should be used to dispense directories
        self.pipeline_layout = PipelineOutputLayoutV0(self.args, model_instance_name)
        self._validate_args()

    def __str__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self.args))

    @property
    # compatibility property
    # no channel argument was passed in command line for this class. the effective channel number used throughout
    # pipeline is 0
    def channel(self):
        return 0

    @property
    # compatibility property
    def input_image(self):
        return self.pipeline_layout.input_image

    # compatibility function
    def input_image_dims(self):
        return self.pipeline_layout.input_image_dims

    def _validate_args(self):
        if not os.path.isdir(self.args.input_dir):
            raise ValueError('{} is not a directory'.format(self.args.input_dir))
        if not 0 < self.args.fg_percent < 1:
            raise ValueError('foreground percent should be in (0, 1)')
        if not 0 <= self.args.offset[0] < self.pipeline_layout.input_image_dims[0]:
            raise ValueError('offset along z axis out of bounds')
        if not 0 <= self.args.offset[1] < self.pipeline_layout.input_image_dims[1]:
            raise ValueError('offset along y axis out of bounds')
        if not 0 <= self.args.offset[2] < self.pipeline_layout.input_image_dims[2]:
            raise ValueError('offset along x axis out of bounds')

    def _add_commands(self, parser):
        parser.add_argument('commands', choices=self._valid_subcommands(), nargs='+')

    def _add_params(self, parser):
        # may or may not want to move memory limitations here
        parser.add_argument('--input_dir', default=None, help='Directory of volume to process')
        parser.add_argument('--image_prefix', default='output',
                            help='prefix of pipeline outputs. suggested prefix is name of the dataset')
        parser.add_argument('--trained_model_path', default=None,
                            help='path to trained model. only used if deploy_pipeline flag is off')
        parser.add_argument('--offset', nargs=3, default=[0, 0, 0], type=int,
                            help="Offset for start of subregion to process in zyx order")
        parser.add_argument('--extent', nargs=3, default=[-1, -1, -1], type=int,
                            help='Extent of subregion to process, use -1 for full extent, in zyx order')
        parser.add_argument('--gb_mem_limit', default=3.5, type=float,
                            help='Use progressively downsampled volumes until size < provided mem limit')
        parser.add_argument('--fg_percent', default=0.01, type=float,
                            help='percentage of voxels to be considered foreground in the image volume')
        parser.add_argument('--skip_soma', action='store_true',
                            help='if this flag is given, the neural network soma segmentation as well as '
                                 'connected component analysis are skipped. app2 is evoked on raw input data. '
                                 'this flag is overridden if soma command is given')
        parser.add_argument('--deploy_pipeline', action='store_true',
                            help='if this flag is given, the script will use settings for deployed pipeline. '
                                 'when this flag is on, trained_model_path is ignored')

    def _order_commands(self):
        ranks = {'soma': 0, 'connected_components': 1, 'app2': 2, 'gcut': 3}
        self.args.commands.sort(key=lambda c: ranks[c])

    @classmethod
    def _valid_subcommands(cls):
        return {'soma', 'connected_components', 'app2', 'gcut'}
