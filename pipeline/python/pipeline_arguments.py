import os
import sys
import argparse
import re
import cv2
from mcp3d_clib import *
from compatibility.compatibility import PipelineArgumentsV0, PipelineArgumentsV1
from pipeline_version import current_version, dated_deploy_version, dev_version_format, PipelineVersion
from mcp3d_path import Mcp3dPathDeploy, Mcp3dPathDev
from trained_models_catalogue import default_options, deploy_models_catalogue


# PipelineArguments parses arguments to pipeline. arguments are processed and validated. an pymcp3d.MImage instance
# associated with --input_dir of pipeline arguments is also maintained.
class PipelineArguments:
    # parse arguments for the pipeline from the list of argument strings in args_str_list
    # if args_str_list is None, parse from sys.argv.
    # if process_args is False, _process_args call will be omitted. process_args = False is useful when parsing
    # arguments from invocation file created by a differing pipeline version or differing dev/deploy mode
    def __init__(self, args_str_list=None, process_args=True):
        self._args, self._args_str_list = None, None
        self._input_image, self._image_info = None, None
        self._trained_model_path = None
        # mask image. can be None
        self._mask = None
        # use sys.argv if args_str_list is None
        if args_str_list is None:
            args_str_list = sys.argv[1:]
        self._args_str_list = args_str_list
        # parse arguments
        self._parse_args(process_args)
        # set input image volume
        self._input_image = pymcp3d.MImage(self.args.input_dir)
        self._input_image.ReadImageInfo(self.args.channel)
        self._image_info = self.input_image.image_info()
        # validate arguments
        self._validate_args()

    def __str__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self._args))

    def _parse_args(self, process_args):
        parser = argparse.ArgumentParser(description='3D Processing Pipeline')
        PipelineArguments._add_args(parser)
        self._args = parser.parse_args(args=self._args_str_list)
        if process_args:
            self._process_args()

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
    def mask(self):
        return self._mask

    @property
    def channel(self):
        return self.args.channel

    @property
    def label_technique(self):
        return self.args.label_technique

    @property
    def model_classes(self):
        return self.args.model_classes

    @property
    def trained_model_path(self):
        return self._trained_model_path

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

    @property
    def resume_from_dev_version_str(self):
        return self.args.resume_from_dev_version

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

    def _process_args(self):
        # if neural_network command given, override skip_soma switch
        if 'neural_network' in self._args.commands:
            if self._args.app2_auto_soma:
                print('neural_network command overrides --app2_auto_soma argument')
                self._args.app2_auto_soma = False
        # in deployed pipeline, label_technique and trained_model_classes are provided to select from available models
        # and load weights and settings from pipeline/trained_models directory.
        # if the (label_technique, trained_model_classes) combination does not have a associated trained model,
        # use label_technique and trained_model_classes from trained_models_catalogue.default_options
        if self._args.deploy_pipeline:
            if not (self._args.label_technique, self._args.model_classes) in deploy_models_catalogue:
                print('no trained models associated with ({}, {})'.format(self._args.label_technique, self._args.model_classes))
                self._args.label_technique = default_options['label_technique']
                self._args.model_classes = default_options['model_classes']
                print('using default options ({}, {})'.format(self._args.label_technique, self._args.model_classes))
            self._trained_model_path = Mcp3dPathDeploy().trained_model_path(label_technique=self._args.label_technique, model_classes=self._args.model_classes)
            self._args.resume_from_dev_version = None
        # in dev pipeline, if the (label_technique, trained_model_classes) combination does not have a associated
        # trained model, error is raised from Mcp3dPathDev().trained_model_path
        else:
            self._trained_model_path = Mcp3dPathDev().trained_model_path(label_technique=self._args.label_technique, model_classes=self._args.model_classes)
            if re.match(dev_version_format(), self._args.resume_from_dev_version):
                print('resuming execution for pipiline_{}'.format(self._args.resume_from_dev_version))
            else:
                self._args.resume_from_dev_version = None

        # if mask is not 'none', try to read it from under input dir. set non zero elements to 1
        if not self._args.mask == 'none':
            mask_path = os.path.join(self._args.input_dir, 'mask', self._args.mask)
            if not os.path.isfile(mask_path):
                raise ValueError('can not find mask image at {}'.format(mask_path))
            self._mask = cv2.imread(mask_path, -1)
            self._mask[self._mask > 0] = 1

        # command ordering
        if any([command not in self._valid_commands() for command in self._args.commands]):
            raise ValueError('unknown command encountered: {}. must be from {}'
                             .format(self._args.commands, PipelineArguments._valid_commands()))
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
                            choices=PipelineArguments._valid_commands(), nargs='+')
        parser.add_argument('--input_dir', help='Directory of volume to process')
        parser.add_argument('--image_prefix', nargs='?', default='output', const='output',
                            help='prefix of pipeline outputs. suggested prefix is name of the dataset. '
                                 'ignored when pipeline input is an imaris file')
        parser.add_argument('--channel', nargs='?', default=0, const=0, type=int,
                            help='channel number, relevant only for imaris files')
        parser.add_argument('--label_technique', nargs='?', default='morph', const='morph',
                            help='used to specify labeling technique of the dataset')
        parser.add_argument('--model_classes', nargs='?', default='soma', const='soma',
                            help='used to select from available models trained to recognize different neuronal components')
        parser.add_argument('--mask', nargs='?', default='none', const='none',
                            help='name of the mask image. the mask image should be placed under input_dir/mask')
        parser.add_argument('--offset', nargs=3, default=[0, 0, 0], type=int,
                            help="Offset for start of subregion to process in zyx order")
        parser.add_argument('--extent', nargs=3, default=[-1, -1, -1], type=int,
                            help='Extent of subregion to process, use -1 for full extent, in zyx order')
        parser.add_argument('--gb_mem_limit', nargs='?', default=3.5, const=3.5, type=float,
                            help='Use progressively downsampled volumes until size < provided mem limit')
        parser.add_argument('--fg_percent', nargs='?', default=0.01, const=0.01, type=float,
                            help='percentage of voxels to be considered foreground in the image volume. '
                                 'only relevant if model_classes is \"soma\" ')
        parser.add_argument('--app2_auto_soma', type=PipelineArguments.str2bool, nargs='?', default=False, const=False,
                            help='if true, the neural network soma segmentation as well as connected '
                                 'component analysis are skipped. app2 is evoked on raw input data. '
                                 'this argument is overridden if soma command is given')
        parser.add_argument('--deploy_pipeline', action='store_true',
                            help='if this flag is given, the script will use settings for deployed pipeline.')
        parser.add_argument('--resume_from_dev_version', nargs='?', default='', const='',
                            help='if given should be a valid dev version string. pipeline will resume execution for given dev '
                                 'version output under input_dir. this argument is ignored in deployed pipeline.')

    def _order_commands(self):
        ranks = {'neural_network': 0, 'connected_components': 1, 'app2': 2, 'gcut': 3}
        self._args.commands.sort(key=lambda c: ranks[c])

    @staticmethod
    def _valid_commands():
        return {'neural_network', 'connected_components', 'app2', 'gcut'}


class PipelineInvocationFile:
    def __init__(self, invocation_file_path, mode):
        assert mode in 'rw'
        self.invocation_file_path = invocation_file_path
        self.mode = mode
        if self.mode == 'r':
            if not os.path.isfile(self.invocation_file_path):
                raise ValueError('invocation file path does not exist: {}'.format(self.invocation_file_path))


class PipelineInvocationFileReader(PipelineInvocationFile):
    def __init__(self, invocation_file_path):
        super(PipelineInvocationFileReader, self).__init__(invocation_file_path, 'r')

    def read_version_str(self):
        with open(self.invocation_file_path, 'r') as f:
            for l in f:
                if l.startswith('# version ='):
                    return l.strip().replace('# version = ', '')
        # < v.1.0, where # version = line does not exist
        return dated_deploy_version()

    @staticmethod
    def _get_args_str_list(line):
        args = []
        arg = ''
        quote_id = None
        for c in line.strip():
            if c == '\'':
                quote_id = 0 if quote_id is None else quote_id + 1
                continue
            if quote_id is None:
                continue
            # for non quote character, gather into arg
            if quote_id % 2 == 0:
                arg += c
            # white space character between quoted arguments
            else:
                args.append(arg)
                arg = ''
        # handle argument immediately before EOL
        if len(arg) > 0:
            args.append(arg)
        return args

    def reconstruct_pipeline_arguments(self):
        invocation_version = PipelineVersion(version_str=self.read_version_str())
        with open(self.invocation_file_path, 'r') as f:
            for l in f:
                if l.startswith('#'):
                    continue
                args_str_list = PipelineInvocationFileReader._get_args_str_list(l)
                print(args_str_list)
                # version specific
                if invocation_version < PipelineVersion(version_str='v.1.0'):
                    return invocation_version, PipelineArgumentsV0(args=args_str_list)
                elif invocation_version < PipelineVersion(version_str='v.2.0'):
                    pipeline_arguments = PipelineArgumentsV1()
                    pipeline_arguments.parse_args(args=args_str_list, minimal=True)
                    return invocation_version, pipeline_arguments
                return invocation_version, PipelineArguments(args_str_list=args_str_list, process_args=False)
        raise ValueError('warning: no argument found in {}'.format(self.invocation_file_path))


class PipelineInvocationFileWriter(PipelineInvocationFile):
    def __init__(self, invocation_file_path):
        super(PipelineInvocationFileWriter, self).__init__(invocation_file_path, 'w')

    def write(self, pipeline_arguments):
        assert isinstance(pipeline_arguments, PipelineArguments)
        with open(self.invocation_file_path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('# version = {}\n'.format(current_version(pipeline_arguments.args.deploy_pipeline)))
            args = ['\'{}\''.format(arg) for arg in pipeline_arguments.args_str_list]
            f.write('python3 run_pipeline.py {}\n'.format(' '.join(args)))
