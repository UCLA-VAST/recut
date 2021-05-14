from shutil import copy
from mcp3d_clib import *
from pipeline_version import time_stamp, PipelineVersion
from pipeline_arguments import PipelineArguments, PipelineInvocationFileWriter
from pipeline_output_layout import PipelineOutputLayout
from imaris_io import imaris_to_tif_sequence
from workflow.deploy_model import deploy_model
from workflow.fill_soma import fill_soma
from workflow.connected_components import find_somas, find_clusters, make_marker_files
from workflow.app2_reconstruction import App2Reconstruction
from workflow.cluster_segmentation import ClusterSegmentation


class Pipeline(object):
    def __init__(self, pipeline_arguments):
        assert isinstance(pipeline_arguments, PipelineArguments)
        self.pipeline_arguments = pipeline_arguments
        self.pipeline_version = PipelineVersion(version_str=self.pipeline_arguments.resume_from_dev_version_str,
                                                deploy=self.pipeline_arguments.deploy_pipeline)
        # self.output_layout should be used to dispense output directories
        self.output_layout = PipelineOutputLayout(self.pipeline_arguments, self.pipeline_version)

    def run(self):
        os.makedirs(self.output_layout.pipeline_version_dir, exist_ok=True)
        invocation_path = self.save_invocation()
        for command in self.pipeline_arguments.commands:
            getattr(self, command)()

    def save_invocation(self):
        invocation_save_path = os.path.join(self.output_layout.pipeline_version_dir, 'invocation_{}.sh'.format(time_stamp('%Y-%m-%d-%H-%M')))
        writer = PipelineInvocationFileWriter(invocation_save_path)
        writer.write(self.pipeline_arguments)
        return invocation_save_path

    # temporarily required by deep learning module.
    # app2 module can read imaris directly
    def convert_imaris_to_tiff(self):
        # convert to tiff sequence if
        # (1) source data is imaris and (2) conversion has not been done before
        imaris_path = self.pipeline_arguments.imaris_path()
        # if imaris_path exists and the channel tiff dir does not exist, convert
        if imaris_path:
            if not os.path.isdir(self.output_layout.tiff_sequence_dir):
                imaris_to_tif_sequence(self.pipeline_arguments.imaris_path(), channel=self.pipeline_arguments.channel,
                                       volume_name=self.pipeline_arguments.imaris_name())

    def neural_network(self):
        if self.pipeline_arguments.app2_auto_soma:
            print('pipeline is run under app2_auto_soma. not deploying neural network.')
            return
        inference_setting = deploy_model(self.pipeline_arguments.input_dir, self.output_layout.neural_net_model_root_dir,
                                         self.pipeline_arguments.trained_model_path, deploy_pipeline=self.pipeline_arguments.deploy_pipeline,
                                         output_prefix='{}_{}'.format(self.output_layout.image_prefix, self.pipeline_arguments.model_classes),
                                         mask=self.pipeline_arguments.mask)
        # fill soma if model_classes = soma
        if self.pipeline_arguments.model_classes == 'soma':
            fill_soma(inference_setting, self.output_layout.neural_net_segmentation_dir, self.output_layout.soma_fill_dir)

    def connected_components(self):
        if self.pipeline_arguments.app2_auto_soma:
            print('pipeline is run under app2_auto_soma mode. not performing connected componenet analysis.')
            return
        # connected component analysis
        find_somas(self.pipeline_arguments, self.output_layout)
        make_marker_files(self.output_layout)
        find_clusters(self.pipeline_arguments, self.output_layout)

    def app2(self):
        # generate pyramids if needed
        self.output_layout.write_tif_pyramids(self.output_layout.tracing_dir)
        app2 = App2Reconstruction(self.pipeline_arguments, self.output_layout)
        app2.trace()

    def gcut(self):
        os.makedirs(self.output_layout.segmented_reconstruction_roi_dir, exist_ok=True)
        for name in os.listdir(self.output_layout.reconstruction_roi_dir):
            if not name.endswith('resample_translate.swc'):
                continue
            swc_path = os.path.join(self.output_layout.reconstruction_roi_dir, name)
            soma_file_path = os.path.join(self.output_layout.reconstruction_roi_dir, name.replace('.swc', '_soma_ind.txt'))
            if not os.path.isfile(soma_file_path):
                copy(swc_path, self.output_layout.segmented_reconstruction_roi_dir)
            else:
                segmentor = ClusterSegmentation(swc_path, soma_file_path)
                segmentor.segment()
                segmentor.save(self.output_layout.segmented_reconstruction_roi_dir)

