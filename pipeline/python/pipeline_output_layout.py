from mcp3d_clib import *
from misc.bounding_box import BoundingBox
from pipeline_version import dated_deploy_version, PipelineVersion
from pipeline_arguments import PipelineArguments


class PipelineOutputLayout:
    def __init__(self, pipeline_arguments, pipeline_version):
        assert isinstance(pipeline_arguments, PipelineArguments)
        assert isinstance(pipeline_version, PipelineVersion)
        self.pipeline_arguments = pipeline_arguments
        self.pipeline_version = pipeline_version

    def verify_version_pipeline_arguments(self):
        # input_dir attribute should exists for all version
        assert hasattr(self.pipeline_arguments.args, 'input_dir')

    @property
    def image_prefix(self):
        if self.pipeline_arguments.input_file_format() == pymcp3d.IMARIS:
            return self.pipeline_arguments.imaris_name()
        else:
            return self.pipeline_arguments.args.image_prefix

    @property
    def tiff_sequence_dir(self):
        # if input_dir contains imaris file, return the directory where
        # converted tiff will be placed
        if self.pipeline_arguments.input_file_format() == pymcp3d.IMARIS:
            imaris_name = os.path.splitext(os.path.basename(self.pipeline_arguments.imaris_path()))[0]
            tiff_sequence_dir = os.path.join(self.pipeline_arguments.args.input_dir,
                                             '{}_ch{}_tif'.format(imaris_name, self.pipeline_arguments.args.channel))
            return tiff_sequence_dir
        # else input_dir itself has tiff sequence data
        else:
            return self.pipeline_arguments.args.input_dir

    @property
    # each pipeline version generate outputs in isolated directories
    def pipeline_version_dir(self):
        return os.path.join(self.pipeline_arguments.args.input_dir, 'pipeline_{}'.format(self.pipeline_version))

    @property
    def pipeline_version_channel_dir(self):
        return os.path.join(self.pipeline_version_dir, 'ch{}'.format(self.pipeline_arguments.args.channel))

    @property
    def pipeline_version_channel_neural_net_dir(self):
        return os.path.join(self.pipeline_version_channel_dir,
                            '{}_{}'.format(self.pipeline_arguments.args.label_technique, self.pipeline_arguments.args.model_classes))

    @property
    def neural_net_model_root_dir(self):
        return os.path.join(self.pipeline_version_channel_neural_net_dir, '{}_segmentation'.format(self.pipeline_arguments.args.model_classes))

    @property
    def neural_net_segmentation_dir(self):
        return os.path.join(self.neural_net_model_root_dir, 'segmentation')

    @property
    def soma_fill_dir(self):
        if self.pipeline_arguments.model_classes == 'neurite+soma':
            raise NotImplementedError('soma_fill_dir is not implemented for model_classes neurite+soma')
        return os.path.join(self.pipeline_version_channel_neural_net_dir, 'soma_fill')

    @property
    # directory from which tracing and cc analysis for neural clusters is performed
    # csv files for soma and neural clusters are also written into this directory
    def tracing_dir(self):
        if self.pipeline_arguments.app2_auto_soma:
            return self.pipeline_arguments.input_dir
        if self.pipeline_arguments.model_classes == 'neurite+soma':
            return self.neural_net_segmentation_dir
        else:
            return self.soma_fill_dir

    @property
    def soma_csv_path(self):
        return os.path.join(self.tracing_dir, '{}_soma_centers_mass_ge_10000.csv'.format(self.image_prefix))

    @property
    def cluster_csv_name(self):
        if self.pipeline_arguments.model_classes == 'soma':
            return '{}_clusters_fg_{}.csv' .format(self.image_prefix, self.pipeline_arguments.fg_percent)
        else:
            return '{}_clusters.csv'.format(self.image_prefix)

    @property
    def cluster_csv_path(self):
        return os.path.join(self.tracing_dir, self.cluster_csv_name)

    @property
    def marker_file_dir(self):
        return os.path.join(self.tracing_dir, 'marker_files')

    @staticmethod
    # determine roi bounds
    def get_roi_bounds(image_dims, offset, extent):
        roi_zmin = offset[0]
        roi_zmax = image_dims[0] if extent[0] < 0 else extent[0] + roi_zmin
        roi_ymin = offset[1]
        roi_ymax = image_dims[1] if extent[1] < 0 else extent[1] + roi_ymin
        roi_xmin = offset[2]
        roi_xmax = image_dims[2] if extent[2] < 0 else extent[2] + roi_xmin
        return BoundingBox(zmin=roi_zmin, zmax=roi_zmax, ymin=roi_ymin,
                           ymax=roi_ymax, xmin=roi_xmin, xmax=roi_xmax)

    @property
    def reconstruction_dir(self):
        return os.path.join(self.pipeline_version_channel_neural_net_dir, 'reconstruction')

    @property
    def reconstruction_roi_dir(self):
        input_image_dims = self.pipeline_arguments.input_image_dims()
        roi_bounds = self.get_roi_bounds(input_image_dims, self.pipeline_arguments.args.offset, self.pipeline_arguments.args.extent)
        roi_str = 'roi_{}_{}_{}_{}_{}_{}'.format(roi_bounds.zmin, roi_bounds.zmax, roi_bounds.ymin,
                                                 roi_bounds.ymax, roi_bounds.xmin, roi_bounds.xmax)
        return os.path.join(self.reconstruction_dir, 'reconstruction_fg_{}_{}'.format(self.pipeline_arguments.args.fg_percent, roi_str))

    @property
    def segmented_reconstruction_roi_dir(self):
        return os.path.join(self.reconstruction_roi_dir, 'segmentation')

    @property
    def benchmark_dir(self):
        return os.path.join(self.pipeline_version_channel_neural_net_dir, 'benchmark')

    @property
    def benchmark_subvolumes_dir(self):
        return os.path.join(self.benchmark_dir, 'subvolumes')

    @property
    def benchmark_soma_swc_pairs_csv_path(self):
        return os.path.join(self.benchmark_dir, 'soma_swc_pairs.csv')

    # write pyramids for tif sequences.
    def write_tif_pyramids(self, tif_dir):
        if tif_dir == self.pipeline_arguments.args.input_dir:
            image = self.pipeline_arguments.input_image
        else:
            image = pymcp3d.MImage(tif_dir)
            # tif sequence volume generated by pipeline are single channel volume
            # TODO: it's possible to have multi channel lightsheet data (set up in c++ backend).
            image.ReadImageInfo(0)
        if image.n_pyr_levels(channel=0) < 3:
            print('generating image pyramids')
            image.WriteImagePyramids(0)
        image.SaveImageInfo()
