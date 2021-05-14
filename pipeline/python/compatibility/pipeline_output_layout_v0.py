from mcp3d_clib import *


class PipelineOutputLayoutV0:
    def __init__(self, args, model_instance_name):
        self.args = args
        self.model_instance_name = model_instance_name
        self.input_image = pymcp3d.MImage(self.args.input_dir)
        self.input_image.ReadImageInfo(0)
        self.image_info = self.input_image.image_info()

    @property
    def input_image_dims(self):
        return self.input_image.xyz_dims()

    @property
    def imaris_path(self):
        if self.image_info.channel_info(0).file_format(0) == pymcp3d.IMARIS:
            return os.path.join(self.args.input_dir, self.image_info.channel_info(0).ImagePath(0, 0))
        return None

    @property
    # compatibility property
    def image_prefix(self):
        return self.args.image_prefix

    @property
    def tiff_sequence_dir(self):
        # if input_dir contains imaris file, return the directory where
        # converted tiff will be placed
        if self.image_info.channel_info(0).file_format(0) == pymcp3d.IMARIS:
            # ImagePath return type is seen as bytes by python...
            imaris_name = os.path.splitext(self.image_info.channel_info(0).ImagePath(0, 0))[0]
            # cover different naming conventions prior to v.1.0
            tiff_sequence_dir0 = os.path.join(self.args.input_dir, '{}_resolution0_tif'.format(imaris_name), 'ch0')
            tiff_sequence_dir1 = os.path.join(self.args.input_dir, '{}_resolution0_tif'.format(self.args.image_prefix), 'ch0')
            if os.path.isdir(tiff_sequence_dir0):
                return tiff_sequence_dir0
            elif os.path.isdir(tiff_sequence_dir1):
                return tiff_sequence_dir1
            else:
                raise ValueError('neither {} not {} exists'.format(tiff_sequence_dir0, tiff_sequence_dir1))
        # else input_dir itself has tiff sequence data
        else:
            return self.args.input_dir

    @property
    # pipeline outputs will be relative to image_dir, instead of self.args.input_dir
    def image_dir(self):
        # if skip_soma flag is passed, the input directory should be passed to
        # app2 directly as image volume directory
        if self.args.skip_soma:
            return self.args.input_dir
        # otherwise tiff sequence data is (or was) needed for neural network
        else:
            return self.tiff_sequence_dir

    @property
    def neural_net_root_dir(self):
        return os.path.join(self.image_dir, 'soma_segmentation')

    @property
    def neural_net_dir(self):
        return os.path.join(self.neural_net_root_dir, self.model_instance_name)

    @property
    def segmentation_dir(self):
        return os.path.join(self.neural_net_dir, 'segmentation')

    @property
    def soma_fill_dir(self):
        return os.path.join(self.image_dir, 'soma_fill', self.model_instance_name)

    @property
    def soma_csv_path(self):
        return os.path.join(self.soma_fill_dir, '{}_soma_centers_mass>10000.csv'.format(self.args.image_prefix))

    @property
    def marker_file_dir(self):
        return os.path.join(self.soma_fill_dir, 'marker_files')

    @property
    def reconstruction_dir(self):
        roi_str = 'roi_{}_{}_{}_{}_{}_{}'.format(self.args.offset[0], self.args.extent[0] if self.args.extent[0] > 0 else self.input_image_dims[0],
                                                 self.args.offset[1], self.args.extent[1] if self.args.extent[0] > 0 else self.input_image_dims[1],
                                                 self.args.offset[2], self.args.extent[2] if self.args.extent[0] > 0 else self.input_image_dims[2])
        if not self.args.skip_soma:
            return os.path.join(self.soma_fill_dir, 'reconstruction_fg_{}_{}'.format(self.args.fg_percent, roi_str))
        else:
            return os.path.join(self.image_dir, 'reconstruction_fg_{}_{}'.format(self.args.fg_percent, roi_str))

    @property
    def segmented_reconstruction_dir(self):
        return os.path.join(self.reconstruction_dir, 'segmentation')

    @property
    # compatibility property
    def benchmark_dir(self):
        return self.args.input_dir

    @property
    # compatibility property
    def benchmark_subvolumes_dir(self):
        return os.path.join(self.benchmark_dir, 'subvolumes')

    @property
    # compatibility property
    def benchmark_soma_swc_pairs_csv_path(self):
        return os.path.join(self.benchmark_dir, 'soma_swc_pairs.csv')

    def write_pyramids(self, image_dir):
        if image_dir == self.args.input_dir:
            image = self.input_image
        else:
            image = pymcp3d.MImage()
            image.ReadImageInfo(image_dir)
        if image.n_pyr_levels() < 3:
            print('generating image pyramids')
            image.WriteImagePyramids(0)
        image.SaveImageInfo()
