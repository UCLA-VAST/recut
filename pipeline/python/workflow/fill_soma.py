import os
import numpy as np
import cv2
from mcp3d_clib import *
from neural_networks.data_io.image_reader import ContextUnpairedReader


def fill_soma(inference_setting, segmentation_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_reader = ContextUnpairedReader(inference_setting.get_batch_iterator_config())
    print(segmentation_dir)
    segmentation_image = pymcp3d.MImage(segmentation_dir)
    segmentation_image.ReadImageInfo(0)
    segmentation_info = segmentation_image.image_info()
    # TODO: this is nasty. replace with proper iteration
    if input_reader._dataset_dims[input_reader._dataset_names[0]] != segmentation_info.channel_info(0).xyz_dims():
        raise ValueError('volumes at {} and {} have different dimensions'
                         .format(input_reader._dataset_dirs[input_reader._dataset_names[0]], segmentation_dir))
    for z in range(segmentation_info.channel_info(0).zdim()):
        segmentation_path = segmentation_info.channel_info(0).ImagePath(0, z)
        print('fill z = {} with {}'.format(z, os.path.basename(segmentation_path)))
        img = input_reader.read_zplane(z)
        segmentation = cv2.imread(segmentation_path, 0)
        target_val = np.percentile(img, 99.9)
        img[segmentation > 0] = target_val
        # segmentation path already uses --image_prefix parameter
        cv2.imwrite(os.path.join(output_dir, os.path.basename(segmentation_path).replace('.tif', '_filled.tif')), img)
