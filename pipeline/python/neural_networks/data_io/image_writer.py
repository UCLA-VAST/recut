from abc import ABCMeta, abstractmethod
import os
import numpy as np
import cv2
import tensorflow as tf


"""
ImageWriter objects will write patches to image volume at designated positions
it does not verify if a given location has already been written to before. 
when is the volume writing complete should be controlled at higher level
"""


class ImageWriter(object):
    def __init__(self, output_dir, volume_dims, n_model_classes, output_prefix=''):
        self.output_dir = output_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.volume_dims = volume_dims
        assert len(self.volume_dims) == 3, 'volume dimensions should be rank 3'
        assert all(map(lambda x: x > 0, self.volume_dims)), 'all volume dimensions should be positive'
        self.n_model_classes = n_model_classes
        assert self.n_model_classes >= 2
        self.output_prefix = output_prefix

    def volume_zdim(self):
        return self.volume_dims[0]

    def volume_ydim(self):
        return self.volume_dims[1]

    def volume_xdim(self):
        return self.volume_dims[2]

    def _pad_zval(self, zval):
        zmax_str = str(self.volume_zdim() - 1)
        return str(zval).zfill(len(zmax_str))

    @abstractmethod
    def flush(self, mask=None):
        pass


# TiffImageWriter assumes that (1) the z planes in a volume will be traversed in order (2) all patches in a z plane
# will be exhausted before next plane is traversed
class TiffImageWriter(ImageWriter):
    def __init__(self, output_dir, volume_dims, n_model_classes, output_prefix=''):
        super(TiffImageWriter, self).__init__(output_dir, volume_dims, n_model_classes, output_prefix=output_prefix)
        self.current_z = None
        self.current_tiff = None

    # patches passed should be clipped to valid regions
    def write_patches(self, clipped_patches, patch_positions, mask=None):
        assert clipped_patches.shape[0] == len(patch_positions)
        for i in range(len(patch_positions)):
            zoffset, yoffset, xoffset, zextent, yextent, xextent = patch_positions[i].numpy()
            assert zoffset < self.volume_zdim()
            assert zextent == 1
            # if a new z offset is seen, write tiff buffer to disk if tiff buffer exists
            if not zoffset == self.current_z:
                # save current tiff to disk
                if self.current_tiff is not None:
                    self.write_current_tiff(mask=mask)
                # if current_tiff is None, create buffer
                else:
                    self.current_tiff = np.zeros(shape=(self.volume_ydim(), self.volume_xdim()), dtype=np.int32)
                # update current z offset
                self.current_z = zoffset
            # write patches to tiff plane
            y_valid_extent = min(self.volume_ydim(), yoffset + yextent) - yoffset
            x_valid_extent = min(self.volume_xdim(), xoffset + xextent) - xoffset
            self.current_tiff[yoffset: yoffset + y_valid_extent, xoffset: xoffset + x_valid_extent] = clipped_patches[i][0: y_valid_extent, 0: x_valid_extent]

    def write_current_tiff(self, mask=None):
        self.current_tiff *= np.iinfo(np.uint16).max // (self.n_model_classes - 1)
        if mask is not None:
            assert mask.shape == self.current_tiff.shape, 'mask must have the same dimension as the image z plane'
            self.current_tiff *= mask
        cv2.imwrite(self.current_tiff_path(), self.current_tiff.astype(np.uint16))

    def flush(self, mask=None):
        if self.current_z is not None:
            self.write_current_tiff(mask=mask)

    def current_tiff_path(self):
        assert self.current_z is not None
        return os.path.join(self.output_dir, '{}_Z{}.tif'.format(self.output_prefix, self._pad_zval(self.current_z)))



