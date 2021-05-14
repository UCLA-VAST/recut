from abc import ABCMeta, abstractmethod
import math
import numpy as np
import cv2
import tensorflow as tf
# https://github.com/tensorflow/tensorflow/issues/14875
Dataset = tf.data.Dataset
from ..util.configurations import BatchIteratorConfiguration
from .image_reader import ContextPairedReader, ContextUnpairedReader
from .image_writer import TiffImageWriter
from .image_transformation import TransformationEngine, identity_engine
from .label_rules import LabelRules, dummy_label_rules


class BatchIteratorBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, label_rules=dummy_label_rules, transformation_engine=identity_engine):
        assert isinstance(config, BatchIteratorConfiguration)
        assert isinstance(label_rules, LabelRules)
        assert isinstance(transformation_engine, TransformationEngine)
        # shared with image reader
        self.config = config

        # self.config.start_scale:
        # resolution of input image.
        # if start_scale = 0, this is full resolution image.
        # if start_scale = i > 0, input image is down sampled to 1/2^i of its
        # original size

        # self.config.n_scales:
        # number of scales to read the image at.
        # the image has already been downsampled to obtain corresponding
        # start_scale. if n_scales = 1, a region of patch height and width is
        # read. if n_scales = i, i scales are read. at each scale s, the patch
        # retrieve effectively patch_height * 2^s and patch_width * 2^s of
        # input area, and downsample the patch to the fixed
        # patch_height * patch_width dimension

        self._image_reader = None
        # when self._training is False, instantiate an image writer
        self._image_writer = None
        self._label_rules = label_rules
        self._engine = transformation_engine
        # different inheriting classes can initialize these variables differently
        self._batch_images, self._batch_label = None, None
        self._patch_images, self._patch_label = None, None

        # total iteration number occurred
        self._batch_number = 0

        self._init_patch_position_variables()

    # iter protocol. returns self as iterator.
    # inheriting classes should implement their own __next__ method
    def __iter__(self):
        return self

    # the callable passed to tf.data.Dataset.from_generator
    def make_iterator(self):
        return self

    @abstractmethod
    def output_tensor_shapes(self):
        pass

    @abstractmethod
    def dataset(self):
        """
        if output shapes are not given at all errors is thrown.
        https://github.com/tensorflow/tensorflow/issues/24520
        its possible TensorShape with None arguments can work too, but here
        we know the tensor dimensions
        :return: Dataset.from_generator, the generator being self
        """
        pass

    @property
    def label_rules(self):
        return self._label_rules

    @abstractmethod
    def n_batches_per_epoch(self):
        pass

    @abstractmethod
    def n_patches_per_epoch(self):
        pass

    # yield batches
    @abstractmethod
    def next_training_batch(self):
        pass

    # yield batches, dataset name and spatial positions
    # spatial positions for each patch is currently coded as
    # [dataset name, [zoffset, yoffset, xoffset], [zextent, yextent, xextent]]
    # offsets are valid offset, extents are valid extents (accounted for overlap)
    # offset + extent can be greater than image dimension
    # (aka patch is only partially in boundary)
    #
    @abstractmethod
    def next_inference_batch(self):
        pass

    @abstractmethod
    def init_image_writer(self, output_dir, output_format='tiff', output_prefix=''):
        pass

    def write_batch(self, batch_images, valid_patch_positions, mask=None):
        assert self._image_writer
        clipped_patches = self.clip(batch_images)
        self._image_writer.write_patches(clipped_patches, valid_patch_positions, mask=mask)

    def flush_image_writer(self, mask=None):
        self._image_writer.flush(mask=mask)

    def clip(self, batch_image):
        """
        clips batch_image to valid region.
        batch_image must have consistent shapes with batch iterator
        :param batch_image:
        :return: clipped batch_image as np array
        """
        if isinstance(batch_image, tf.Tensor):
            batch_image = batch_image.numpy()
        if self.config.patch_x_overlap == 0 and self.config.patch_y_overlap == 0:
            return batch_image
        if batch_image.ndim == 4:
            assert batch_image.shape == (self.config.batch_size, self.config.patch_height, self.config.patch_width, self.config.sequence_length)
            return batch_image[...,
                               self.config.patch_y_overlap: self.config.patch_height - self.config.patch_y_overlap,
                               self.config.patch_x_overlap:self.config.patch_width - self.config.patch_x_overlap, :]
        else:
            assert batch_image.shape == (self.config.batch_size, self.config.patch_height, self.config.patch_width)
            return batch_image[:, self.config.patch_y_overlap: self.config.patch_height - self.config.patch_y_overlap,
                                  self.config.patch_x_overlap: self.config.patch_width - self.config.patch_x_overlap]

    def tally(self):
        """
        this method is used to obtain size of the data, how many patches/batches
        does one epoch of data contain
        """

    def empty(self):
        return self._image_reader.empty()

    # return current epoch number
    @abstractmethod
    def epoch_id(self):
        pass

    def _init_patch_position_variables(self):
        # the start values can be negative, while valid start values are always
        # in bounds. in the patch, the valid region is the center rectangle
        # if patch portion outside of the valid region maps to outside of image
        # bounds, black is padded
        self._x_start = None
        self._y_start = None
        self._x_start_valid = None
        self._y_start_valid = None

    # scale is relative to start scale.
    # with self.start_scale = 1, n_scales = 2, this function should recieve
    # arguments 0 and 1, which reads at original on disk image at scales 1 and 2
    def _read_image_sequence_patch_at_scale(self, image_context, scale):
        assert 0 <= scale < self.config.n_scales
        # crop from full size image
        image_height, image_width = image_context[0].shape
        crop_height = self.config.patch_height * pow(2, scale)
        crop_width = self.config.patch_width * pow(2, scale)
        scale_x_start = self._x_start - (crop_width - self.config.patch_width) // 2
        scale_y_start = self._y_start - (crop_height - self.config.patch_height) // 2
        scale_x_end = self._x_start + self.config.patch_width + (crop_width - self.config.patch_width) // 2
        scale_y_end = self._y_start + self.config.patch_height + (crop_height - self.config.patch_height) // 2
        scale_valid_x_start = max(scale_x_start, 0)
        scale_valid_y_start = max(scale_y_start, 0)
        scale_valid_x_end = min(scale_x_end, image_width)
        scale_valid_y_end = min(scale_y_end, image_height)

        crop_x_offset = 0 if scale_x_start > 0 else -scale_x_start
        crop_y_offset = 0 if scale_y_start > 0 else -scale_y_start
        crop_x_extent = scale_valid_x_end - scale_valid_x_start
        crop_y_extent = scale_valid_y_end - scale_valid_y_start

        for i in range(self.config.sequence_length):
            image_crop = np.zeros(shape=(crop_height, crop_width), dtype=np.uint16)
            image_crop[crop_y_offset: crop_y_offset + crop_y_extent,
                       crop_x_offset: crop_x_offset + crop_x_extent] = \
                image_context[i][scale_valid_y_start: scale_valid_y_end,
                                 scale_valid_x_start: scale_valid_x_end]
            image_crop = self._scale_crop(image_crop, scale)
            self._patch_images[:, :, scale * self.config.sequence_length + i] = image_crop

    def _scale_crop(self, image_crop, scale):
        while scale > self.config.start_scale:
            image_crop = cv2.pyrDown(image_crop)
            scale -= 1
        return image_crop

    def _preprocess_image(self):
        # put range to [0, 1]
        self._batch_images /= np.iinfo(np.uint16).max
        if self.config.preprocess_method == 'none':
            return
        # center or norm
        for i in range(self.config.batch_size):
            mean = np.mean(self._batch_images[i, ...])
            self._batch_images[i, ...] -= mean
        if self.config.preprocess_method == 'center':
            return
        elif self.config.preprocess_method == 'norm':
            for i in range(self.config.batch_size):
                std = np.std(self._batch_images[i, ...])
                # guard against all black padding patches
                if std > 0:
                    self._batch_images[i, ...] /= std
        else:
            raise ValueError('preprocess method not understood')


# only support tiff format
class ContextPairedBatchIterator(BatchIteratorBase):
    def __init__(self, config, label_rules=dummy_label_rules, transformation_engine=identity_engine):
        super(ContextPairedBatchIterator, self).__init__(config, label_rules=label_rules, transformation_engine=transformation_engine)
        self._image_reader = ContextPairedReader(self.config, self._label_rules)
        # if multiple scales present, patches are organized on last dimension as
        # [context0_scale0, context1_scale0, context0_scale1, context1_scale1]
        self._batch_images = np.zeros(shape=(self.config.batch_size, self.config.patch_height, self.config.patch_width,
                                             self.config.sequence_length * self.config.n_scales),
                                      dtype=np.float32)
        self._batch_labels = np.zeros(shape=(self.config.batch_size, self.config.patch_height, self.config.patch_width),
                                      dtype=np.int32)
        self._patch_images = np.zeros(shape=(self.config.patch_height, self.config.patch_width,
                                             self.config.n_scales * self.config.sequence_length),
                                      dtype=np.uint16)
        self._patch_label = np.zeros(shape=(self.config.patch_height, self.config.patch_width), dtype=np.int32)
        # number of iterations needed to see whole data once (on average)
        # equal to number of patches in the whole data divided by batch size
        self._n_batches_per_epoch = None
        self._n_batches_per_shard = None
        # number of total batches yielded
        self._batch_number = 0
        # number of iterations occurred within current shard
        self._shard_batch_number = 0
        self.tally()

    def __next__(self):
        return self.next_training_batch()

    def output_tensor_shapes(self):
        return (tf.TensorShape(self._batch_images.shape),
                tf.TensorShape((self._batch_images.shape[0], self._batch_images.shape[1], self._batch_images.shape[2], 1)))

    def dataset(self):
        return Dataset.from_generator(self.make_iterator, (np.float32, np.int32),
                                      output_shapes=self.output_tensor_shapes())

    def epoch_id(self):
        return self._batch_number // self._n_batches_per_epoch

    def n_batches_per_epoch(self):
        return self._n_batches_per_epoch

    def n_patches_per_epoch(self):
        return self._image_reader.n_patches()

    def next_training_batch(self):
        # if entire shard has been seen, determine if new shard needed
        if self._shard_batch_number > self._n_batches_per_shard:
            if not self.config.data_fit_in_memory:
                print('reading new data shard')
                self._image_reader.read_shard()
            self._shard_batch_number = 0
        for i in range(self.config.batch_size):
            # get image label pair
            image_context, label = self._image_reader.next_context()
            # extract patch
            while True:
                self.read_patch(image_context, label)
                # if high sample rate not required, continue to next patch
                if not self.config.high_sample_rate:
                    break
                if tf.reduce_sum(self._patch_label) > 0:
                    break
                # if label has no non background voxels, continue to next
                # patch
                if tf.reduce_sum(label) == 0:
                    break
            # transform patch
            self._batch_images[i, ...], self._batch_labels[i, ...] = \
                self._engine.transform(self._patch_images, labels=self._patch_label)
        # preprocess
        self._preprocess_image()
        self._batch_number += 1
        self._shard_batch_number += 1
        # must give a pair of tensors or a pair of numpy arrays
        return self._batch_images, np.expand_dims(self._batch_labels, -1)

    def read_patch(self, image_context, label):
        assert len(image_context) > 0, 'empty image context'
        self._set_random_patch_position(image_context)
        for scale in range(self.config.n_scales):
            self._read_image_sequence_patch_at_scale(image_context, scale)
        self._read_label_patch(label)

    # use patch valid patch height and width as argument to n_patches()
    def tally(self):
        if self._n_batches_per_epoch is not None:
            return
        self._n_batches_per_epoch = self.n_patches_per_epoch() // self.config.batch_size
        self._n_batches_per_shard = self._image_reader.n_shard_patches() // self.config.batch_size
        if self.config.data_fit_in_memory:
            assert self._n_batches_per_epoch == self._n_batches_per_shard

    def init_image_writer(self, output_dir, output_format='tiff', output_prefix=''):
        raise NotImplementedError('ContextPairedBatchIterator does not have an ImageWriter attribute')

    def _set_random_patch_position(self, image_context):
        xmax = image_context[0].shape[1] - 1
        ymax = image_context[0].shape[0] - 1
        self._x_start_valid = np.random.random_integers(0, high=xmax)
        self._y_start_valid = np.random.random_integers(0, high=ymax)
        self._x_start = self._x_start_valid - self.config.patch_x_overlap
        self._y_start = self._y_start_valid - self.config.patch_y_overlap

    def _read_label_patch(self, label):
        self._patch_label.fill(0)
        # crop from full size label
        image_height, image_width = label.shape
        valid_x_start = max(self._x_start, 0)
        valid_y_start = max(self._y_start, 0)
        x_end = self._x_start + self.config.patch_width
        y_end = self._y_start + self.config.patch_height
        valid_x_end = min(x_end, image_width)
        valid_y_end = min(y_end, image_height)
        patch_x_offset = 0 if self._x_start > 0 else -self._x_start
        patch_y_offset = 0 if self._y_start > 0 else -self._y_start
        patch_x_extent = valid_x_end - valid_x_start
        patch_y_extent = valid_y_end - valid_y_start
        self._patch_label[patch_y_offset: patch_y_offset + patch_y_extent,
                          patch_x_offset: patch_x_offset + patch_x_extent] = \
            label[valid_y_start: valid_y_end, valid_x_start: valid_x_end]


class ContextUnpairedBatchIterator(BatchIteratorBase):
    def __init__(self, config, label_rules=dummy_label_rules):
        super(ContextUnpairedBatchIterator, self).__init__(config, label_rules=label_rules)
        self._image_reader = ContextUnpairedReader(self.config)
        # if multiple scales present, patches are organized on last dimension as
        # [context0_scale0, context1_scale0, context0_scale1, context1_scale1]
        self._batch_images = np.zeros(shape=(self.config.batch_size, self.config.patch_height, self.config.patch_width,
                                             self.config.sequence_length * self.config.n_scales),
                                      dtype=np.float32)
        self._patch_images = np.zeros(shape=(self.config.patch_height, self.config.patch_width,
                                             self.config.n_scales * self.config.sequence_length),
                                      dtype=np.uint16)

        self._volume_zoffset, self._volume_yoffset, self._volume_xoffset = None, None, None
        # number of batches needed to see whole data once (on average)
        # equal to number of patches in the whole data divided by batch size
        self._n_batches_per_epoch = None
        self._n_batches_per_shard = None
        # number of patches in shard. for inference data this is the same
        # as number of patches within each z plane
        self._n_shard_patches = None
        # number of patches read from current shards. for inference data this is
        # the same as number of patches that have been read from current z plane
        self._shard_patch_number = None
        # number of total batches yielded
        self._batch_number = 0
        # number of iterations occurred within current shard
        self._shard_batch_number = 0

        self.tally()

    def __next__(self):
        if self.config.training:
            return self.next_training_batch()
        else:
            return self.next_inference_batch()

    def output_tensor_shapes(self):
        if self.config.training:
            return (tf.TensorShape(self._batch_images.shape),)
        else:
            return (tf.TensorShape(self._batch_images.shape),
                    tf.TensorShape((self.config.batch_size, 6)))

    def dataset(self):
        if self.config.training:
            return Dataset.from_generator(self.make_iterator, (np.float32,), output_shapes=self.output_tensor_shapes())
        else:
            return Dataset.from_generator(self.make_iterator, (np.float32, np.int32), output_shapes=self.output_tensor_shapes())

    def init_image_writer(self, output_dir, output_format='tiff', output_prefix=''):
        if self.config.training:
            raise NotImplementedError('training mode ContextUnpairedBatchIterator instance does not have ImageWriter attribute')
        else:
            dataset_name, dataset_dims = self._image_reader.dataset_dims().popitem()
            if output_format == 'tiff':
                self._image_writer = TiffImageWriter(output_dir, dataset_dims, self._label_rules.n_labels(), output_prefix=output_prefix)
            else:
                raise NotImplementedError('only tiff output format is currently implemented')

    def tally(self):
        # be precise here. inference batches need to exhaust all patches
        self._n_batches_per_epoch = math.ceil(self.n_patches_per_epoch() / self.config.batch_size)
        self._n_shard_patches = self._image_reader.n_shard_patches()
        # todo: self._n_batches_per_shard

    def n_batches_per_epoch(self):
        return self._n_batches_per_epoch

    # use valid yx extents to n_patches()
    def n_patches_per_epoch(self):
        return self._image_reader.n_patches()

    def _retrieve_image_context(self):
        # if no image context has been read, or all patches from one shard
        # have been extracted, request the next context
        # for inference data, return image context along with the zyx offsets of
        # the context. for training data, return image context without the offsets
        new_shard = False
        if self._shard_patch_number is None or self._shard_patch_number == self._n_shard_patches:
            self._shard_patch_number = 0
            new_shard = True
        return self._image_reader.next_context(new_shard=new_shard)

    def next_training_batch(self):
        """
        training/validation/test batches
        image_reader.next_context() returns a single data patch
        only a single scale is supported
        :return:
        """
        assert self.config.training
        for i in range(self.config.batch_size):
            self._batch_images[i, ...] = self._retrieve_image_context()
            self._shard_patch_number += 1
        self._preprocess_image()
        return self._batch_images

    def next_inference_batch(self):
        """
        inference time batches
        iterates infinitely
        :return:
        """
        assert not self.config.training
        patch_positions = np.zeros(shape=(self.config.batch_size, 6), dtype=np.int32)
        for i in range(self.config.batch_size):
            image_context, self._volume_zoffset, self._volume_yoffset, self._volume_xoffset = self._retrieve_image_context()
            # extract one patch
            self._batch_images[i, ...], plane_yoffset, plane_xoffset = \
                self.read_patch(image_context, reset_patch_position=(self._shard_patch_number == 0))
            # at the end of dataset there can be a few all black patches
            patch_positions[i] = np.array([self._volume_zoffset,
                                           self._volume_yoffset + plane_yoffset,
                                           self._volume_xoffset + plane_xoffset,
                                           1, self.config.valid_patch_height, self.config.valid_patch_width])
            self._shard_patch_number += 1
        self._preprocess_image()
        return self._batch_images, patch_positions

    def read_patch(self, image_context, reset_patch_position=False):
        assert len(image_context) > 0, 'empty image context'
        self._set_sequential_patch_position(image_context, reset=reset_patch_position)
        for scale in range(self.config.n_scales):
            self._read_image_sequence_patch_at_scale(image_context, scale)
        yoffset = self._y_start_valid
        xoffset = self._x_start_valid
        return self._patch_images, yoffset, xoffset

    # attempt to set next sequential patch position
    # if no more patch can be extracted from current context, return False
    def _set_sequential_patch_position(self, image_context, reset=False):
        if reset:
            self._reset_sequential_patch_position()
        else:
            start_scale_width = image_context[0].shape[1]
            start_scale_height = image_context[0].shape[0]
            if self._x_start_valid + self.config.valid_patch_width < start_scale_width:
                self._x_start_valid += self.config.valid_patch_width
                self._x_start = self._x_start_valid - self.config.patch_x_overlap
            elif self._y_start_valid + self.config.valid_patch_height < start_scale_height:
                self._x_start_valid = 0
                self._x_start = - self.config.patch_x_overlap
                self._y_start_valid += self.config.valid_patch_height
                self._y_start = self._y_start_valid - self.config.patch_y_overlap
            else:
                raise ValueError('image is exhausted of patches')

    def _reset_sequential_patch_position(self):
        self._x_start_valid = 0
        self._y_start_valid = 0
        self._x_start = - self.config.patch_x_overlap
        self._y_start = - self.config.patch_y_overlap


class InstanceBatchIterator(BatchIteratorBase):
    def __init__(self, config):
        super(InstanceBatchIterator, self).__init__(config)
