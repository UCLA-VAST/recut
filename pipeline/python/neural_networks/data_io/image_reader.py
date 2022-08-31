import re
from abc import ABCMeta
from collections import defaultdict
from functools import reduce
import math
from random import choices
import bisect
import numpy as np
import cv2
from mcp3d_clib import *
from ..util.configurations import BatchIteratorConfiguration
from .label_rules import LabelRules


# has_paired_data: if true, only support tiff format
# retrieve pattern:
#   - context: image0, image1, image2, label1[if has_paired data] (n_contexts = 3)
#   - instance: image, label0, label1, ..., labeln[if has_paired data]
#   - sequence: image0, image1, image2..., label1, ... labeln
# sequence_length: covers all retrieve patterns and corresponds to number of
# input images in a retrieved unit of image and label (if paired data required)
# n_shards: data that does not fit in memory are consisted of shards,
# with n_shards describing number of shards. a single shard of data should
# fit it memory. for data that entirely fit in memory, n_shards equal to 1
# has_paired_data: if true, data should be pairs of (image(s), label(s))
# paired data should be under dataset_dir/image and dataset_dir/label
# unpaired data is under dataset_dir. if false, image is not paired with label,
# and pymcp3d will be used. only tiff format is supported for paired data. for
# unpaired data, it is assumed that each dataset is a continuous volume
# training: refer to if the model is performing training or inference
# operation. image reader will consider all training, validation, testing
# datasets to fall under training type data, as opposed to inference data,
# which does not need to be randomized and augmented. if is false, assume that
# traversal through a potentially large volume of tiff, ims or tiled tiff format
# is wanted. pymcp3d will be used. data is to be traversed in deterministic order.
# paired data is always considered training data, while unpaired data can be
# either training or inference phase data
# return image as np.uint16 (scale np.uint8 if needed) and label as np.int32
class ImageReaderBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, label_rules=None):
        assert isinstance(config, BatchIteratorConfiguration)
        self.config = config

        if self.config.data_fit_in_memory:
            self.config.n_shards = 1

        # key: dataset_name, val: dataset_path
        self._dataset_dirs = {}
        self._dataset_names = []
        self._find_dataset_names()

        # different inheriting classes implement this differently
        self._shard_images = None
        self._shard_labels = None

        if label_rules is not None:
            assert isinstance(label_rules, LabelRules)
        self._label_rules = label_rules

    def _find_dataset_names(self):
        for dataset_path in self.config.dataset_paths:
            if os.path.isdir(dataset_path):
                dataset_name = os.path.basename(dataset_path)
                self._dataset_dirs[dataset_name] = dataset_path
                self._dataset_names.append(os.path.basename(dataset_name))
            else:
                print('{} is not a directory, ignoring'.format(dataset_path))

    def translate_labels(self):
        if self._label_rules is None:
            raise ValueError('no assigned LabelRule instance')
        for dataset_name, dataset_labels in self._shard_labels.items():
            for label_name in dataset_labels:
                print('translating labels: {}'.format(label_name))
                self._shard_labels[dataset_name][label_name] = \
                    self._label_rules.translate_to_output(self._shard_labels[dataset_name][label_name] )

    def scale_images(self):
        if self.config.start_scale == 0:
            return
        scale = 0
        while scale < self.config.start_scale:
            for dataset_name in self._dataset_names:
                self._scale_dataset_images(dataset_name)
                self._scale_dataset_labels(dataset_name)
            scale += 1

    def _scale_dataset_images(self, dataset_name):
        dataset_images = self._shard_images[dataset_name]
        for image_name in dataset_images:
            dataset_images[image_name] = cv2.pyrDown(dataset_images[image_name])

    def _scale_dataset_labels(self, dataset_name):
        dataset_labels = self._shard_labels[dataset_name]
        for label_name in dataset_labels:
            label = dataset_labels[label_name]
            dataset_labels[label_name] = cv2.resize(label, (label.shape[0] // 2,
                                                            label.shape[1] // 2),
                                                    cv2.INTER_NEAREST)

    def convert_input_dtype(self, input_image):
        assert input_image.dtype == np.uint8 or input_image.dtype == np.uint16
        # should preserve the ratio of image_max / dtype_max
        if input_image.dtype == np.uint8:
            scale_factor = np.iinfo(np.uint16).max / np.iinfo(np.uint8).max
            output_image = input_image.astype(np.float32) * scale_factor
            output_image = np.floor(output_image)
            return output_image.astype(np.uint16)
        else:
            return input_image


class ContextPairedReader(ImageReaderBase):
    def __init__(self, configuration, label_rules):
        super(ContextPairedReader, self).__init__(configuration, label_rules=label_rules)
        assert self.config.training
        assert self.config.has_paired_data
        # key: dataset name. val: list, sorted image names of a data set
        # only images required by its paired label are included
        self._image_names = defaultdict(list)
        # key: dataset name. val: list, sorted label names of a data set
        # not all image needs to have paring label
        self._label_names = defaultdict(list)
        # self._shard_images: key dataset name, value: {image name: image sequence}
        self._shard_images = defaultdict(dict)
        # self._label_images: key dataset name, value: {label name: label}
        self._shard_labels = defaultdict(dict)
        # difference from self._label_names: only include label names drawn for
        # the shard.accordingly
        self._shard_label_names = defaultdict(list)
        # key: dataset name. val: label_name: image_names
        self._label_to_image_mapping = defaultdict(dict)
        # pupulate self._image_names, self._label_names,
        # self._label_to_image_names
        self._find_pair_names()
        self._n_labels = sum([len(value) for value in self._label_names.values()])
        self._n_shard_labels = int(math.ceil(self._n_labels / self.config.n_shards))

    # each shard should contain same number of labeled images. if total number
    # of label images is not wholely divisible by n_shards, number of label
    # images per shard is rounded up: ceil(total_n_labels / n_shards) per shard
    def read_shard(self):
        if self.config.data_fit_in_memory:
            # if data fit in memory entirely, and shard_labels is not empty,
            # data has been read already, nothing needs to be done
            if len(self._shard_labels) > 0:
                return
            self._shard_label_names = self._label_names
        # draw shard labels
        else:
            self._shard_label_names.clear()
            self._shard_images.clear()
            self._shard_labels.clear()
            names = []
            for dataset_name, label_names in self._label_names.items():
                for label_name in label_names:
                    names.append((dataset_name, label_name))
            np.random.shuffle(names)
            # for single shard datasets, all image-label pairs are read once
            # for multi-shard datasets, chance of selecting from a particular
            # dataset is proportional to number of labeled examples the dataset has
            for i in range(self._n_shard_labels):
                choice = np.random.randint(0, high=len(names))
                choice_dataset, choice_label_name = names[choice]
                self._shard_label_names[choice_dataset].append(choice_label_name)
                names.pop(choice)
        # read shard data
        for dataset_name in self._shard_label_names.keys():
            print('reading from dataset {}...'.format(dataset_name))
            self._read_shard_dataset(dataset_name)
        # translate labels
        self.translate_labels()
        # scale if needed
        self.scale_images()

    def _read_shard_dataset(self, dataset_name):
        for label_name in self._shard_label_names[dataset_name]:
            print('reading label {}'.format(label_name))
            label_path = os.path.join(self._dataset_dirs[dataset_name], 'label', label_name)
            self._shard_labels[dataset_name][label_name] = cv2.imread(label_path, -1)
            assert self._shard_labels[dataset_name][label_name].dtype == np.uint8, 'label image must be uint8'
            for image_name in self._label_to_image_mapping[dataset_name][label_name]:
                if image_name is not None:
                    image_path = os.path.join(self._dataset_dirs[dataset_name], 'image', image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)
                    ### modified, convert all non-zero pixels to 255
                    image[image != 0] = 255
                    self._shard_images[dataset_name][image_name] = self.convert_input_dtype(image)

    def n_shard_patches(self):
        if len(self._shard_labels) == 0:
            self.read_shard()
        n = 0
        for dataset_labels in self._shard_labels.values():
            for label in dataset_labels.values():
                n += (label.shape[0] // self.config.patch_height + int(label.shape[0] % self.config.patch_height > 0)) * \
                     (label.shape[1] // self.config.patch_width + int(label.shape[1] % self.config.patch_width > 0))
        return n

    # if data does not fit in memory, estimate using patch number from a single
    # shard
    def n_patches(self):
        if len(self._shard_labels) == 0:
            self.read_shard()
        return self.config.n_shards * self.n_shard_patches()

    # single label and its image contexts
    def next_context(self):
        if len(self._shard_labels) == 0:
            self.read_shard()
        # draw a random dataset in shard. chance of a dataset being drawn is proportional
        # to the number of labeled examples in it
        shard_dataset_names = [key for key in self._shard_label_names.keys()]
        dataset_name = choices([name for name in self._shard_labels.keys()],
                               weights=[len(self._shard_labels[name]) for name in self._shard_labels.keys()])[0]
        #dataset_id = np.random.randint(0, high=len(shard_dataset_names))
        #dataset_name = shard_dataset_names[dataset_id]
        # draw a random label from the selected dataset
        label_id = np.random.randint(0, high=len(self._shard_label_names[dataset_name]))
        label_name = self._shard_label_names[dataset_name][label_id]
        # retrieve the drawn label and its context
        image_contexts = []
        height, width = self._shard_labels[dataset_name][label_name].shape
        for image_name in self._label_to_image_mapping[dataset_name][label_name]:
            # if image context is None, pad np.zeros
            if image_name is None:
                image_contexts.append(np.zeros(shape=(height, width), dtype=np.uint16))
            else:
                image_contexts.append(self._shard_images[dataset_name][image_name])
        label = self._shard_labels[dataset_name][label_name]
        self._valid_context_shapes(image_contexts, label)
        return image_contexts, label

    def _find_pair_names(self):
        empty_datasets = []
        for dataset_name in self._dataset_names:
            # return false if no label found for dataset
            if self._find_dataset_label_names(dataset_name):
                # match image and label index. self._image_names should only
                # have required image names
                self._find_dataset_label_to_image_mapping(dataset_name)
            else:
                empty_datasets.append(dataset_name)
        for dataset_name in empty_datasets:
            self._dataset_names.remove(dataset_name)
            self._dataset_dirs.pop(dataset_name)

    def _find_dataset_label_names(self, dataset_name):
        # find and sort label names
        label_names = os.listdir(os.path.join(self._dataset_dirs[dataset_name], 'label'))
        label_names.sort()
        for label_name in label_names:
            if label_name.find('.tif') > 0:
                self._label_names[dataset_name].append(label_name)
        # remove key dataset_name from self._label_names if dataset has no label
        if len(self._label_names[dataset_name]) == 0:
            self._label_names.pop(dataset_name)
            return False
        return True

    def _find_dataset_label_to_image_mapping(self, dataset_name):
        image_names = []
        for image_name in os.listdir(os.path.join(self._dataset_dirs[dataset_name], 'image')):
            if image_name.find('.tif') > 0:
                image_names.append(image_name)
        image_names.sort()
        required_image_names = set()
        for label_name in self._label_names[dataset_name]:
            matched_image_name = ContextPairedReader._matching_image_name(label_name)
            matched_index = bisect.bisect_left(image_names, matched_image_name)
            if not image_names[matched_index] == matched_image_name:
                raise ValueError('can not find image name matching {}'.format(label_name))
            self._label_to_image_mapping[dataset_name][label_name] = \
                self._matching_image_context_names(image_names, matched_index, pad_none=True)
            required_image_names.update(self._label_to_image_mapping[dataset_name][label_name])
        required_image_names.discard(None)
        self._image_names[dataset_name] = sorted(required_image_names)

    @staticmethod
    def _matching_image_name(label_name):
        return re.sub(re.compile('_label_.*tif'), '.tif', label_name)

    def _matching_image_context_names(self, image_names, image_index,
                                      pad_none=False):
        context_names = []
        for i in range(- self.config.sequence_length // 2 + 1, self.config.sequence_length // 2 + 1):
            index = image_index + i
            if 0 <= index < len(image_names):
                context_names.append(image_names[index])
            elif pad_none:
                context_names.append(None)
        return context_names

    @staticmethod
    def _valid_context_shapes(contexts, label):
        for context_image in contexts:
            assert context_image.shape == label.shape


class ContextUnpairedReader(ImageReaderBase):
    """
    currently assuming channel 0 and pyr_level 0 is to be read for imaris files
    next_context() method:
    (1) for inference data, return full tiff plane sequences and the position of
        the center tiff image
    (2) for training data, return a patch of dimension
        (patch_height, patch_width, sequence_length). patch positions are not
        returned, but will be maintained internally for testing purpose
    for training data, each data shard currently is hard coded to hold 8GB of
    data in memory, instead of calculating shard size from self.n_shards value,
    since it may be impractical to know the size of entire collection of
    training datasets in advance for a user
    due to io performance issue, will retrieve imaris files by reading an entire
    chunk at a time. should implement threading. while training is performed
    on shard data, next shard is being read
    """
    def __init__(self, configuration):
        super(ContextUnpairedReader, self).__init__(configuration)
        assert not self.config.has_paired_data

        # dataset_name: MImage instance
        self._mimages = {}
        # dataset_name: data dimensions (zyx)
        self._dataset_dims = {}

        self._read_dataset_infos()
        self._check_dataset_constraints()

        self._current_dataset_id = None
        # for training data, list elements are 3d arrays with dimensions patch_height * patch_width * seq_length
        # for inference data, only imaris format uses it. at the moment, imaris data retrival is in unit of a volume of
        # chunk_zdim * ydim * xdim and kept in self._shard_images until the sequential traversal goes beyond the volume.
        self._shard_images = []
        # for training data, hold dataset id and zyx offsets for patches in self._shard_images
        # for inference imaris data, hold z offset of top plane of the volume in self._shard_images
        self._shard_image_offsets = []
        # only relevant for inference data. length of list is equal to seq_length.
        # list elements are 2d arrays with dimensions tiff_height * tiff_width
        self._image_context = []
        # zyx offset of context center image in current dataset
        # for tiff images, y and x offset should both be zero, since
        # an entire tiff z plane is read within shard. for ims files,
        # xy planes may be partitioned to allow more efficient retrieval of
        # chunked data
        self._zoffset, self._yoffset, self._xoffset  = None, None, None
        # for inference instances, self._shard_images key: dataset_name,
        # val: {zval: zplane image}

    def n_datasets(self):
        return len(self._dataset_names)

    def _read_dataset_infos(self):
        for dataset_name in self._dataset_names:
            self._mimages[dataset_name] = pymcp3d.MImage(self._dataset_dirs[dataset_name])
            self._mimages[dataset_name].ReadImageInfo(0, ignore_saved=True)
            self._dataset_dims[dataset_name] = self._mimages[dataset_name].xyz_dims(pyr_level=self.config.start_scale)

    # for random dispensing of training data, only imaris data is supported
    # additionally, only a single scale is supported
    # for inference phase, only a single tiff dataset is supported
    def _check_dataset_constraints(self):
        if len(self._dataset_names) == 0:
            raise ValueError('no dataset found')
        if self.config.training:
            if not all([self._dataset_format(dataset_name) == pymcp3d.IMARIS
                        for dataset_name in self._dataset_names]):
                raise ValueError('only support imaris unpaired training data')
            if self.config.n_scales > 1:
                raise ValueError('only support unpaired training data with 1 scale')
        else:
            if len(self._dataset_names) > 1:
                raise ValueError('only support a single inference dataset')

    def _dataset_format(self, dataset_name):
        """
        :return: format of the dataset as pymcp3d constant
        """
        return self._mimages[dataset_name].image_info().channel_info(0).file_format(0)

    def _dataset_zdim(self, dataset_name):
        return self._dataset_dims[dataset_name][0]

    def _dataset_ydim(self, dataset_name):
        return self._dataset_dims[dataset_name][1]

    def _dataset_xdim(self, dataset_name):
        return self._dataset_dims[dataset_name][2]

    # return a copy of self._dataset_dims.
    def dataset_dims(self):
        return self._dataset_dims.copy()

    def _shard_zdim(self):
        if self.config.training:
            raise NotImplementedError('not implemented')
        return 1

    def _shard_ydim(self,):
        if self.config.training:
            raise NotImplementedError('not implemented')
        dataset_name = self._dataset_names[0]
        return self._dataset_dims[dataset_name][1]

    def _shard_xdim(self):
        if self.config.training:
            raise NotImplementedError('not implemented')
        dataset_name = self._dataset_names[0]
        return self._dataset_dims[dataset_name][2]

    # number of patches in a shard of data
    def n_shard_patches(self):
        """
        number of patches in shard. separate implementation for training and
        inference data. different format and dataset may return different results.
        for inference data, patch overlap is accounted for. for training data,
        patch overlap is ignored and result is calculated with zero overlap.
        in inference data, this number is calculated as number of patches in a single z plane, which is
        what is used by batch iterator to request data from image reader. to the batch iterator, each shard
        is centered around a z plane. whether image reader retrieve data shards of different size for
        efficiency reason is not known to batch iterator
        :return:
        """
        if self.config.training:
            return self.n_shard_bytes_training() // self.n_patch_bytes()
        else:
            dataset_name = self._dataset_names[0]
            # training: patches in a single z plane
            ydim = self._dataset_ydim(dataset_name)
            xdim = self._dataset_xdim(dataset_name)
            n_ypatches = math.ceil(ydim / self.config.valid_patch_height)
            n_xpatches = math.ceil(xdim / self.config.valid_patch_width)
            return n_ypatches * n_xpatches

    # number of patches needed to traverse the entire data once. image dimensions
    # are used to calculate result. for training data, patch overlap is ignored
    # and result is calculated with zero overlap
    def n_patches(self):
        if not self.config.training:
            dataset_name = self._dataset_names[0]
            return self._dataset_zdim(dataset_name) * self.n_shard_patches()
        else:
            # find total number of bytes across all datasets
            n_total_bytes = sum([reduce(lambda x, y: x * y, self.dataset_dims()[dataset_name])
                                 for dataset_name in self._dataset_names]) * 2
            # number of patches in full shards
            n_shards = n_total_bytes // self.n_shard_bytes_training()
            n_patches_per_shard = self.n_shard_bytes_training() // self.n_patch_bytes()
            n = n_shards * n_patches_per_shard
            # add number of patches remaining outside of full shards
            return n + n_total_bytes % self.n_shard_bytes_training() // self.n_patch_bytes()

    def n_shard_bytes_training(self):
        if not self.config.training:
            raise NotImplementedError('not implemented for inference data')
        return 8 * (1024 ** 3)

    # assuming 16bit image data
    def n_patch_bytes(self):
        return self.config.sequence_length * self.config.patch_height * self.config.patch_width * 2

    def read_shard(self):
        """
        read shard
        for each dataset, read amount of data proportional to their total size
        within each dataset, random z planes are retrieved if self.training is
        true. otherwise data is sequentially traversed
        if self.training is False, each shard should contain only one image
        context. but each subsequent context only need to read one new z plane
        and discard one old z plane
        """
        if self.config.training:
            self.read_shard_random()
        else:
            self.read_shard_sequential()

    def read_shard_random(self):
        """
        read next random shard
        for efficiency reasons only support ims format
        read random patches to self._shard_images till shard size is full
        """
        assert self.config.training
        self._shard_images.clear()
        self._shard_image_offsets.clear()
        n_shard_patches = self.n_shard_patches()
        for i in range(0, n_shard_patches):
            if i % (n_shard_patches // 10) == 0:
                print('{}: read random shard: {}% complete'.format(self.__class__.__name__, i // (n_shard_patches // 10) * 10))
            self._read_shard_random_imaris()

    def _read_shard_random_imaris(self):
        self._set_next_random_position()
        self._shard_images.append(np.zeros(shape=(self.config.patch_height, self.config.patch_width,
                                                  self.config.sequence_length), dtype=np.uint16))
        dataset_name = self._dataset_names[self._current_dataset_id]
        z_start = max(0, self._zoffset - self.config.sequence_length // 2)
        z_end = min(self._dataset_zdim(dataset_name), self._zoffset + self.config.sequence_length // 2 + 1)
        block = pymcp3d.MImageBlock([z_start, self._yoffset, self._xoffset],
                                    [z_end - z_start, self.config.patch_height, self.config.patch_width])
        self._mimages[dataset_name].SelectView(block, 0, rl=self.config.start_scale)
        zvals = [z - (self._zoffset - self.config.sequence_length // 2)
                 for z in range(self._zoffset - self.config.sequence_length // 2,
                                self._zoffset + self.config.sequence_length // 2 + 1)
                 if 0 <= z < self._dataset_zdim(dataset_name)]
        # imaris data is read in zyx axis order, z axis needs to become the final channel axis
        if len(zvals) > 0:
            self._shard_images[-1][..., zvals] = np.moveaxis(self._mimages[dataset_name].ReadData(mode='quiet'), 0, -1)
        self._shard_image_offsets.append((self._current_dataset_id, self._zoffset, self._yoffset, self._xoffset))

    def read_shard_sequential(self):
        """
        read next sequential shard. sequential position should has been set
        prior to calling this function
        """
        # read one plane further along z axis, discard one plane with smallest
        # z value
        assert not self.config.training
        self._read_shard_sequential_tiff()

    # this is a hacky api to get a single iteration of sequentially dispensed z planes. only intended for inference data
    def read_zplane(self, z):
        if self.config.training:
            raise NotImplementedError('not impelmented')
        if self._dataset_format(self._dataset_names[0]) == pymcp3d.IMARIS:
            return self._read_imaris_zplane(z)
        else:
            return self._read_tiff_zplane(z)

    def _read_tiff_zplane(self, z):
        dataset_name = self._dataset_names[0]
        if 0 <= z < self._dataset_zdim(dataset_name):
            block = pymcp3d.MImageBlock([z, 0, 0], [1, 0, 0])
            self._mimages[dataset_name].SelectView(block, 0, rl=self.config.start_scale)
            # squeeze to remove the z dimension with size 1
            return self._mimages[dataset_name].ReadData(mode='quiet').squeeze()
        else:
            return np.zeros((self._dataset_ydim(dataset_name),
                             self._dataset_xdim(dataset_name)), dtype=np.uint16)

    # assuming imaris chunk z dimension is 8
    def _read_imaris_zplane(self, z):
        dataset_name = self._dataset_names[0]
        if 0 <= z < self._dataset_zdim(dataset_name):
            # this should only happen for z = 0
            if not self._shard_images:
                assert z == 0
                block = pymcp3d.MImageBlock([0, 0, 0], [8, 0, 0])
                self._mimages[dataset_name].SelectView(block, 0, rl=self.config.start_scale)
                self._shard_images.append(self._mimages[dataset_name].ReadData(mode='quiet'))
                self._shard_image_offsets.append(0)

            assert len(self._shard_images) == len(self._shard_image_offsets) == 1
            # requested plane is not held in the shard, retrieve next shard and discard current one
            if not self._shard_image_offsets[0] <= z < self._shard_image_offsets[0] + 8:
                self._shard_image_offsets[0] += 8
                block = pymcp3d.MImageBlock([self._shard_image_offsets[0], 0, 0], [8, 0, 0])
                self._mimages[dataset_name].SelectView(block, 0, rl=self.config.start_scale)
                self._shard_images.pop()
                self._shard_images.append(self._mimages[dataset_name].ReadData(mode='quiet'))
            return self._shard_images[0][z - self._shard_image_offsets[0], :, :]
        else:
            return np.zeros((self._dataset_ydim(dataset_name),
                             self._dataset_xdim(dataset_name)), dtype=np.uint16)

    def _read_shard_sequential_tiff(self):
        self._set_next_sequential_position()
        # no images have been read for current dataset
        if not self._image_context:
            assert self._zoffset == 0
            for z in range(self._zoffset - self.config.sequence_length // 2,
                           self._zoffset + self.config.sequence_length // 2 + 1):
                self._image_context.append(self._read_tiff_zplane(z))
        # if a sequence of images have been read already, only need to discard
        # the earliest image along z axis and read one new plane
        else:
            assert len(self._image_context) == self.config.sequence_length
            self._image_context.pop(0)
            z = self._zoffset + self.config.sequence_length // 2
            self._image_context.append(self._read_tiff_zplane(z))

    def _read_shard_sequential_imaris(self):
        self._set_next_sequential_position()
        # no images have been read for current dataset
        if not self._image_context:
            assert self._zoffset == 0
            for z in range(self._zoffset - self.config.sequence_length // 2,
                           self._zoffset + self.config.sequence_length // 2 + 1):
                self._image_context.append(self._read_imaris_zplane(z))
        # if a sequence of images have been read already, only need to discard
        # the earliest image along z axis and read one new plane
        else:
            assert len(self._image_context) == self.config.sequence_length
            self._image_context.pop(0)
            z = self._zoffset + self.config.sequence_length // 2
            self._image_context.append(self._read_imaris_zplane(z))

    # draw random dataset, then draw position within dataset
    # zyx offsets must be within [0, dim_size)
    def _set_next_random_position(self):
        self._current_dataset_id = np.random.randint(0, self.n_datasets())
        dataset_name = self._dataset_names[self._current_dataset_id]
        zdim, ydim, xdim = self.dataset_dims()[dataset_name]
        self._zoffset = np.random.randint(0, zdim)
        self._yoffset = np.random.randint(0, ydim)
        self._xoffset = np.random.randint(0, xdim)

    def _set_next_sequential_position(self):
        """
        advance offsets during sequential read
        """
        dataset_name = self._dataset_names[0]
        if self._zoffset is None:
            self._zoffset, self._yoffset, self._xoffset = 0, 0, 0
        # if maximum z val reached, discard content in self._image_context and
        # cycle back to self._zoffset = 0.
        elif self._zoffset == self._dataset_zdim(dataset_name) - 1:
            self._image_context.clear()
            self._shard_images.clear()
            self._shard_image_offsets.clear()
            self._zoffset = 0
        else:
            self._zoffset += 1

    # obtain the next image context and return it, along with volume offsets
    def next_context(self, new_shard=False):
        """
        give next image context.
        for inference data (self.training = True), each image context is
        full tiff planes centered at self._zoffset
        if training data (self.training = False), each image context is a random
        element from the current shard
        zyx volume offsets
        :param new_shard: call self.read_shard() if True
        :return:
        """
        if self.config.training:
            if new_shard and not self.config.data_fit_in_memory:
                self.read_shard()
            patch_id = np.random.randint(0, len(self._shard_images))
            return self._shard_images[patch_id]
        else:
            if new_shard:
                self.read_shard()
            return self._image_context, self._zoffset, self._yoffset, self._xoffset


"""
class InstancePairedReader(ImageReaderBase):
    def __init__(self, configuration, label_rules):
        super(InstancePairedReader, self).__init__(configuration, label_rules)
        assert self.training
        assert self.has_paired_data
        # key: dataset name. val: list, sorted image names of a data set
        self._image_names = defaultdict(list)
        # key: dataset name. val: list, sorted label names of a data set
        self._label_names = defaultdict(list)
        # only include image names drawn for the shard
        self._shard_image_names = defaultdict(list)
        self._shard_label_names = defaultdict(list)
        # key: dataset name. val: image_name: label_names
        self._image_to_label_mapping = defaultdict(dict)
        # pupulate self._image_names, self._label_names,
        # self._label_to_image_names
        self._find_pair_names()
        self._n_images = sum([len(value) for value in self._image_names.values()])
        self._n_shard_images = int(math.ceil(self._n_images / self.n_shards))

    def _find_pair_names(self):
        empty_datasets = []
        for dataset_name in self._dataset_names:
            # return false if no label found for dataset
            if self._find_dataset_image_names(dataset_name):
                # match image and label index. self._image_names should only
                # have required image names
                self._find_dataset_image_to_label_mapping(dataset_name)
            else:
                empty_datasets.append(dataset_name)
        for dataset_name in empty_datasets:
            self._dataset_names.remove(dataset_name)
            self._dataset_dirs.pop(dataset_name)

    def _find_dataset_image_names(self, dataset_name):
        # find and sort label names
        image_names = os.listdir(os.path.join(self._dataset_dirs[dataset_name], 'image'))
        image_names.sort()
        for image_name in image_names:
            if image_name.find('.tif') > 0:
                self._image_names[dataset_name].append(image_name)
        # remove key dataset_name from self._image_names if dataset has no image
        if len(self._image_names[dataset_name]) == 0:
            self._image_names.pop(dataset_name)
            return False
        return True

    def _find_dataset_image_to_label_mapping(self, dataset_name):
        image_names = []
        for image_name in os.listdir(os.path.join(self._dataset_dirs[dataset_name], 'image')):
            if image_name.find('.tif') > 0:
                image_names.append(image_name)
        image_names.sort()
        self._label_names[dataset_name] = os.listdir(os.path.join(self._dataset_dirs[dataset_name], 'label'))
        label_names = set(self._label_names[dataset_name])
        for image_name in image_names:
            self._image_to_label_mapping[image_name] = [label_name for label_name in label_names
                                                        if label_name.startswith(image_name.replace('.tif', '_label'))]
            self._image_to_label_mapping[image_name].sort()
            label_names -= self._image_to_label_mapping[image_name]

    def read_shard(self):
        if self.data_fit_in_memory:
            # if data fit in memory entirely, and shard_labels is not empty,
            # data has been read already, nothing needs to be done
            if len(self._shard_images) > 0:
                return
            self._shard_image_names = self._image_names
            self._shard_label_names = self._label_names
        # draw shard labels
        else:
            self._shard_image_names.clear()
            self._shard_label_names.clear()
            self._shard_images.clear()
            self._shard_labels.clear()
            names = []
            for dataset_name, image_names in self._image_names.items():
                for image_name in image_names:
                    names.append((dataset_name, image_name))
            np.random.shuffle(names)
            for i in range(self._n_shard_images):
                choice = np.random.randint(0, high=len(names))
                choice_dataset, choice_label_name = names[choice]
                self._shard_label_names[choice_dataset].append(choice_label_name)
                names.pop(choice)
        # read shard data
        for dataset_name in self._shard_label_names.keys():
            print('reading from dataset {}...'.format(dataset_name))
            self._read_shard_dataset(dataset_name)
        # translate labels
        self.translate_labels()
        # scale if needed
        self.scale_images()
"""



