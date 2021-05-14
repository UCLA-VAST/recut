from copy import deepcopy
from pprint import pformat
import math
import numpy as np


def is_powers_of_2(number):
    if not isinstance(number, int):
        return False
    if number < 1:
        return False
    if number == 1:
        return True
    while number >= 2:
        if number % 2 > 0:
            return False
        number /= 2
    return True


class ConfigurationValidator(object):
    def __init__(self, config):
        assert isinstance(config, Configuration)
        self.config = config
        self.validations = {
            'dataset_paths': [],
            'retrieve_pattern': [ConfigurationValidator._retrieval_pattern_known],
            'training': [ConfigurationValidator._is_bool],
            'high_sample_rate': [ConfigurationValidator._is_bool],
            'has_paired_data': [ConfigurationValidator._is_bool],
            'data_fit_in_memory': [ConfigurationValidator._is_bool],
            'sequence_length': [ConfigurationValidator._is_positive],
            'n_shards': [ConfigurationValidator._is_positive],
            'start_scale': [ConfigurationValidator._is_non_negative],
            'patch_height': [ConfigurationValidator._is_positive,
                             ConfigurationValidator._is_powers_of_2,
                             self._equal_height_width],
            'patch_width': [ConfigurationValidator._is_positive,
                            ConfigurationValidator._is_powers_of_2,
                            self._equal_height_width],
            'n_scales': [ConfigurationValidator._is_positive],
            'patch_x_overlap_percent': [ConfigurationValidator._valid_patch_overlap_percent],
            'patch_y_overlap_percent': [ConfigurationValidator._valid_patch_overlap_percent],
            'patch_x_overlap': [self._valid_patch_overlap],
            'patch_y_overlap': [self._valid_patch_overlap],
            'valid_patch_height': [self._valid_effective_patch_dimension],
            'valid_patch_width': [self._valid_effective_patch_dimension],
            'transformation_engine_type': [ConfigurationValidator._transformation_engine_type_known],
            'batch_size': [ConfigurationValidator._is_positive],
            'preprocess_method': [ConfigurationValidator._preprocess_method_known],
            'n_epochs': [ConfigurationValidator._is_positive],
            'learning_rate': [ConfigurationValidator._valid_learning_rate],
            'learning_rate_decay': [ConfigurationValidator._valid_learning_rate],
            'model_kwargs': [ConfigurationValidator._model_kwargs_is_dict]
        }

    def validate(self, pedantic=True):
        for setting_name, val in self.config.export().items():
            if pedantic:
                if setting_name not in self.validations:
                    raise KeyError('unknown setting {}'.format(setting_name))
            else:
                continue
            for validator in self.validations[setting_name]:
                validator(val, setting_name)

    @staticmethod
    def _retrieval_pattern_known(val, setting_name):
        assert val in {'context', 'instance', 'sequence'}, \
            'unknown {} {}'.format(setting_name, val)

    @staticmethod
    def _valid_sequence_length(val, setting_name):
        assert val > 0, '{} must be positive'.format(setting_name)

    @staticmethod
    def _is_positive(val, setting_name):
        assert val > 0, '{} must be positive'.format(setting_name)

    @staticmethod
    def _is_non_negative(val, setting_name):
        assert val >= 0, '{} must be non negative'.format(setting_name)

    @staticmethod
    def _is_powers_of_2(val, setting_name):
        assert is_powers_of_2(val), '{} must be powers of 2'.format(setting_name)

    @staticmethod
    def _supported_image_type(val, setting_name):
        assert val == np.uint16 or val == np.uint8, \
            '{}: unsupported image data type'.format(setting_name)

    def _equal_height_width(self, val, setting_name):
        assert self.config.patch_height == self.config.patch_width, \
            '{} = {}: patch must have equal height and width' \
            .format(setting_name, val)

    @staticmethod
    def _valid_patch_overlap_percent(val, setting_name):
        assert 0 <= val < 0.5, '{} = {}: patch overlap percent must ' \
                               'be in range [0, 0.5)'.format(setting_name, val)

    def _valid_patch_overlap(self, val, setting_name):
        assert 0 <= val < self.config.patch_height // 2, \
            '{} = {}: patch overlap must be [0, patch dimension // 2)'\
                .format(setting_name, val)

    def _valid_effective_patch_dimension(self, val, setting_name):
        assert 0 < val <= self.config.patch_height, \
            '{} = {}: valid patch dimension must be in range (0, patch dimension]'\
                .format(setting_name, val)

    @staticmethod
    def _is_bool(val, setting_name):
        assert isinstance(val, bool), '{} must be boolean type, got {} instead' \
            .format(setting_name, type(val))

    @staticmethod
    def _transformation_engine_type_known(val, setting_name):
        assert val in {'random', 'queue'}, \
            'unknown {} {}'.format(setting_name, val)

    @staticmethod
    def _preprocess_method_known(val, setting_name):
        assert val in {'center', 'norm', 'none'}, \
            'unknown {} {}'.format(setting_name, val)

    @staticmethod
    def _valid_learning_rate(val, setting_name):
        assert 0 <= val < 1, '{} = {}: should be in [0.0, 1.0)' \
            .format(setting_name, val)

    @staticmethod
    def _model_kwargs_is_dict(val, setting_name):
        assert isinstance(val, dict), '{} should be a dict'.format(setting_name)


class Configuration(object):
    """
    maintain validity upon creation and modification
    """
    def __init__(self):
        pass

    def __dir__(self):
        return []

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return '{}:\n{}\n'.format(self.__class__.__name__,
                                  pformat(self.__dict__))

    def edit(self, other_config=None, setting_dict=None, verbose=False):
        """
        keys unknown to instance (not in __dict__) are ignored
        """
        if other_config is not None:
            assert isinstance(other_config, Configuration)
            for key, val in other_config.export().items():
                if key in self.__dict__:
                    self.__dict__[key] = val
        if setting_dict is not None:
            assert isinstance(setting_dict, dict)
            for key, val in setting_dict.items():
                if key in self.__dict__:
                    self.__dict__[key] = val
                elif verbose:
                    print('warning: unknown config attribute {}'.format(key))
        self._validate()

    def export(self, key=None):
        if key is None:
            return deepcopy(self.__dict__)
        else:
            return self.__dict__[key]

    def next_setting(self):
        for key in self.__dir__():
            yield key, self.__dict__[key]

    def intersect(self, other_config):
        """
        values from self takes precedence
        """
        assert isinstance(other_config, Configuration)
        if type(self) is type(other_config):
            return self
        intersect_keys = self.__dict__.keys() & \
                         other_config.__dict__.keys()
        intersect_dict = {}
        for intersect_key in intersect_keys:
            intersect_dict[intersect_key] = self.__dict__[intersect_key]
        return ArbitraryConfiguration(setting_dict=intersect_dict)

    def foreign_kwargs(self, kargs_dict):
        assert isinstance(kargs_dict, dict)
        foreign = {}
        for k, v in kargs_dict.items():
            if not k in self.__dict__:
                foreign[k] = v
        return foreign

    def _validate(self, pedantic=True):
        validator = ConfigurationValidator(self)
        validator.validate(pedantic=pedantic)


def configure_instance(instance, config, allow_unknown=False):
    assert isinstance(config, Configuration)
    for setting_name, setting_value in config.export().items():
        if setting_name not in instance.__dict__:
            if not allow_unknown:
                raise ValueError('unknown setting {}'.format(setting_name))
            else:
                continue
        instance.__dict__[setting_name] = setting_value


class BatchIteratorConfiguration(Configuration):
    def __init__(self, dataset_paths, retrieve_pattern='context', training=True,
                 has_paired_data=True, high_sample_rate=True,
                 data_fit_in_memory=True, sequence_length=3, n_shards=1,
                 start_scale=0, n_scales=1, patch_height=256, patch_width=256,
                 patch_x_overlap_percent=0.25, patch_y_overlap_percent=0.25,
                 batch_size=8, preprocess_method='center'):
        super(BatchIteratorConfiguration, self).__init__()
        self.dataset_paths = dataset_paths
        self.retrieve_pattern = retrieve_pattern
        self.training = training
        self.has_paired_data = has_paired_data
        self.high_sample_rate = high_sample_rate
        self.data_fit_in_memory = data_fit_in_memory
        self.sequence_length = sequence_length
        self.n_shards = n_shards
        self.start_scale = start_scale

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.n_scales = n_scales
        self.patch_x_overlap_percent = patch_x_overlap_percent
        self.patch_y_overlap_percent = patch_y_overlap_percent
        self.patch_x_overlap = int(math.floor(self.patch_x_overlap_percent * self.patch_width))
        self.patch_y_overlap = int(math.floor(self.patch_y_overlap_percent * self.patch_height))
        self.valid_patch_height = self.patch_height - 2 * self.patch_y_overlap
        self.valid_patch_width = self.patch_width - 2 * self.patch_x_overlap
        self.batch_size = batch_size
        self.preprocess_method = preprocess_method
        self._validate()

    def __dir__(self):
        return BatchIteratorConfiguration.settings()

    @classmethod
    def settings(cls):
        return ['dataset_paths', 'retrieve_pattern', 'training',
                'has_paired_data', 'high_sample_rate', 'data_fit_in_memory',
                'sequence_length', 'n_shards', 'start_scale',
                'patch_height', 'patch_width', 'n_scales',
                'patch_x_overlap_percent', 'patch_y_overlap_percent',
                'batch_size', 'preprocess_method']


class NetConfiguration(Configuration):
    def __init__(self, n_epochs=80, learning_rate=1e-3, learning_rate_decay=0.9,
                 save_output_batch_interval=400, save_model_epoch_interval=5, model_kwargs=None):
        super(NetConfiguration, self).__init__()
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.save_output_batch_interval = save_output_batch_interval
        self.save_model_epoch_interval = save_model_epoch_interval
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

    def __dir__(self):
        return NetConfiguration.settings()

    @classmethod
    def settings(cls):
        return ['learning_rate', 'learning_rate_decay', 'model_kwargs']


class ArbitraryConfiguration(Configuration):
    def __init__(self, setting_dict={}, **settings):
        super(ArbitraryConfiguration, self).__init__()
        setting_dict.update(settings)
        self.__dict__ = deepcopy(setting_dict)
        self._validate(pedantic=False)

    def __dir__(self):
        return self.__dict__.keys()

