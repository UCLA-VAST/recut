from abc import ABCMeta, abstractmethod
from copy import deepcopy
from .util.model_utils import ModelUtils
from .util.configurations import BatchIteratorConfiguration, NetConfiguration
from .data_io.label_rules import LabelRules
from .data_io.image_transformation import identity_engine, TransformationRandom
from .data_io.batch_iterator import ContextPairedBatchIterator, ContextUnpairedBatchIterator


class ExecutionSetting:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @staticmethod
    def default_setting():
        setting = {
            'model_root_dir': None,
            'model_name': None,
            'dataset_directories': {
                'training': None,
                'validation': None,
                'test': None,
                'inference': None
            },
            # only relevant during inference time as roi selection
            'dataset_selections': {
                'offsets': [],
                'extents': [],
            },
            'trained_model': {
                'model_path': None,
                'previous_epochs': 0,
                'reset_optimizer': False
            },
            'label_rules': {
                'label_input_colors': {
                    'nissl': (255, 0, 0), 'background': (0, 0, 0)
                },
                'label_weights': {
                    'soma': 10, 'background': 1
                }
            },
            'batch_iterator': {
                'retrieve_pattern': 'context',
                'has_paired_data': True,
                'training': True,
                'high_sample_rate': False,
                'data_fit_in_memory': False,
                'sequence_length': 3,
                'n_shards': 1,
                'start_scale': 0,
                'patch_height': 256,
                'patch_width': 256,
                'n_scales': 1,
                'patch_x_overlap_percent': 0.25,
                'patch_y_overlap_percent': 0.25,
                'batch_size': 8,
                'preprocess_method': 'none'
            },
            'transformation': {
                'params': {
                    'rotation': (8, (0, 90, 180, 270, 'random', 'random',
                                     'random', 'random')),
                    'gaussian': (1, (0, 3)),
                    'reverse': (1, ())
                },
                'nesting_orders': [('gaussian', 'rotation'),
                                   ('gaussian', 'rotation', 'reverse')],
                'include_identity': False
            },
            'net': {
                'n_epochs': 80,
                'learning_rate': 1e-3,
                'learning_rate_decay': 0.01,
                'save_output_batch_interval': 400,
                'save_model_epoch_interval': 5,
                'model_kwargs': {}
            }
        }
        return setting

    @abstractmethod
    def _setting(self):
        """
        :return: inheriting should return its own setting dict
        """
        pass

    @abstractmethod
    def _batch_config(self):
        """
        :return: inheriting should return its batch iterator configuration.
                 when more than one of training/validation/test iterator is
                 available, return training iterator
        """
        pass

    def get_net_config(self):
        pass

    @property
    def model_name(self):
        return self._setting()['model_name']

    @property
    def trained_model_path(self):
        return self._setting()['trained_model']['model_path']

    @property
    def training(self):
        return self._batch_config().training

    @property
    def batch_size(self):
        return self._batch_config().batch_size

    @property
    def patch_height(self):
        return self._batch_config().patch_height

    @property
    def patch_width(self):
        return self._batch_config().patch_width

    @property
    def sequence_length(self):
        return self._batch_config().sequence_length

    @property
    def n_scales(self):
        return self._batch_config().n_scales

    @property
    def retrieve_pattern(self):
        return self._batch_config().retrieve_pattern

    @property
    def n_epochs(self):
        return self.get_net_config().n_epochs

    @property
    def learning_rate(self):
        return self.get_net_config().learning_rate

    @property
    def previous_epochs(self):
        return self._setting()['trained_model']['previous_epochs']

    @property
    def reset_optimizer(self):
        return self._setting()['trained_model']['reset_optimizer']

    @property
    def model_kwargs(self):
        return self.get_net_config().model_kwargs

    @property
    def save_output_batch_interval(self):
        return self.get_net_config().save_output_batch_interval

    @property
    def save_model_epoch_interval(self):
        return self.get_net_config().save_model_epoch_interval

    @property
    def label_rules(self):
        return LabelRules(self._setting()['label_rules']['label_input_colors'],
                          label_weights=self._setting()['label_rules']['label_weights'])


class TrainingSetting(ExecutionSetting):
    def __init__(self, setting_dict=None):
        """
        when setting_dict is None, supplie default setting. the default setting
        is useful for code testing. setting_dict should follow the layout of
        default setting
        """
        super(TrainingSetting, self).__init__()
        if setting_dict is None:
            self._training_setting = ExecutionSetting.default_setting()
        else:
            self._training_setting = deepcopy(setting_dict)
        self._validate_setting()
        trained_model_path = self._training_setting['trained_model']['model_path']
        if trained_model_path is not None:
            previous_training_setting = ModelUtils.load_previous_training_setting(trained_model_path)
            skip_keys = {'n_shards', 'data_fit_in_memory'}
            # make sure model input output dimensions will be compatible
            for key in self._training_setting['batch_iterator']:
                if key not in skip_keys:
                    assert self._training_setting['batch_iterator'][key] == previous_training_setting['batch_iterator'][key]
        for label_name, label_color in self._training_setting['label_rules']['label_input_colors'].items():
            self._training_setting['label_rules']['label_input_colors'][label_name] = tuple(label_color)

    def _validate_setting(self):
        """
        assert self._training_setting has the same structure as default setting
        """
        # same keys in self._training_setting and default settings
        default_setting = ExecutionSetting.default_setting()
        assert self._training_setting.keys() == default_setting.keys()
        # for dict values, assert same key set
        for k, v in self._training_setting.items():
            if isinstance(v, dict):
                assert v.keys() == default_setting[k].keys()

    def _setting(self):
        return self._training_setting

    def _batch_config(self):
        return self.get_batch_iterator_config(data_type='training')

    def training_setting(self):
        return self._training_setting

    def set_batch_iteator(self, **kwargs):
        """
        set batch iterator key value pairs
        """

    def get_net_config(self):
        setting = self._training_setting['net']
        return NetConfiguration(n_epochs=setting['n_epochs'],
                                learning_rate=setting['learning_rate'],
                                learning_rate_decay=setting['learning_rate_decay'],
                                save_output_batch_interval=setting['save_output_batch_interval'],
                                save_model_epoch_interval=setting['save_model_epoch_interval'],
                                model_kwargs=setting['model_kwargs'])

    def get_batch_iterator_config(self, data_type='training'):
        config = self._training_setting['batch_iterator']
        return BatchIteratorConfiguration(self._training_setting['dataset_directories'][data_type],
                                          retrieve_pattern=config['retrieve_pattern'],
                                          training=config['training'], has_paired_data=config['has_paired_data'],
                                          high_sample_rate=config['high_sample_rate'],
                                          data_fit_in_memory=config['data_fit_in_memory'],
                                          sequence_length=config['sequence_length'], n_shards=config['n_shards'],
                                          start_scale=config['start_scale'], n_scales=config['n_scales'],
                                          patch_height=config['patch_height'], patch_width=config['patch_width'],
                                          patch_x_overlap_percent=config['patch_x_overlap_percent'],
                                          patch_y_overlap_percent=config['patch_y_overlap_percent'],
                                          batch_size=config['batch_size'], preprocess_method=config['preprocess_method'])

    # data_type: one of training, validation, test, all belonging to model
    # training phase rather than inference phase
    def get_batch_iterator(self, label_rules, engine, data_type='training'):
        # if dataset_directories is empty, return None
        if not self._training_setting['dataset_directories'][data_type]:
            return None
        batch_iterator_config = self.get_batch_iterator_config(data_type=data_type)
        _engine = engine if data_type == 'training' else identity_engine
        # validation and test data should come from unaltered data generation
        # distribution
        if not data_type == 'training':
            batch_iterator_config.high_sample_rate = False
        if batch_iterator_config.retrieve_pattern == 'context':
            if batch_iterator_config.has_paired_data:
                return ContextPairedBatchIterator(batch_iterator_config, label_rules=label_rules, transformation_engine=_engine)
            else:
                return ContextUnpairedBatchIterator(batch_iterator_config)
        elif batch_iterator_config.retrieve_pattern == 'instance':
            pass
        elif batch_iterator_config.retrieve_pattern == 'sequence':
            pass
        else:
            raise ValueError('unsupported data retrieve pattern')

    def get_batch_iterators(self):
        batch_iterators = []
        label_rules = LabelRules(self._training_setting['label_rules']['label_input_colors'],
                                 label_weights=self._training_setting['label_rules']['label_weights'])
        engine = TransformationRandom(transformation_params=self._training_setting['transformation']['params'],
                                      transformation_nesting_orders=self._training_setting['transformation']['nesting_orders'],
                                      include_identity=self._training_setting['transformation']['include_identity'])

        data_types = ['training', 'validation', 'test']
        for data_type in data_types:
            batch_iterators.append(self.get_batch_iterator(label_rules, engine, data_type=data_type))
        return batch_iterators

    def get_model_util(self):
        return ModelUtils(self._training_setting['model_root_dir'],
                          model_name=self._training_setting['model_name'],
                          trained_model_path=self._training_setting['trained_model']['model_path'],
                          training=True)


class InferenceSetting(ExecutionSetting):
    def __init__(self, model_root_dir, inference_dataset_dir, trained_model_path):
        """
        obtain inference settings by recover serialized training setting
        from previous training session
        :param model_root_dir:
        :param inference_dataset_dir:
        :param trained_model_path:
        """
        super(InferenceSetting, self).__init__()
        self._init_from_previous_model_path(model_root_dir, inference_dataset_dir, trained_model_path)

    def _init_from_previous_model_path(self, model_root_dir, inference_dataset_dir, trained_model_path):
        self._inference_setting = ModelUtils.load_previous_training_setting(trained_model_path)
        for label_name, label_color in self._inference_setting['label_rules']['label_input_colors'].items():
            self._inference_setting['label_rules']['label_input_colors'][label_name] = tuple(label_color)
        self._inference_setting['trained_model']['model_path'] = trained_model_path
        self._inference_setting['trained_model']['previous_epochs'] = self._inference_setting['net']['n_epochs']
        self._inference_setting['model_root_dir'] = model_root_dir
        self._inference_setting['dataset_directories']['training'] = None
        self._inference_setting['dataset_directories']['validation'] = None
        self._inference_setting['dataset_directories']['testing'] = None
        self._inference_setting['dataset_directories']['inference'] = [inference_dataset_dir]
        self._inference_setting['batch_iterator']['training'] = False
        self._inference_setting['batch_iterator']['has_paired_data'] = False
        self._inference_setting['batch_iterator']['high_sample_rate'] = False
        self._inference_setting['transformation']['params'] = None
        self._inference_setting['transformation']['nesting_orders'] = None
        self._inference_setting['transformation']['include_identity'] = True

    def _setting(self):
        return self._inference_setting

    def _batch_config(self):
        return self.get_batch_iterator_config()

    def inference_setting(self):
        return self._inference_setting

    def get_net_config(self):
        setting = self._inference_setting['net']
        # ignoring keys other than 'model_kwargs': they are not needed for
        # inference and may change
        return NetConfiguration(model_kwargs=setting['model_kwargs'])

    def get_batch_iterator_config(self):
        config = self._inference_setting['batch_iterator']
        return BatchIteratorConfiguration(self._inference_setting['dataset_directories']['inference'] ,
                                          retrieve_pattern=config['retrieve_pattern'], training=config['training'],
                                          has_paired_data=config['has_paired_data'], high_sample_rate=config['high_sample_rate'],
                                          data_fit_in_memory=config['data_fit_in_memory'], sequence_length=config['sequence_length'],
                                          n_shards=config['n_shards'], start_scale=config['start_scale'],
                                          n_scales=config['n_scales'], patch_height=config['patch_height'],
                                          patch_width=config['patch_width'], patch_x_overlap_percent=config['patch_x_overlap_percent'],
                                          patch_y_overlap_percent=config['patch_y_overlap_percent'], batch_size=config['batch_size'],
                                          preprocess_method=config['preprocess_method'])

    def get_batch_iterator(self):
        batch_iterator_config = self.get_batch_iterator_config()
        label_rules = LabelRules(self._inference_setting['label_rules']['label_input_colors'],
                                 label_weights=self._inference_setting['label_rules']['label_weights'])
        if not batch_iterator_config.dataset_paths:
            return None
        if batch_iterator_config.retrieve_pattern == 'context':
            return ContextUnpairedBatchIterator(batch_iterator_config, label_rules=label_rules)
        elif batch_iterator_config.retrieve_pattern == 'instance':
            pass
        elif batch_iterator_config.retrieve_pattern == 'sequence':
            pass
        else:
            raise ValueError('unsupported data retrieve pattern')

    def get_model_util(self, deploy_pipeline):
        return ModelUtils(self._inference_setting['model_root_dir'],
                          model_name=self._inference_setting['model_name'],
                          trained_model_path=self._inference_setting['trained_model']['model_path'],
                          training=False, deploy=deploy_pipeline)



