import os
from shutil import copy, copytree, rmtree
import json
import re
import numpy as np
import tensorflow as tf
import cv2
from mcp3d_path import Mcp3dPath
from pipeline_version import time_stamp


class ModelUtils:
    MODEL_FILE_NAMES = {'ContextualUNet': 'contextual_unet.py',
                        'ContextualUNetV1': 'contextual_unet_v1.py',
                        'ContextualUNetV2': 'contextual_unet_v2.py',
                        'DCGan': 'dcgan.py',
                        'NisslNet': 'nissl_net.py',
                        'EncoderDecoder': 'encoder_decoder.py'}
    AUXILIARY_FILE_NAMES = ['net.py', 'train_net.py', '../train_model.py',
                            'deploy_net.py', '../workflow/deploy_model.py',
                            'custom_layers.py', 'custom_losses.py',
                            'custom_metrics.py', 'custom_callbacks.py',
                            'model_factory.py', 'util/model_utils.py',
                            'util/configurations.py', 'execution_setting.py',
                            'data_io/image_transformation/fundamental_transformations.py',
                            'data_io/image_transformation/transformation_factory.py',
                            'data_io/image_transformation/transformation_engine.py',
                            'data_io/batch_iterator.py', 'data_io/image_reader.py',
                            'data_io/image_writer.py', 'data_io/label_rules.py'
                            ]

    # will create directory and save outputs at
    # model_root_dir/model_name_timestamp
    def __init__(self, model_root_dir, model_name='ContextualUnetV1', trained_model_path=None, training=True, deploy=False):
        self._model_root_dir = model_root_dir
        self.trained_model_path = trained_model_path
        self._training = training
        self._deploy = deploy
        # if instantiated in deployed pipeline, the usage is always inference
        if self._deploy:
            self._training = False
            assert self.trained_model_path

        if self.trained_model_path is None:
            self._model_name = model_name
            self._time_stamp = time_stamp("%y%m%d_%H%M")
            self._trained_model_src_dir = None
        # if restoring model for inference, use the model name and time stamp
        # of the restored model. if restoring for training, use model name of
        # the restored model, but a new time stamp
        else:
            if not self._training:
                self._model_name, self._time_stamp = ModelUtils.trained_instance_name_and_timestamp(trained_model_path)
            else:
                self._model_name = ModelUtils.trained_instance_name_and_timestamp(trained_model_path, return_timestamp=False)
                self._time_stamp = time_stamp("%y%m%d_%H%M")
            # deployed pipeline has no access to "scripts" directory of trained model
            if not self._deploy:
                self._trained_model_src_dir = os.path.join(os.path.dirname(os.path.dirname(trained_model_path)), 'scripts')
                if not os.path.isdir(self._trained_model_src_dir):
                    raise ValueError('can not find script dir {} for restored model'.format(self._trained_model_src_dir))

        if not self._deploy:
            # if not starting from trained model, only save scripts from current
            # source directory content. otherwise additionally copy scripts
            # generating the restored model
            self._src_dir = os.path.join(Mcp3dPath().python_dir, 'neural_networks')
            self._script_paths = self._get_script_paths()
            self.save_scripts()

    @property
    def data_root_dir(self):
        return self._model_root_dir

    @property
    def model_name(self):
        return self._model_name

    def save_scripts(self):
        for sciprt_path in self._script_paths:
            copy(sciprt_path, self._script_save_dir())
        if self._trained_model_src_dir:
            rmtree(os.path.join(self._script_save_dir(), 'restored_model_scripts'),
                   ignore_errors=True)
            copytree(self._trained_model_src_dir,
                     os.path.join(self._script_save_dir(), 'restored_model_scripts'))

    def save_training_setting(self, training_config):
        json_path = os.path.join(self._instance_save_dir(), 'training_setting.json')
        with open(json_path, 'w') as f:
            json.dump(training_config, f, indent=4)

    def save_inference_setting(self, inference_config):
        json_path = os.path.join(self._instance_save_dir(), 'inference_setting.json')
        with open(json_path, 'w') as f:
            json.dump(inference_config, f, indent=4)

    # model full path is instance_save_dir/model_checkpoint/saved_model.{epoch:03d}-{val_loss:.6f}
    @staticmethod
    def load_previous_training_setting(trained_model_path):
        setting_json_path = ModelUtils.previous_training_setting_path(trained_model_path)
        assert os.path.isfile(setting_json_path)
        with open(setting_json_path, 'r') as f:
            setting = json.load(f)
        return setting

    @staticmethod
    def trained_instance_name(trained_model_path):
        trained_instance_dir = ModelUtils._trained_instance_save_dir(trained_model_path)
        instance_pattern = re.compile('(.+)_([0-9]{6}_[0-9]{4})')
        m = re.match(instance_pattern, os.path.basename(trained_instance_dir))
        assert m
        return m.group()

    @staticmethod
    def trained_instance_name_and_timestamp(trained_model_path, return_timestamp=True):
        trained_instance_name = ModelUtils.trained_instance_name(trained_model_path)
        instance_pattern = re.compile('(.+)_([0-9]{6}_[0-9]{4})')
        m = re.match(instance_pattern, trained_instance_name)
        assert m
        if return_timestamp:
            return m.group(1), m.group(2)
        return m.group(1)

    def tensorboard_log_dir(self):
        tensorboard_log_dir = os.path.join(self._instance_save_dir(),
                                           'tensorboard_log', self._time_stamp)
        if not os.path.isdir(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)
        return tensorboard_log_dir

    def segmentation_save_dir(self):
        segmentation_dir = os.path.join(self._instance_save_dir(), 'segmentation')
        if not os.path.isdir(segmentation_dir):
            os.makedirs(segmentation_dir)
        return segmentation_dir

    @staticmethod
    def previous_training_setting_path(trained_model_path):
        return os.path.join(ModelUtils._trained_instance_save_dir(trained_model_path),
                            'training_setting.json')

    def model_checkpoint_filepath(self):
        return os.path.join(self._model_checkpoint_dir(), 'saved_model.{epoch:03d}-{val_loss:.6f}')

    # batch_images as provided by batch iterator needs to be passed.
    # gradient flow to network input, at end of train_op the network's input
    # is modified and doesn't match that of batch iterator output
    def save_outputs(self, batch_images, batch_outputs, epoch_id, batch_id,
                     batch_labels=None, label_rules=None, save_image_as_8bit=True):
        epoch_str = str(epoch_id).zfill(3)
        batch_str = str(batch_id).zfill(7)
        if batch_labels is not None:
            batch_labels = np.squeeze(batch_labels)
        for i in range(batch_images.shape[0]):
            center_index = batch_images.shape[-1] // 2
            batch_image = batch_images[i, :, :, center_index]
            batch_output = batch_outputs[i, :, :]
            assert batch_image.shape == batch_output.shape
            i_str = str(i).zfill(3)
            image_name = 'epoch_{}_iter_{}_{}_image.tif'.format(epoch_str, batch_str, i_str)
            image = ModelUtils._convert_img(batch_image, as_8bit=save_image_as_8bit)
            cv2.imwrite(os.path.join(self._outputs_save_dir(epoch_id), image_name), image)
            output_name = 'epoch_{}_iter_{}_{}_output.tif'.format(epoch_str, batch_str, i_str)
            if label_rules is not None:
                batch_output = label_rules.translate_to_input(batch_output)
            else:
                batch_output = ModelUtils._convert_img(batch_output, as_8bit=save_image_as_8bit)
            cv2.imwrite(os.path.join(self._outputs_save_dir(epoch_id), output_name), batch_output)
            if batch_labels is not None:
                assert label_rules is not None
                batch_label = batch_labels[i, :, :]
                assert batch_image.shape == batch_label.shape
                label_name = 'epoch_{}_iter_{}_{}_label.tif'.format(epoch_str, batch_str, i_str)
                label = label_rules.translate_to_input(batch_label)
                cv2.imwrite(os.path.join(self._outputs_save_dir(epoch_id), label_name), label)

    @staticmethod
    def _convert_img(image, as_8bit=True):
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        if image.dtype == np.float32 or image.dtype == np.float64:
            min_val = np.amin(image)
            if min_val < 0:
                image += np.abs(min_val)
        else:
            image = image.astype(np.float32)
        if as_8bit:
            image *= 255 / np.amax(image)
            image = image.astype(np.uint8)
        else:
            image *= 65535 / np.amax(image)
            image = image.astype(np.uint16)
        return image

    def _get_script_paths(self):
        if self._model_name not in ModelUtils.MODEL_FILE_NAMES:
            raise KeyError('can not find script for {} instance'.format(self._model_name))
        script_paths = []
        model_script_path = os.path.normpath(os.path.join(self._src_dir,
                                                          ModelUtils.MODEL_FILE_NAMES[self._model_name]))
        script_paths.append(model_script_path)
        for auxiliary_file_name in ModelUtils.AUXILIARY_FILE_NAMES:
            script_paths.append(os.path.join(self._src_dir, auxiliary_file_name))
        for script_path in script_paths:
            if not os.path.isfile(script_path):
                raise ValueError('{}: script does not exist'.format(script_path))
        return script_paths

    def _root_save_dir(self):
        if not os.path.isdir(self._model_root_dir):
            os.makedirs(self._model_root_dir)
        assert os.path.isdir(self._model_root_dir)
        return self._model_root_dir

    # if self._training = False, do not nest model name and time stamp under root
    # save dir. use root save dir as instance save dir
    def _instance_save_dir(self):
        if self._training:
            model_save_dir = os.path.join(self._root_save_dir(), '{}_{}'.format(self._model_name, self._time_stamp))
        else:
            model_save_dir = self._root_save_dir()
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
        return model_save_dir

    @staticmethod
    def _trained_instance_save_dir(trained_model_path):
        d = os.path.normpath(os.path.join(trained_model_path, '../..'))
        assert os.path.isdir(d), 'previous model instance save directory {} not found'.format(d)
        return d

    def _script_save_dir(self):
        script_save_dir = os.path.join(self._instance_save_dir(), 'scripts')
        if not os.path.isdir(script_save_dir):
            os.makedirs(script_save_dir)
        return script_save_dir

    def _outputs_save_dir(self, epoch):
        epoch_str = str(epoch).zfill(3)
        predictions_save_dir = os.path.join(self._instance_save_dir(), 'outputs',
                                            'epoch_{}'.format(epoch_str))
        if not os.path.isdir(predictions_save_dir):
            os.makedirs(predictions_save_dir)
        return predictions_save_dir

    def _model_checkpoint_dir(self):
        model_checkpoint_dir = os.path.join(self._instance_save_dir(), 'model_checkpoint')
        if not os.path.isdir(model_checkpoint_dir):
            os.makedirs(model_checkpoint_dir)
        return model_checkpoint_dir

    def _model_weights_save_path(self, iteration):
        return os.path.join(self._model_checkpoint_dir(),
                            '{}_{}_iter{}'.format(self._model_name, self._time_stamp, iteration))

    def _segmentation_patch_save_dir(self):
        patch_save_dir = os.path.join(self._instance_save_dir(), 'segmentation_patches')
        if not os.path.isdir(patch_save_dir):
            os.makedirs(patch_save_dir)
        return patch_save_dir

