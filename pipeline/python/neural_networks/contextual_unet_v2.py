import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
from tensorflow.keras.layers import MaxPooling2D, Concatenate, Conv2DTranspose, \
    Conv2D, Softmax, BatchNormalization, LayerNormalization, Lambda, ReLU, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal
from tensorflow.keras.callbacks import TensorBoard
from .custom_metrics import MeanIoUFromProbabilities
from .custom_losses import sample_weighted_loss_function
from .custom_layers import Conv2DNormRelu, Conv2DTransposeNormRelu, Conv2DResidualBlockPartial, Conv2DResidualBlockV1
from .custom_callbacks import WritePrediction, SaveModel
from .util.model_utils import ModelUtils
from .execution_setting import TrainingSetting, InferenceSetting


class ContextualUNetV2(Model):
    """
    This implementation is a hybrid of UNet and RefineNet
    minor adjustments for neurite+soma segmentation
    output 16 bit neurite 65535 soma 32767
    """
    def __init__(self):
        super(ContextualUNetV2, self).__init__()
        self.label_rules, self.n_labels = None, None
        self.training_setting, self.inference_setting = None, None
        self.training_dataset, self.inference_dataset = None, None
        self.optimizer = None

    def __str__(self):
        return self.__class__.__name__

    def _optimizer(self):
        if self.training_setting:
            learning_rate = self.training_setting.learning_rate
            return Adam(learning_rate=learning_rate, epsilon=1e-8)
        # inference phase does not need to specify optimizer parameters
        else:
            return Adam()

    def initialize_optimizer(self):
        self.optimizer = self._optimizer()

    def build(self, input_shapes):
        assert self.n_labels is not None
        norm_class = BatchNormalization
        negative_slope = 0.001

        kernal_initializer = GlorotNormal()
        contract_filters0 = 32

        self.conv2d_res0 = Conv2DResidualBlockV1(contract_filters0, 5, n_conv2d=2, internal_normalize=False, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_res0')
        self.pool0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='pool0')

        contract_filters1 = contract_filters0 * 2
        self.conv2d_res1 = Conv2DResidualBlockV1(contract_filters1, 3, n_conv2d=2, internal_normalize=False, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_res1')
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='pool1')

        contract_filters2 = contract_filters1 * 2
        self.conv2d_res2 = Conv2DResidualBlockV1(contract_filters2, 3, n_conv2d=2, internal_normalize=False, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_res2')
        self.pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='pool2')

        contract_filters3 = contract_filters2 * 2
        self.conv2d_res3 = Conv2DResidualBlockV1(contract_filters3, 3, n_conv2d=2, internal_normalize=False, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_res3')
        self.pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='pool3')

        expand_filters3 = contract_filters3
        self.conv2dt3 = Conv2DTransposeNormRelu(expand_filters3, 3, pre_activation=True, strides=2, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2dt3')

        expand_filters2 = expand_filters3 // 2
        self.conv2dt2 = Conv2DTransposeNormRelu(expand_filters2, 3, pre_activation=True, strides=2, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2dt2')

        expand_filters1 = expand_filters2 // 2
        self.conv2dt1 = Conv2DTransposeNormRelu(expand_filters1, 3, pre_activation=True, strides=2, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2dt1')

        expand_filters0 = expand_filters1 // 2
        self.conv2dt0 = Conv2DTransposeNormRelu(expand_filters0, 3, pre_activation=True, strides=2, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2dt0')

        self.conv2d_logits = Conv2DNormRelu(self.n_labels, 1, pre_activation=True, strides=1, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_logits')
        # adding batch norm before softmax gives worse training result
        self.softmax = Softmax(axis=-1, name='softmax')

    def call(self, inputs, training=None):
        # proj = self.projection(inputs, training=training)
        # proj = self.projection_dropout(proj, training=training)
        conv0 = self.conv2d_res0(inputs, training=training)
        pool0 = self.pool0(conv0)

        conv1 = self.conv2d_res1(pool0, training=training)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2d_res2(pool1, training=training)
        pool2 = self.pool2(conv2)

        conv3 = self.conv2d_res3(pool2, training=training)
        pool3 = self.pool3(conv3)

        # conv4 = self.conv2d4(pool3, training=training)
        # conv4 = self.conv2d4_norm_relu(conv4, training=training)
        # pool4 = self.pool4(conv4)
        # pool4 = self.pool4_dropout(pool4, training=training)

        # convt4 = self.conv2dt4(pool4, training=training) + conv4
        convt3 = self.conv2dt3(pool3, training=training) + conv3
        convt2 = self.conv2dt2(convt3, training=training) + conv2
        convt1 = self.conv2dt1(convt2, training=training) + conv1
        convt0 = self.conv2dt0(convt1, training=training) + conv0

        logits = self.conv2d_logits(convt0)
        probabilities = self.softmax(logits)
        return probabilities

    # compile model and create variables
    def init_model(self, is_training=True):
        self._compile()
        self._create_varibles(is_training)
        self.summary()

    # restore previously trained model. whether the phase is training or inference
    # is determined by type of setting
    def restore_model(self):
        assert (self.training_setting is None) ^ (self.inference_setting is None)
        setting = self.training_setting if self.training_setting is not None else self.inference_setting
        is_training = isinstance(setting, TrainingSetting)
        self.init_model(is_training=is_training)
        self.load_weights(setting.trained_model_path)
        # if a clean state optimizer wanted, recompile model
        # (recompilation does not change weights)
        if setting.reset_optimizer:
            print('using newly constructed optimizer')
            self.initialize_optimizer()

    # compile model without creating variables
    def _compile(self):
        self.initialize_optimizer()
        self.compile(optimizer=self.optimizer,
                     #loss=sample_weighted_loss_function(SparseCategoricalCrossentropy, self.n_labels, static=False),
                     loss=SparseCategoricalCrossentropy(),
                     metrics=[MeanIoUFromProbabilities(self.n_labels)])

    # create variables
    def _create_varibles(self, is_training):
        if is_training:
            for batch_images, batch_labels in self.training_dataset:
                _ = self.train_on_batch(batch_images, batch_labels)
                return
        else:
            batch_zeros = np.zeros(shape=(self.inference_setting.batch_size,
                                          self.inference_setting.patch_height,
                                          self.inference_setting.patch_width,
                                          self.inference_setting.sequence_length *
                                          self.inference_setting.n_scales),
                                   dtype=np.float32)
            _ = self.predict_on_batch(batch_zeros)
            return

    def predict_from_inputs(self, inputs):
        probabilities = self.predict_on_batch(inputs)
        predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32).numpy()
        return predictions

    def train(self, training_setting, model_util):
        assert isinstance(training_setting, TrainingSetting)
        assert isinstance(model_util, ModelUtils)
        self.training_setting = training_setting
        self.inference_setting = None

        # number of classes to predict
        self.label_rules = self.training_setting.label_rules
        self.n_labels = self.label_rules.n_labels()

        # training and validation datasets
        training_batch_iter, validation_batch_iter, test_batch_iter = training_setting.get_batch_iterators()
        self.training_dataset = training_batch_iter.dataset()
        validation_dataset = None
        validation_steps = 0
        if validation_batch_iter:
            validation_dataset = validation_batch_iter.dataset()
            validation_steps = validation_batch_iter.n_batches_per_epoch()

        # initialize model if no previously trained model needs to be loaded
        # otherwise restore previously trained model
        trained_model_path = self.training_setting.trained_model_path
        if trained_model_path is None:
            self.init_model(is_training=True)
        else:
            self.restore_model()

        # callbacks
        tensorboard_callback = TensorBoard(log_dir=model_util.tensorboard_log_dir(),
                                           histogram_freq=1, write_graph=True, write_images=False)
        save_model_callback = SaveModel(model_util.model_checkpoint_filepath(), self.training_setting.n_epochs,
                                        self.training_setting.save_model_epoch_interval)
        if validation_dataset:
            write_prediction_callback = WritePrediction(model_util, validation_dataset, self.label_rules,
                                                        self.training_setting.save_output_batch_interval)
        else:
            write_prediction_callback = WritePrediction(model_util, self.training_dataset, self.label_rules,
                                                        self.training_setting.save_output_batch_interval)
        # half learning rate every 5 epoch
        # self.learning_rate_scheduler_callback = LearningRateScheduler(schedule=lambda x: 2 ** -(x // 5) * self.learning_rate, verbose=1)
        callbacks = [tensorboard_callback, save_model_callback, write_prediction_callback]

        # fit model
        self.fit(x=self.training_dataset,
                 steps_per_epoch=training_batch_iter.n_batches_per_epoch(),
                 # previous epochs will be subtracted
                 epochs=training_setting.n_epochs,
                 # useful if loading trained model and need to continue training
                 initial_epoch=training_setting.previous_epochs,
                 validation_data=validation_dataset,
                 validation_steps=validation_steps,
                 # if workers=0 is omitted when using keras.Model.fit_generator in tf version < 2.1.0,
                 # data batches appear to be all zeros for unknown reason
                 # may be related to multiprocessing based threading
                 # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit_generator
                 # the problem does not come up in tf 2.1.0 with generators
                 # wrapped by tf.data.Dataset
                 validation_freq=1, shuffle=False,
                 use_multiprocessing=False, callbacks=callbacks)
        print('end of training')

    def inference(self, inference_setting, model_util, output_prefix='', mask=None):
        assert isinstance(inference_setting, InferenceSetting)
        assert isinstance(model_util, ModelUtils)
        self.training_setting = None
        self.inference_setting = inference_setting

        # number of classes to predict
        self.label_rules = self.inference_setting.label_rules
        self.n_labels = self.label_rules.n_labels()

        self.restore_model()
        print('restored model at {}'.format(self.inference_setting.trained_model_path))

        inference_batch_iter = self.inference_setting.get_batch_iterator()
        inference_batch_iter.init_image_writer(model_util.segmentation_save_dir(), output_prefix=output_prefix)

        inference_dataset = inference_batch_iter.dataset()
        n_inference_batches_per_epoch = inference_batch_iter.n_batches_per_epoch()

        n_batches = 0
        inference_time = 0
        for batch_input, patch_positions in inference_dataset:
            # if iteration complete, break
            if n_batches == n_inference_batches_per_epoch:
                break
            if n_batches % 1000 == 0:
                print('{}/{} batches completed'.format(n_batches, n_inference_batches_per_epoch))
            start = time.time()
            batch_segmented = self.predict_from_inputs(batch_input)
            end = time.time()
            inference_time += (end - start)
            inference_batch_iter.write_batch(batch_segmented, patch_positions, mask=mask)
            n_batches += 1

        # should only flush when number of patches is divisible by batch size
        if inference_batch_iter.n_patches_per_epoch() % self.inference_setting.batch_size == 0:
            inference_batch_iter.flush_image_writer(mask=mask)
        print('{0}/{0} batches completed'.format(n_inference_batches_per_epoch))
        print('gpu inference time = {} seconds'.format(inference_time))
