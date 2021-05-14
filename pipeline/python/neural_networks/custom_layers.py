import tensorflow as tf
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, \
    LayerNormalization, ReLU, Lambda, Conv2DTranspose, MaxPooling2D, concatenate, add
from tensorflow_addons.layers import InstanceNormalization


def get_norm_name(norm_class):
    if norm_class == BatchNormalization:
        return 'batch_norm'
    elif norm_class == InstanceNormalization:
        return 'instance_norm'
    else:
        return 'layer_norm'


class NormRelu(Layer):
    def __init__(self, norm_class=BatchNormalization, negative_slope=0.01, dtype=tf.float32, name=None):
        super(NormRelu, self).__init__(trainable=True, dtype=dtype, name=name)
        self.norm_class = norm_class
        self.norm_name = get_norm_name(self.norm_class)
        self.negative_slope = negative_slope

    def build(self, input_shape):
        self.norm = self.norm_class(axis=-1, name=self.norm_name)
        self.relu = ReLU(negative_slope=self.negative_slope, name='relu')

    def call(self, inputs, training=None):
        output = self.norm(inputs, training=training)
        return self.relu(output)


class Conv2DNormRelu(Layer):
    def __init__(self, n_filters, kernel_size, strides=1,
                 norm_class=BatchNormalization, negative_slope=0.01,
                 has_activation=True, pre_activation=False,
                 kernal_initializer=GlorotNormal(), dtype=tf.float32, name=None):
        super(Conv2DNormRelu, self).__init__(trainable=True, dtype=dtype, name=name)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.kernal_initializer = kernal_initializer
        self.strides = strides
        self.norm_class = norm_class
        self.norm_name = get_norm_name(self.norm_class)
        self.negative_slope = negative_slope
        self.has_activation = has_activation
        self.pre_activation = pre_activation

    def build(self, input_shape):
        self.conv = Conv2D(self.n_filters, self.kernel_size, strides=self.strides, use_bias=True,
                           dtype=self.dtype, padding='same', data_format='channels_last', name='conv2d',
                           kernel_initializer=self.kernal_initializer,
                           bias_initializer=tf.initializers.Constant(0))
        self.norm = self.norm_class(axis=-1, trainable=True, center=True, scale=True, name=self.norm_name)
        if self.has_activation:
            self.relu = ReLU(negative_slope=self.negative_slope, name='relu')
        else:
            self.relu = Lambda(lambda x: x, name='identity')

    def call(self, inputs, training=None):
        if self.pre_activation:
            output = self.norm(inputs, training=training)
            output = self.relu(output)
            return self.conv(output)
        else:
            output = self.conv(inputs)
            output = self.norm(output, training=training)
            return self.relu(output)


class Conv2DTransposeNormRelu(Layer):
    def __init__(self, n_filters, kernel_size, strides=1,
                 norm_class=BatchNormalization, negative_slope=0.01,
                 has_activation=True, pre_activation=False,
                 kernal_initializer=GlorotNormal(), dtype=tf.float32, name=None):
        super(Conv2DTransposeNormRelu, self).__init__(trainable=True, dtype=dtype, name=name)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.kernal_initializer = kernal_initializer
        self.strides = strides
        self.norm_class = norm_class
        self.norm_name = get_norm_name(self.norm_class)
        self.negative_slope = negative_slope
        self.has_activation = has_activation
        self.pre_activation = pre_activation

    def build(self, input_shape):
        self.convt = Conv2DTranspose(self.n_filters, self.kernel_size, strides=self.strides, use_bias=True,
                                     dtype=self.dtype, padding='same', data_format='channels_last', name='conv2dt',
                                     kernel_initializer=self.kernal_initializer, bias_initializer=tf.initializers.Constant(0))
        self.norm = self.norm_class(axis=-1, trainable=True, center=True, scale=True, name=self.norm_name)
        if self.has_activation:
            self.relu = ReLU(negative_slope=self.negative_slope, name='relu')
        else:
            self.relu = Lambda(lambda x: x, name='identity')

    def call(self, inputs, training=None):
        if self.pre_activation:
            output = self.norm(inputs, training=training)
            output = self.relu(output)
            return self.convt(output)
        else:
            output = self.convt(inputs)
            output = self.norm(output, training=training)
            return self.relu(output)


# Note: includes no BN or activation
class Conv2DResidualBlockPartial(Layer):
    """
    input -----------> (1x1 convolution, n_filters) --- sum -->
          |                                              |
          --> Conv2D --> ... --> Conv2D ------------------
    stride of convolution is 1 in all layers
    last conv2d layer will be summed to potentially projected input without
    batch normalize or activation
    filters in all conv2d layers have identical dimensions (kernel_size, kernel_size, n_filters, n_filters)
    """
    def __init__(self, n_filters, kernel_size, n_conv2d=1, negative_slope=0.01,
                 kernal_initializer=GlorotNormal(), dtype=tf.float32, name=None):
        super(Conv2DResidualBlockPartial, self).__init__(trainable=True, dtype=dtype, name=name)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.kernal_initializer = kernal_initializer
        self.n_conv2d = n_conv2d
        if self.n_conv2d < 1:
            raise ValueError('at least one conv2d layer required')
        self.negative_slope = negative_slope

    def build(self, input_shape):
        # Conv2DBNRelu layers
        for i in range(self.n_conv2d):
            conv2d_name = 'conv_{0}x{0}s1_{1}'.format(self.kernel_size, i)
            setattr(self, conv2d_name,
                    Conv2D(self.n_filters, self.kernel_size, strides=1, use_bias=True,
                           dtype=self.dtype, padding='same', data_format='channels_last',
                           kernel_initializer=self.kernal_initializer,
                           bias_initializer=tf.initializers.Constant(0),
                           name=conv2d_name))
        # if input and output dimensions are consistent, identity mapping
        if input_shape[-1] == self.n_filters:
            self.projection = Lambda(lambda x: x)
        # 1x1 convolution to adjust dimension
        else:
            self.projection = Conv2D(self.n_filters, 1, strides=1, use_bias=True, dtype=self.dtype,
                                     padding='same', data_format='channels_last',
                                     kernel_initializer=self.kernal_initializer,
                                     bias_initializer=tf.initializers.Constant(0),
                                     name='projection')

    def call(self, inputs, training=None):
        result = None
        for i in range(self.n_conv2d):
            conv2d_name = 'conv_{0}x{0}s1_{1}'.format(self.kernel_size, i)
            conv2d_layer = getattr(self, conv2d_name)
            if i == 0:
                result = conv2d_layer(inputs, training=training)
            else:
                result = conv2d_layer(result, training=training)
        result += self.projection(inputs)
        return result


class Conv2DResidualBlockV1(Layer):
    """
    Kaiming He, Xiangyu Zhang
    Identity Mappings in Deep Residual Networks
    """
    def __init__(self, n_filters, kernel_size, n_conv2d=1, negative_slope=0.01,
                 norm_class=BatchNormalization, internal_normalize=False, has_activation=True,
                 kernal_initializer=GlorotNormal(), dtype=tf.float32, name=None):
        super(Conv2DResidualBlockV1, self).__init__(trainable=True, dtype=dtype, name=name)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.kernal_initializer = kernal_initializer
        self.n_conv2d = n_conv2d
        if self.n_conv2d < 1:
            raise ValueError('at least one conv2d layer required')
        self.negative_slope = negative_slope
        self.norm_class = norm_class
        self.norm_name = get_norm_name(self.norm_class)
        self.internal_normalize = internal_normalize
        self.has_activation = has_activation

    def build(self, input_shape):
        # bn and relu layers for input
        self.input_norm = self.norm_class(axis=-1, trainable=True, center=True, scale=True, name='input_norm')
        self.input_relu = ReLU(negative_slope=self.negative_slope, name='input_relu')
        for i in range(self.n_conv2d - 1):
            conv2d_name = 'conv2d_{0}'.format(i)
            if self.internal_normalize:
                setattr(self, conv2d_name,
                        Conv2DNormRelu(self.n_filters, self.kernel_size, strides=1,
                                       norm_class=self.norm_class, negative_slope=self.negative_slope,
                                       has_activation=True, kernal_initializer=self.kernal_initializer,
                                       dtype=self.dtype, name=conv2d_name))
            else:
                activation_func = lambda x: relu(x, alpha=self.negative_slope)
                setattr(self, conv2d_name,
                        Conv2D(self.n_filters, self.kernel_size, strides=1, use_bias=True, activation=activation_func,
                               dtype=self.dtype, padding='same', data_format='channels_last', name=conv2d_name,
                               kernel_initializer=self.kernal_initializer, bias_initializer=tf.initializers.Constant(0)))
        conv2d_name = 'conv2d_{0}'.format(self.n_conv2d - 1)
        setattr(self, conv2d_name,
                Conv2D(self.n_filters, self.kernel_size, strides=1, use_bias=True, dtype=self.dtype,
                       padding='same', data_format='channels_last', name=conv2d_name,
                       kernel_initializer=self.kernal_initializer, bias_initializer=tf.initializers.Constant(0)))
        # if input and output dimensions are consistent, identity mapping
        if input_shape[-1] == self.n_filters:
            self.projection = Lambda(lambda x: x, name='identity')
        # 1x1 convolution to adjust dimension
        else:
            self.projection = Conv2D(self.n_filters, 1, strides=1, use_bias=True, dtype=self.dtype,
                                     padding='same', data_format='channels_last', name='projection',
                                     kernel_initializer=self.kernal_initializer, bias_initializer=tf.initializers.Constant(0))

    def call(self, inputs, training=None):
        inputs_norm = self.input_norm(inputs, training=training)
        inputs_activation = self.input_relu(inputs_norm, training=training)
        for i in range(self.n_conv2d):
            conv2d_name = 'conv2d_{0}'.format(i)
            conv2d_layer = getattr(self, conv2d_name)
            inputs_activation = conv2d_layer(inputs_activation)
        inputs_projection = self.projection(inputs)
        return add([inputs_projection, inputs_activation])


class ResidualNext(Layer):
    def __init__(self, cardinality, dtype=tf.float32, name=None):
        pass


class Inception(Layer):
    # implements Going Deeper with Convolutions CVPR 2015:
    # Inception module with dimensionality reduction
    # for each filter size k, if reduced dimensions is not None,
    # 1 x 1 x reduced_dimension convolution is followed by
    # k x k x filter_numbers convolution
    def __init__(self, n_filters, kernel_sizes,
                 strides=1, reduced_dimensions=None,
                 projection=False, projection_dimension=None,
                 max_pool=False, pooling_projection_dimension=None,
                 relu_negative_slope=0.01, has_activation=True, name=None):
        super(Inception, self).__init__(trainable=True, name=name)
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        assert len(self.n_filters) == len(self.kernel_sizes)
        self.strides = strides
        if reduced_dimensions is None:
            self.reduced_dimensions = [None] * len(self.n_filters)
        else:
            assert len(self.n_filters) == len(reduced_dimensions)
            self.reduced_dimensions = reduced_dimensions
        self.has_projection = projection
        self.projection_dimension = projection_dimension
        if self.has_projection:
            assert self.projection_dimension > 0
        self.has_max_pool = max_pool
        self.pooling_projection_dimension = pooling_projection_dimension
        if self.has_max_pool:
            assert self.pooling_projection_dimension > 0
        self.relu_negative_slope = relu_negative_slope
        assert self.relu_negative_slope >= 0.0
        self.has_activation = has_activation
        self._construct_layers()

    def _construct_layers(self):
        for filter_number, filter_size, reduced_dimension in \
            zip(self.n_filters, self.kernel_sizes,
                self.reduced_dimensions):
            filter_name = '{0}x{0}_{1}'.format(filter_size, self.strides)
            reduction_name = '{}_reduce'.format(filter_name)
            if reduced_dimension is not None:
                setattr(self, reduction_name,
                        Conv2DNormRelu(reduced_dimension, 1, strides=self.strides,
                                       negative_slope=self.relu_negative_slope, has_activation=self.has_activation))
            else:
                setattr(self, reduction_name, Lambda((lambda x: x)))
            setattr(self, filter_name,
                    Conv2DNormRelu(filter_number, filter_size, strides=self.strides,
                                   negative_slope=self.relu_negative_slope, has_activation=self.has_activation))

        if self.has_projection:
            setattr(self, 'projection',
                    Conv2DNormRelu(self.projection_dimension, 1, strides=self.strides,
                                   negative_slope=self.relu_negative_slope, has_activation=self.has_activation))
        if self.has_max_pool:
            setattr(self, 'max_pool',
                    MaxPooling2D(pool_size=(3, 3), strides=self.strides,
                                 padding='same', data_format='channels_last'))
            setattr(self, 'max_pool_project',
                    Conv2DNormRelu(self.pooling_projection_dimension, 1, strides=self.strides,
                                   negative_slope=self.relu_negative_slope, has_activation=self.has_activation))

    def call(self, inputs, training=None):
        outputs = []
        for filter_size in self.kernel_sizes:
            filter_name = '{0}x{0}_{1}'.format(filter_size, self.strides)
            reduction_name = '{}_reduce'.format(filter_name)
            results = getattr(self, reduction_name)(inputs, training=training)
            results = getattr(self, filter_name)(results, training=training)
            outputs.append(results)
        if self.has_max_pool:
            results = getattr(self, 'max_pool')(inputs)
            results = getattr(self, 'max_pool_project')(results, training=training)
            outputs.append(results)
        if self.has_projection:
            results = getattr(self, 'projection')(inputs, training=training)
            outputs.append(results)
        return concatenate(outputs, axis=-1)


class WeightedSum(Layer):
    """
    this class takes a list of input tensor (of same shape), and returns
    their weighted sum. weights are non negative and trainable
    """
    def __init__(self, name=None):
        super(WeightedSum, self).__init__(trainable=True, name=name)
        self.n_summands = 0

    def build(self, input_shapes):
        self.n_summands = len(input_shapes)
        assert self.n_summands > 0
        for i in range(self.n_summands):
            setattr(self, 'summand_weight{}'.format(i),
                    self.add_weight(name='summand_weight{}'.format(i), trainable=True,
                                    initializer=tf.keras.initializers.Constant(1 / self.n_summands),
                                    constraint=tf.keras.constraints.NonNeg()))
        setattr(self, 'output_weight',
                self.add_weight(name='output_weight', trainable=True,
                                initializer=tf.keras.initializers.Constant(1),
                                constraint=tf.keras.constraints.NonNeg()))

    @tf.function
    def _sum_layers(self, inputs):
        assert isinstance(inputs, list)
        outputs = []
        for i in range(self.n_summands):
            outputs.append(tf.math.scalar_mul(getattr(self, 'summand_weight{}'.format(i)), inputs[i]))
        return add(outputs)

    def call(self, inputs, training=None):
        output = self._sum_layers(inputs)
        output = tf.math.scalar_mul(getattr(self, 'output_weight'), output)
        return output






