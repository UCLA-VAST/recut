## Imports
from ast import walk
import sys
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal
from tensorflow.keras.layers import Layer, Conv2D, Softmax, BatchNormalization, \
    LayerNormalization, ReLU, Lambda, Conv2DTranspose, MaxPooling2D, concatenate, add
# from tensorflow_addons.layers import InstanceNormalization
# from tensorflow.keras import Model

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils.vis_utils import plot_model

layerIndex = int(sys.argv[1])
prj_path = sys.argv[2]

## Seeding 
seed = 2019
random.seed = seed
# np.random.seed = seed
tf.seed = seed

image_size = 256

def get_norm_name(norm_class):
    if norm_class == BatchNormalization:
        return 'batch_norm'
    # elif norm_class == InstanceNormalization:
    #     return 'instance_norm'
    else:
        return 'layer_norm'

def Conv2DNormRelu(inputs, n_filters, kernel_size, strides=1,
                 norm_class=BatchNormalization, negative_slope=0.01,
                 has_activation=True, pre_activation=False,
                 kernal_initializer=GlorotNormal(), dtype=tf.float32, name=None):
  if pre_activation:
    norm = norm_class(axis=-1, trainable=True, center=True, scale=True, name=get_norm_name(norm_class))(inputs)
    if has_activation:
        relu = ReLU(negative_slope=negative_slope, name='relu')(norm)
    else:
        relu = Lambda(lambda x: x, name='identity')(norm)
    conv = Conv2D(n_filters, kernel_size, strides=strides, use_bias=True,
                                dtype=dtype, padding='same', data_format='channels_last',
                                kernel_initializer=kernal_initializer,
                                bias_initializer=tf.initializers.Constant(0))(relu)
    return conv
  else:
    conv = Conv2D(n_filters, kernel_size, strides=strides, use_bias=True,
                            dtype=dtype, padding='same', data_format='channels_last',
                            kernel_initializer=kernal_initializer,
                            bias_initializer=tf.initializers.Constant(0))(inputs)
    norm = norm_class(axis=-1, trainable=True, center=True, scale=True)(conv)
    if has_activation:
        relu = ReLU(negative_slope=negative_slope)(norm)
    else:
        relu = Lambda(lambda x: x)(norm)
    return relu

def Conv2DResidualBlockV1(inputs, n_filters, kernel_size, n_conv2d=1, negative_slope=0.01,
                 norm_class=BatchNormalization, internal_normalize=False, has_activation=True,
                 kernal_initializer=GlorotNormal(), dtype=tf.float32, name=None):
   # bn and relu layers for input
  input_norm = norm_class(axis=-1, trainable=True, center=True, scale=True)(inputs)
  input_relu = ReLU(negative_slope=negative_slope)(input_norm)
  # return input_relu
  for i in range(n_conv2d - 1):
      conv2d_name = 'conv2d_{0}'.format(i)
      if internal_normalize:
        input_relu = Conv2DNormRelu(n_filters, kernel_size, strides=1,
                        norm_class=norm_class, negative_slope=negative_slope,
                        has_activation=True, kernal_initializer=kernal_initializer,
                        dtype=dtype)(input_relu)
      else:
        activation_func = lambda x: relu(x, alpha=negative_slope)

        input_relu = Conv2D(n_filters, kernel_size, strides=1, use_bias=True, activation=activation_func,
                            dtype=dtype, padding='same', data_format='channels_last',
                            kernel_initializer=kernal_initializer, bias_initializer=tf.initializers.Constant(0))(input_relu)
  conv2d_name = 'conv2d_{0}'.format(n_conv2d - 1)
  conv = Conv2D(n_filters, kernel_size, strides=1, use_bias=True, dtype=dtype,
                       padding='same', data_format='channels_last',
                       kernel_initializer=kernal_initializer, bias_initializer=tf.initializers.Constant(0))(input_relu)
  # if inputs.shape[-1] == n_filters:
  #     projection = Lambda(lambda x: x, name='identity')(conv)
  # # 1x1 convolution to adjust dimension
  # else:
  projection = Conv2D(n_filters, 1, strides=1, use_bias=True, dtype=dtype,
                                     padding='same', data_format='channels_last',
                                     kernel_initializer=kernal_initializer, bias_initializer=tf.initializers.Constant(0))(inputs)
  concat = keras.layers.add([projection, conv])

  return concat

def Conv2DTransposeNormRelu(inputs, n_filters, kernel_size, strides=1,
                 norm_class=BatchNormalization, negative_slope=0.01,
                 has_activation=True, pre_activation=False,
                 kernal_initializer=GlorotNormal(), dtype=tf.float32, name=None):
  print(pre_activation)
  if pre_activation:
    norm = norm_class(axis=-1, trainable=True, center=True, scale=True)(inputs)
    if has_activation:
        relu = ReLU(negative_slope=negative_slope)(norm)
    else:
        relu = Lambda(lambda x: x)(norm)
    convt = Conv2DTranspose(n_filters, kernel_size, strides=strides, use_bias=True,
                                     dtype=dtype, padding='same', data_format='channels_last',
                                     kernel_initializer=kernal_initializer, bias_initializer=tf.initializers.Constant(0))(relu)
    return convt
  else:

    convt = Conv2DTranspose(n_filters, kernel_size, strides=strides, use_bias=True,
                                      dtype=dtype, padding='same', data_format='channels_last',
                                      kernel_initializer=kernal_initializer, bias_initializer=tf.initializers.Constant(0))(inputs)
    norm = norm_class(axis=-1, trainable=True, center=True, scale=True)(convt)
    if has_activation:
        relu = ReLU(negative_slope=negative_slope)(norm)
    else:
        relu = Lambda(lambda x: x)(norm)
    return relu
    
  
def UNet():
    norm_class = BatchNormalization
    negative_slope = 0.001
    kernal_initializer = GlorotNormal()
    contract_filters0 = 32
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    conv2d_res0 = Conv2DResidualBlockV1(inputs, contract_filters0, 5, n_conv2d=2, internal_normalize=False, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_res0')
    pool0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='pool0')(conv2d_res0)
    
    contract_filters1 = contract_filters0 * 2
    conv2d_res1 = Conv2DResidualBlockV1(pool0, contract_filters1, 3, n_conv2d=2, internal_normalize=False, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_res1')
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='pool1')(conv2d_res1)

    contract_filters2 = contract_filters1 * 2
    conv2d_res2 = Conv2DResidualBlockV1(pool1, contract_filters2, 3, n_conv2d=2, internal_normalize=False, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_res2')
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='pool2')(conv2d_res2)

    contract_filters3 = contract_filters2 * 2
    conv2d_res3 = Conv2DResidualBlockV1(pool2, contract_filters3, 3, n_conv2d=2, internal_normalize=False, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_res3')
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='pool3')(conv2d_res3)
    
    expand_filters3 = contract_filters3
    conv2dt3 = Conv2DTransposeNormRelu(pool3, expand_filters3, 3, pre_activation=True, strides=2, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2dt3')
    concat3 = keras.layers.add([conv2dt3, conv2d_res3])

    expand_filters2 = expand_filters3 // 2
    conv2dt2 = Conv2DTransposeNormRelu(concat3, expand_filters2, 3, pre_activation=True, strides=2, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2dt2')
    concat2 = keras.layers.add([conv2dt2, conv2d_res2])

    expand_filters1 = expand_filters2 // 2
    conv2dt1 = Conv2DTransposeNormRelu(concat2, expand_filters1, 3, pre_activation=True, strides=2, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2dt1')
    concat1 = keras.layers.add([conv2dt1, conv2d_res1])

    expand_filters0 = expand_filters1 // 2
    conv2dt0 = Conv2DTransposeNormRelu(concat1, expand_filters0, 3, pre_activation=True, strides=2, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2dt0')
    concat0 = keras.layers.add([conv2dt0, conv2d_res0])

    conv2d_logits = Conv2DNormRelu(concat0, 3, 1, pre_activation=True, strides=1, kernal_initializer=kernal_initializer, norm_class=norm_class, negative_slope=negative_slope, name='conv2d_logits')
    
    softmax = Softmax(axis=-1, name='softmax')(conv2d_logits)
    
    model = keras.models.Model(inputs, softmax)
    return model

def dumpLayerOutputs(layerIndex, inputs):
    layer = model.get_layer(layers_dict[layerIndex])
    intermediate_layer_model = keras.models.Model(inputs=model.input,
                                    outputs=layer.output)
    intermediate_output = intermediate_layer_model.predict(inputs)
    intermediate_output = np.swapaxes(intermediate_output,1,3)
    intermediate_output = np.swapaxes(intermediate_output,2,3)
    intermediate_output.tofile(prj_path+"/data/L"+str(layerIndex)+"_outputs.dat", sep="\n", format="%s")

def dumpLayerWeights(layerIndex):
    layer = model.get_layer(layers_dict[layerIndex])
    if len(layer.get_weights())>1:
        w = layer.get_weights()[0]
        w = np.swapaxes(w,2,1)
        w = np.swapaxes(w,1,0)
        if layerIndex==1:
            w_z = np.zeros([5,1,1,32])
            w = np.concatenate((w, w_z), axis=0)
        if layerIndex==2:
            w_z = np.zeros([5,5,5,32])
            w = np.concatenate((w, w_z), axis=0)
        w = np.swapaxes(w,0,3)
        w.tofile(prj_path + "/data/L"+str(layerIndex)+"_weights.dat", sep="\n", format="%s")
        return prj_path + "/data/L"+str(layerIndex)+"_weights.dat"
    else:
        return None

model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.load_weights(prj_path+"/UNET_tf/unet_recut.h5")


np.random.seed(2019)
inputs = np.random.rand(1,256,256,3)
inputs -= 0.5

layers = model.layers
layers_dict = {}
for i in range(4):
    if i==0:
        layers_dict[i*5+1] = 'conv2d_'+str(i*3+2)
        layers_dict[i*5+2] = 'conv2d'
        layers_dict[i*5+3] = 'conv2d_'+str(i*3+1)
        layers_dict[i*5+4] = 'add'
        layers_dict[i*5+5] = 'pool0'
    else:
        layers_dict[i*5+1] = 'conv2d_'+str(i*3+2)
        layers_dict[i*5+2] = 'conv2d_'+str(i*3)
        layers_dict[i*5+3] = 'conv2d_'+str(i*3+1)
        layers_dict[i*5+4] = 'add_'+str(i)
        layers_dict[i*5+5] = 'pool'+str(i)

dumpLayerOutputs(layerIndex, inputs)

fileNames = []
i=1
while(i<layerIndex+1):
    if(dumpLayerWeights(i)!=None):
        fileNames.append(dumpLayerWeights(i))
    i+=1

with open(prj_path + '/data/weights.dat', 'w') as outfile:
    for fname in fileNames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
            outfile.write('\n')