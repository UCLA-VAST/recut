## Imports
from ast import walk
import os
import sys
import random

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal
from tensorflow.keras.layers import Layer, Conv2D, Softmax, BatchNormalization, \
    LayerNormalization, ReLU, Lambda, Conv2DTranspose, MaxPooling2D, concatenate, add
# from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras import Model

## Seeding 
seed = 2019
random.seed = seed
# np.random.seed = seed
tf.seed = seed

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, id_name, "images", id_name) + ".png"
        mask_path = os.path.join(self.path, id_name, "masks/")
        all_masks = os.listdir(mask_path)
        
        ## Reading Image
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        mask = np.zeros((self.image_size, self.image_size, 1))
        
        ## Reading Masks
        for name in all_masks:
            _mask_path = mask_path + name
            _mask_image = cv2.imread(_mask_path, -1)
            _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size)) #128x128
            _mask_image = np.expand_dims(_mask_image, axis=-1)
            mask = np.maximum(mask, _mask_image)
            
        ## Normalizaing 
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

image_size = 256
train_path = "D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/UNET_tf/dataset/stage1_train/"
epochs = 5
batch_size = 8

## Training Ids
train_ids = next(os.walk(train_path))[1]

## Validation Data Size
val_data_size = 10

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
x, y = gen.__getitem__(0)
# print(x.shape, y.shape)

r = random.randint(0, len(x)-1)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(x[r])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(y[r], (image_size, image_size)), cmap="gray")


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    print(filters)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

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
    
  


  #     else:
  #         activation_func = lambda x: relu(x, alpha=self.negative_slope)
  #         setattr(self, conv2d_name,
  #                 Conv2D(self.n_filters, self.kernel_size, strides=1, use_bias=True, activation=activation_func,
  #                         dtype=self.dtype, padding='same', data_format='channels_last', name=conv2d_name,
  #                         kernel_initializer=self.kernal_initializer, bias_initializer=tf.initializers.Constant(0)))
  # conv2d_name = 'conv2d_{0}'.format(self.n_conv2d - 1)
  # setattr(self, conv2d_name,
  #         Conv2D(self.n_filters, self.kernel_size, strides=1, use_bias=True, dtype=self.dtype,
  #                 padding='same', data_format='channels_last', name=conv2d_name,
  #                 kernel_initializer=self.kernal_initializer, bias_initializer=tf.initializers.Constant(0)))
  # # if input and output dimensions are consistent, identity mapping
  # if input_shape[-1] == self.n_filters:
  #     self.projection = Lambda(lambda x: x, name='identity')
  # # 1x1 convolution to adjust dimension
  # else:
  #     self.projection = Conv2D(self.n_filters, 1, strides=1, use_bias=True, dtype=self.dtype,
  #                               padding='same', data_format='channels_last', name='projection',
  #                               kernel_initializer=self.kernal_initializer, bias_initializer=tf.initializers.Constant(0))
  
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
    # # adding batch norm before softmax gives worse training result
    # softmax = Softmax(axis=-1, name='softmax')(conv2d_logits)
    # c1, p1 = down_block(p0, f[0]) #128 -> 64
    # c2, p2 = down_block(p1, f[1]) #64 -> 32
    # c3, p3 = down_block(p2, f[2]) #32 -> 16
    # c4, p4 = down_block(p3, f[3]) #16->8
    
    # bn = bottleneck(p4, f[4])
    
    # u1 = up_block(bn, c4, f[3]) #8 -> 16
    # u2 = up_block(u1, c3, f[2]) #16 -> 32
    # u3 = up_block(u2, c2, f[1]) #32 -> 64
    # u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    # outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    
    model = keras.models.Model(inputs, softmax)
    return model

def dumpLayerOutputs(layerIndex, setWeightsOnes):
    layer = model.get_layer(index=layerIndex)
    w = layer.get_weights()
    if setWeightsOnes==True:
        # print(w[0].shape)
        w[0].fill(1)
        layer.set_weights(w)
    intermediate_layer_model = keras.models.Model(inputs=model.input,
                                    outputs=layer.output)
    intermediate_output = intermediate_layer_model.predict(x)
    intermediate_output = np.swapaxes(intermediate_output,1,3)
    intermediate_output.tofile("D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/data/L"+str(layerIndex)+"_outputs.dat", sep="\n", format="%s")

def dumpLayerWeights(layerIndex):
    layer = model.get_layer(index=layerIndex)
    if len(layer.get_weights())>1:
        w = layer.get_weights()[0]
        print(w.shape)
        w = np.swapaxes(w,2,1)
        w = np.swapaxes(w,1,0)
        if layerIndex==1:
            w_z = np.zeros([5,3,3,16])
            w = np.concatenate((w, w_z), axis=0)
        w = np.swapaxes(w,0,3)
        w = np.swapaxes(w,1,2)
        # print(w.shape)
        w.tofile("D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/data/L"+str(layerIndex)+"_weights.dat", sep="\n", format="%s")
        return "D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/data/L"+str(layerIndex)+"_weights.dat"
    else:
        return None

model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()

# for layer in model.layers:
#   print("output of " + layer.name + '' + str(layer.input.shape))
train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)




## Save the Weights
# model.save_weights("recut.h5")
model.load_weights("unet_recut.h5")

# model.load_weights("D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/UNET_tf/recut_server.h5")
# for layer in model.layers:
#         weights = layer.get_weights()
#         for w in weights:
#             w.fill(0.1)
#         layer.set_weights(weights)

## Dataset for prediction
# x, y = valid_gen.__getitem__(1)
x = np.ones(shape=(1,256,256,3))
np.random.seed(2019)
inputs = np.random.rand(1,256,256,3)
inputs -= 0.5
layer = model.get_layer('batch_normalization')
w = layer.get_weights()
# print(w)#w[0] = gamma, w[1] = beta, w[2] = u (mean), w[3] = variance normalization = (gamma*(x-u)/variance + beta)

gamma = []
beta = []
for i in range(len(w[0])):
  gamma.append(w[0][i]/w[3][i])
  beta.append((w[1][i])-(w[0][i]*w[2][i]/w[3][i]))

print(gamma[2]*inputs[0][0][0][2] + beta[2])
print(gamma[2]*inputs[0][0][1][2] + beta[2])
print(gamma[2]*inputs[0][0][2][2] + beta[2])
# print(gamma)
# print(beta)
intermediate_layer_model = keras.models.Model(inputs=model.input,
                                outputs=layer.output)
intermediate_output = intermediate_layer_model.predict(inputs)
# print(inputs)
print(intermediate_output)

print(((0.3413472*(0.40348221-0.004217315))/1.0154312)+0.0699169)
# result = model.predict(inputs)
# print(result)
# result.tofile("D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/data/output.dat", sep="\n", format="%s")
# l1 = model.get_layer('input_1')
# print(l1)
# result = result > 0.5

# x = np.reshape(x[0], [1, 128, 128, 3])


# layerIndex = int(sys.argv[1])
# print(layerIndex)
# dumpLayerOutputs(layerIndex, False)
# dumpLayerOutputs("conv2d", False)
# dumpLayerOutputs(2, False)
# dumpLayerOutputs("max_pooling2d", False)
# i = 0
# while(model.get_layer(index=i)!=None):
#     print(model.get_layer(index=i).name)
#     i+=1



# fileNames = []
# i=1
# while(i<layerIndex+1):
#     if(dumpLayerWeights(i)!=None):
#         fileNames.append(dumpLayerWeights(i))
#     i+=1

# # fileNames.append(dumpLayerWeights("conv2d_1"))


# with open('D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/data/weights.dat', 'w') as outfile:
#     for fname in fileNames:
#         with open(fname) as infile:
#             for line in infile:
#                 outfile.write(line)


# layer = model.get_layer('conv2d_1')
# w = layer.get_weights()
# print(w[0].shape)
# w[0].fill(1)
# layer.set_weights(w)
# intermediate_layer_model = keras.models.Model(inputs=model.input,
#                                  outputs=layer.output)
# intermediate_output = intermediate_layer_model.predict(x)
# intermediate_output = np.swapaxes(intermediate_output,1,3)
# print(intermediate_output.shape)
# intermediate_output.tofile("output_L2.txt", sep="\n", format="%s")


# x = np.swapaxes(x,1,3)
# print(x.shape)
# x.tofile("inputs.txt", sep="\n", format="%s")





# a = np.zeros([2,2,3,1])
# b = np.zeros([1,2,3,1])
# print(a.shape)
# print(b.shape)
# com = np.concatenate((a, b), axis=0)
# print(com.shape)
