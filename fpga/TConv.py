
## Imports
import os
import sys
import random

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

def TCONV():
  inputs = keras.layers.Input((32, 32, 16))
  # outputs = keras.layers.UpSampling2D((2, 2))(inputs)
  outputs = keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='valid', dilation_rate=(1, 1), activation="relu")(inputs)
  # outputs = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(up)
  model = keras.models.Model(inputs, outputs)
  return model

model = TCONV()
np.random.seed(42)
x = np.random.rand(1,32,32,16)
x = np.ones((1,32,32,16))

# x.tofile("D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/inputs.dat", sep="\n", format="%s")


# x_ = x
# x_[0][0] = 0
# x_[0][len(x_[0])-1] = 0
# x_ = np.swapaxes(x_,1,2)
# x_[0][0] = 0
# x_[0][len(x_[0])-1] = 0
# x = np.swapaxes(x_,1,2)

# x_ = np.swapaxes(x_,1,3)
# x_ = np.swapaxes(x_,2,3)
# df = pd.DataFrame(x_[0][1])
# print(df)

# x_ = np.swapaxes(x_,2,3)
# x_[0][0][0] = 0
# x_[0][0][len(x_[0][0])-1] = 0
# df = pd.DataFrame(x_[0][0])
# print(df)
layer = model.get_layer('conv2d_transpose')
w = layer.get_weights()
w[0].fill(1)
layer.set_weights(w)

intermediate_layer_model = keras.models.Model(inputs=model.input,
                                outputs=layer.output)
intermediate_output = intermediate_layer_model.predict(x)

# print(x.shape)
result = model.predict(x)
print(result.shape)
result = np.swapaxes(result,1,3)
print(result.shape)
result = np.swapaxes(result,2,3)
print(result.shape)
# model.summary()
df = pd.DataFrame(result[0][0])
print(df)

# # OUT = ((32 - 1) * 2 + 3 - 2 * 1)
# # print(OUT)