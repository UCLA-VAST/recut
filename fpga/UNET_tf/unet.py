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

## Seeding 
seed = 2019
random.seed = seed
np.random.seed = seed
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

image_size = 128
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

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    
    model = keras.models.Model(inputs, outputs)
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


train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)




## Save the Weights
# model.save_weights("UNetW.h5")
model.load_weights("D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/UNET_tf/UNetW.h5")


## Dataset for prediction
x, y = valid_gen.__getitem__(1)
# print(x.shape)
result = model.predict(x)
# l1 = model.get_layer('input_1')
# print(l1)
result = result > 0.5

x = np.reshape(x[0], [1, 128, 128, 3])


layerIndex = int(sys.argv[1])
# print(layerIndex)
dumpLayerOutputs(layerIndex, False)
# dumpLayerOutputs("conv2d", False)
# dumpLayerOutputs(2, False)
# dumpLayerOutputs("max_pooling2d", False)
# i = 0
# while(model.get_layer(index=i)!=None):
#     print(model.get_layer(index=i).name)
#     i+=1



fileNames = []
i=1
while(i<layerIndex+1):
    if(dumpLayerWeights(i)!=None):
        fileNames.append(dumpLayerWeights(i))
    i+=1

# fileNames.append(dumpLayerWeights("conv2d_1"))


with open('D:/Winter2021/Research/FlexCNN/SDx_project/FlexCNN_opt/data/weights.dat', 'w') as outfile:
    for fname in fileNames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)


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
