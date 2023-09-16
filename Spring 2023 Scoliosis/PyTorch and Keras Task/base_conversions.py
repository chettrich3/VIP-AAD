import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


################################### PYTORCH ###################################

import torch.nn as nn
from torchinfo import summary

class NeuralNet(nn.Module):
  def __init__(self):
    super(NeuralNet, self).__init__()

    ### BASE NETWORK 

    self.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding = (3, 3), bias = False)
    self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    ### LAYER 2 DOWNSAMPLE

    self.layer_2_zero = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    self.layer_2_one = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    ### LAYER 3 DOWNSAMPLE

    self.layer_3_zero = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    self.layer_3_one = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    ### LAYER 4 DOWNSAMPLE

    self.layer_4_zero = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    self.layer_4_one = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    ### DEC_C4 - UP

    self.dec_c4_up_zero = Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.dec_c4_up_one = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.dec_c4_up_two = ReLU(inplace)

    ### DEC_C4 - CAT_CONV

    self.dec_c4_cat_conv_zero = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    self.dec_c4_cat_conv_one = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.dec_c4_cat_conv_two = ReLU(inplace)

    ### HM

    self.hm_zero = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.hm_one = ReLU(inplace)
    self.hm_two = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    
    ### REG

    self.reg_zero = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.reg_one = ReLU(inplace)
    self.reg_two = Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    
    ### WH

    self.wh_zero = Conv2d(64, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    self.wh_one = ReLU(inplace)
    self.wh_two = Conv2d(256, 8, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))

model = NeuralNet()
summary(model)

#################################### KERAS ####################################

from tensorflow import keras

# model will infer the input_shape based on the call

model = keras.Sequential()

### BASE NETWORK 

model.add(keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", use_bias=False)) # conv1
model.add(keras.layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-05, center=True, scale=True)) # bn1
model.add(keras.layers.Activation('relu')) # relu
model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same", data_format="channels_last")) # maxpool
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)) # conv2
model.add(keras.layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-05, center=True, scale=True)) # bn2

### LAYER 2 DOWNSAMPLE

model.add(keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='valid', use_bias=False)) # layer_2_zero
model.add(keras.layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-05, center=True, scale=True)) # layer_2_one

### LAYER 3 DOWNSAMPLE

model.add(keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='valid', use_bias=False)) # layer_3_zero
model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-05, center=True, scale=True)) # layer_3_one

### LAYER 4 DOWNSAMPLE

model.add(keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2), padding='valid', use_bias=False)) # layer_4_zero
model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-05, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')) # layer_4_zero

### DEC_C4 - UP

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same') # dec_c4_up_zero
model.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-05, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')) # dec_c4_up_one
model.add(Activation('relu')) # dec_c4_up_two

### DEC_C4 - CAT_CONV

model.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)) # dec_c4_cat_conv_zero
model.add(BatchNormalization(epsilon=1e-05, momentum=0.1, center=True, scale=True)) # dec_c4_cat_conv_one
model.add(ReLU()) # dec_c4_cat_conv_two

### HM

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same') # hm_zero
model.add(ReLU()) # hm_one
model.add(Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)) # hm_two

### REG

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)) # reg_zero
model.add(ReLU()) # reg_one
model.add(Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), use_bias=True)) # reg_two

### WH

model.add(Conv2D(filters=256, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', use_bias=True)) # wh_zero
model.add(ReLU()) # wh_one
model.add(Conv2D(filters=8, kernel_size=(7, 7), strides=(1, 1), padding='same', activation=None)) # wh_two

print(model.summary())

# BatchNormalization weights are two dimensional while in Conv2D the weights are four dimensional.
# ReLu does not have any weights, if the numbers are less than zero it makes them zero, if the numbers are greater than zero it does not touch them
# whenever a layer gets trained, it does a lot of math, and the weights are the coefficinets to the math being done
# if the weights are not copied over then the results are going to be completely different

# When there is padding make sure to add the ZeroPadding2d with the appropriate parameters
# The weights for conv2d need to be transposed