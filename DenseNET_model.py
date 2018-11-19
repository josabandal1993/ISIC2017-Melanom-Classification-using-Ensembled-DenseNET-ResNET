"""DenseNET Model patterened from Roel Atienza's github implementation
"""
from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras import backend as K
from keras.layers.merge import concatenate
from keras.models import Model

def DenseNET(input_shape, depth, num_classes, num_dense_blocks, growth_rate, compression_factor, use_max_pool, data_augmentation):
  num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
  num_filters_bef_dense_block = 2 * growth_rate

  inputs = Input(shape=input_shape)
  x = BatchNormalization()(inputs)
  x = Activation('relu')(x)
  x = Conv2D(num_filters_bef_dense_block,
             kernel_size=3,
             padding='same',
             kernel_initializer='he_normal')(x)
  x = concatenate([inputs, x])

  # stack of dense blocks bridged by transition layers
  for i in range(num_dense_blocks):
      # a dense block is a stack of bottleneck layers
      for j in range(num_bottleneck_layers):
          y = BatchNormalization()(x)
          y = Activation('relu')(y)
          y = Conv2D(4 * growth_rate,
                     kernel_size=1,
                     padding='same',
                     kernel_initializer='he_normal')(y)
          if not data_augmentation:
              y = Dropout(0.2)(y)
          y = BatchNormalization()(y)
          y = Activation('relu')(y)
          y = Conv2D(growth_rate,
                     kernel_size=3,
                     padding='same',
                     kernel_initializer='he_normal')(y)
          if not data_augmentation:
              y = Dropout(0.2)(y)
          x = concatenate([x, y])

      # no transition layer after the last dense block
      if i == num_dense_blocks - 1:
          continue

      # transition layer compresses num of feature maps and reduces the size by 2
      num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
      num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
      y = BatchNormalization()(x)
      y = Conv2D(num_filters_bef_dense_block,
                 kernel_size=1,
                 padding='same',
                 kernel_initializer='he_normal')(y)
      if not data_augmentation:
          y = Dropout(0.2)(y)
      x = AveragePooling2D()(y)


  # add classifier on top
  # after average pooling, size of feature map is 1 x 1
  x = AveragePooling2D(pool_size=8)(x)
  y = Flatten(name='flatten')(x)
  outputs = Dense(num_classes,
                  kernel_initializer='he_normal',
                  activation='softmax')(y)

  # instantiate and compile model
  # orig paper uses SGD but RMSprop works better for DenseNet
  model = Model(inputs=inputs, outputs=outputs)

  return model