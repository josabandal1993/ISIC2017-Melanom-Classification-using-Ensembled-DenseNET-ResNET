""" Ensemble Melanoma:
  This model is an ensembled of DenseNET and ResNET melanoma. The best weights for of each model was used
  as the base for the input data. Live data-augmentation was used so that each training image will be
  randomly augmented and pass on to the ResNET and DenseNET model to be predicted each epoch.
  Basically the inputs of the Ensemble model are the output of the flattened layer of ResNET and DenseNET.
  Softmax will be applied to bought input before they are concatenated and send to the final Dense Layer.

  Hidden units =256, time/epoch = 272s, paramaters total =957,954 Trainable params: 957,954, test_acc= 0.87
  
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import RMSprop
from keras.layers.merge import concatenate
from keras.models import Model
import numpy as np
import os
import pickle
import sys
from ResNET_model import resnet_v2
from DenseNET_model import DenseNET
import time

#
# Training parameters
w = 64

batch_size = 4 
epochs = 200
data_augmentation = True
num_classes = 2
hidden_units = 256

num_dense_blocks = 3
use_max_pool = False

growth_rate = 12
depth_dense = 200

num_filters_bef_dense_block = 2 * growth_rate
compression_factor = 0.5
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
n = 37

version = 2

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2


print("N: ",n,"batch_size:",batch_size)
print("Training data is not undersampled")
print("Saved only best val_acc")

model_type = 'ResNet%dv%d' % (depth, version)


train = pickle.load( open( 'ISIC2017_train_melanoma_colored_'+str(w)+'.p', "rb" ) )
val = pickle.load( open( 'ISIC2017_val_melanoma_colored_'+str(w)+'.p', "rb" ) )
test = pickle.load( open( 'ISIC2017_test_melanoma_colored_'+str(w)+'.p', "rb" ) )


x_train = train['features']
y_train = train['labels']
x_test = test['features']
y_test = test['labels']

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

model_ResNET = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
model_ResNET.load_weights('saved_models/IS17_ResNET_melanoma_335_64x64p2.h5')
# model_ResNET.layers.pop()


model_DenseNET = DenseNET(input_shape, depth_dense, num_classes, num_dense_blocks, growth_rate, compression_factor, use_max_pool, data_augmentation)
model_DenseNET.load_weights('saved_models/IS17_DenseNET_melanoma_200_64p2.h5')


intermediate_layer_ResNET = Model(inputs=model_ResNET.input,
                                 outputs=model_ResNET.get_layer('flatten').output)
#intermediate_layer_ResNET.summary()
ctr = 0 
for layer in intermediate_layer_ResNET.layers:
    ctr = ctr + 1
    if(ctr <2000):
        layer.trainable = False
# print(ctr)

intermediate_layer_DenseNET = Model(inputs=model_DenseNET.input,
                                 outputs=model_DenseNET.get_layer('flatten').output)

ctr = 0 
#intermediate_layer_DenseNET.summary()

for layer in intermediate_layer_DenseNET.layers:
    ctr = ctr + 1
    if(ctr <2000):
        layer.trainable = False
print(ctr)
intermediate_layer_DenseNET.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(1e-3),
              metrics=['accuracy'])



ctr =0
while ctr < 1:
    ans1 = intermediate_layer_ResNET.predict(np.reshape(x_test[ctr],[-1,w,w,3]))
    print(np.reshape(x_test[ctr],[-1,w,w,3]).shape)
    ans2 = intermediate_layer_DenseNET.predict(np.reshape(x_test[ctr],[-1,w,w,3]))
    print(ans2.shape)
#     x= np.concatenate(([ans1,ans2]),axis=None)
#     print(x.shape)
#     x = np.reshape(x,[-1,3736])
#     print(x.shape)
#     ans = model_ResNET.predict(np.reshape(x_test[ctr],[-1,64,64,3]))
#     print(ans.shape)
#     ans = model_DenseNET.predict(np.reshape(x_test[ctr],[-1,64,64,3]))
#     print(ans.shape)
    ctr = ctr + 1


#ENSEMBLE

inputs_ResNET = Input(shape=(1,1024))
inputs_DenseNET = Input(shape=(1,2712))
y_ResNET = Flatten()(inputs_ResNET)
y_ResNET = Dense(hidden_units,
                  kernel_initializer='he_normal',
                  activation='softmax')(y_ResNET)

# y_ResNET = keras.activations.softmax(y_ResNET, axis=-1)
y_DenseNET = Flatten()(inputs_DenseNET)
y_DenseNET = Dense(hidden_units,
                  kernel_initializer='he_normal',
                  activation='softmax')(y_DenseNET)
# y_DenseNET = keras.activations.softmax(y_DenseNET, axis=-1)
y = concatenate([y_ResNET, y_DenseNET])
outputs3 = Dense(num_classes,
                  kernel_initializer='he_normal',
                  activation='softmax')(y)
model_Ensemble = Model(inputs=[inputs_ResNET,inputs_DenseNET], outputs=outputs3)

model_Ensemble.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

model_Ensemble.summary()
def ensemble_generator(batches):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    
    while True:
        batch_x, batch_y = next(batches)
        batch_crops1 = np.zeros((batch_x.shape[0], 1, 1024))
        batch_crops2 = np.zeros((batch_x.shape[0], 1, 2712))
        for i in range(batch_x.shape[0]):
            b1 = intermediate_layer_ResNET.predict(np.reshape(batch_x[i],[-1,w,w,3]))
            b2 = intermediate_layer_DenseNET.predict(np.reshape(batch_x[i],[-1,w,w,3]))
            # batch_z = np.concatenate(([batch_x1,batch_x2]),axis=None)
            batch_crops1[i] = np.reshape(b1,[-1,1024])
            batch_crops2[i] = np.reshape(b2,[-1,2712])
        yield ([batch_crops1, batch_crops2], batch_y)


#model_Ensemble.summary()

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_na = 'IS17_melanoma_ensemble_'+str(hidden_units)+'_64p2'
model_name = model_na +'.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

csv_logger = CSVLogger(model_na+'.log')

callbacks = [checkpoint, lr_reducer, lr_scheduler, csv_logger]


#This part is for the conversion of the test dataset without online data-augmentation
i = 0
b_crops1 = np.zeros((len(y_test), 1, 1024))
b_crops2 = np.zeros((len(y_test), 1, 2712))
while i < len(y_test):
    b1 = intermediate_layer_ResNET.predict(np.reshape(x_test[i],[-1,w,w,3]))
    b2 = intermediate_layer_DenseNET.predict(np.reshape(x_test[i],[-1,w,w,3]))
    b_crops1[i] = np.reshape(b1,[-1,1024])
    b_crops2[i] = np.reshape(b2,[-1,2712])
    i = i+1
x_val = [b_crops1,b_crops2]
y_val = y_test


if not data_augmentation:
    print('Not using data augmentation.')
    model_Ensemble.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (deg 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)
    # Fit the model on the batches generated by datagen.flow().
    model_Ensemble.fit_generator(ensemble_generator(datagen.flow(x_train, y_train, batch_size=batch_size)),
                        steps_per_epoch = len(y_train) // batch_size,
                        validation_data=(x_val, y_val),
                        validation_steps = len(y_val) // batch_size,
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)



# Loads best weight, if you just want to the prediction and evaluation of data
model_Ensemble.load_weights('saved_models/'+model_name)
model_Ensemble.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

scores = model_Ensemble.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])