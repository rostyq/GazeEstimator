# coding: utf-8

import pandas as pd
from .utils import *
from sklearn.model_selection import train_test_split

# If there is no dataset and utils:
# TODO bash command (maybe be crossplatform?)
get_ipython().system('wget https://raw.githubusercontent.com/rostyslavb/GazeEstimator/master/model/utils.py')
get_ipython().system('wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz')
get_ipython().system('tar -xzfv MPIIGaze.tar.gz')

# ## Prepare data

# ### Gather Data from Structure
index, image, pose, gaze = gather_all_data('./MPIIGaze/Data/Normalized')

# prepare data
gaze = gaze3Dto2D(gaze)
pose = pose3Dto2D(pose)

print_shapes(['Indices', 'Images', 'Poses', 'Gazes'], (index, image, pose, gaze))


random_state = 42
index_train, index_test = train_test_split(pd.DataFrame(index),
                                           stratify=index[:, [0, -1]],
                                           test_size=0.2,
                                           random_state=random_state)

index_train = index_train.index
index_test = index_test.index

# **Train:**
print_shapes(['Indices', 'Images', 'Poses', 'Gazes'],
             (index[index_train], image[index_train], pose[index_train], gaze[index_train]))

# **Test:**
print_shapes(['Indices', 'Images', 'Poses', 'Gazes'],
             (index[index_test], image[index_test], pose[index_test], gaze[index_test]))


# ## Create NN
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Concatenate, Flatten, Dropout
from keras.initializers import RandomNormal, glorot_normal
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf


# ### Model
def calc_angle(vector1, vector2):
    def to_vector(array):
        x = (-1) * K.cos(array[:, 0]) * K.sin(array[:, 1])
        y = (-1) * K.sin(array[:, 0])
        z = (-1) * K.cos(array[:, 0]) * K.cos(array[:, 1])

        return tf.stack((x, y, z), axis=1)

    def calc_norm(array):
        return tf.norm(array, axis=1)

    v1, v2 = to_vector(vector1), to_vector(vector2)
    norm1, norm2 = calc_norm(vector1), calc_norm(vector2)

    angle_value = tf.divide(tf.reduce_sum(tf.multiply(v1, v2), axis=1),
                            tf.multiply(norm1, norm2))

    return tf.where(tf.abs(angle_value) >= 1.0, tf.pow(angle_value, -1), angle_value)


def angle_loss(target, predicted):
    return K.mean(1 - calc_angle(target, predicted))


def angle_accuracy(target, predicted):
    return K.mean(tf.acos(calc_angle(target, predicted)) * 180 / 3.14159265)


### LAYERS ###

# input
input_img = Input(shape=(36, 60, 1), name='InputNormalizedImage')
input_pose = Input(shape=(2,), name='InputHeadPose')

# convolutional
conv1 = Conv2D(filters=20,
               kernel_size=(5, 5),
               strides=(1, 1),
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=42),
               bias_initializer='zeros',
               name='conv1'
               )(input_img)
pool1 = MaxPool2D(pool_size=(2, 2),
                  strides=(2, 2),
                  padding='valid',
                  name='maxpool1'
                  )(conv1)
conv2 = Conv2D(filters=50,
               kernel_size=(5, 5),
               strides=(1, 1),
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001, seed=42),
               bias_initializer='zeros',
               name='conv2'
               )(pool1)
pool2 = MaxPool2D(pool_size=(2, 2),
                  strides=(2, 2),
                  padding='valid',
                  name='maxpool2'
                  )(conv2)

flatt = Flatten(name='flatt')(pool2)

# inner product 1
dense1 = Dense(units=500,
               activation='relu',
               kernel_initializer=glorot_normal(seed=42),
               bias_initializer='zeros',
               name='ip1'
               )(flatt)

# concatanate with head pose
cat = Concatenate(axis=-1, name='concat')([dense1, input_pose])

dropout = Dropout(rate=0.1)(cat)

# inner product 2
dense2 = Dense(units=2,
               kernel_initializer=glorot_normal(seed=42),
               bias_initializer='zeros',
               name='ip2'
               )(dropout)

### OPTIMIZER ###
adam = Adam(lr=1e-5)

### CALLBACKS ###
tbCallBack = TensorBoard(log_dir='./log',
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True)
# TODO callbacks
# checkpoint = ModelCheckpoint('./checkpoints/', monitor='val_loss', period=10)
# earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10)

### COMPILE MODEL ###
model = Model([input_img, input_pose], dense2)
model.compile(optimizer=adam, loss=angle_loss, metrics=[angle_accuracy])

# ### Train
model.fit(x=[image[index_train], pose[index_train]], y=gaze[index_train],
          batch_size=500,
          verbose=1,
          epochs=1,
          validation_data=([image[index_test], pose[index_test]], gaze[index_test]),
          callbacks=[tbCallBack])

model.save('./model_epoch1.h5')
