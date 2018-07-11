from keras.layers import Input
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Dropout

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.normalization import BatchNormalization

from keras.initializers import RandomNormal
from keras.initializers import glorot_uniform

from keras.regularizers import l2

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TerminateOnNaN

from keras.optimizers import SGD
from keras.models import Model
from keras import backend as K
from numpy import pi

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from os import path as Path

debug = False
if debug:
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)
else:
    pass


def calc_angle(angles1, angles2):

    def to_vector(angle):
        x = (-1) * tf.cos(angle[:, 1]) * tf.sin(angle[:, 0])
        y = (-1) * tf.sin(angle[:, 1])
        z = (-1) * tf.cos(angle[:, 1]) * tf.cos(angle[:, 0])
        return tf.stack((x, y, z), axis=1)

    def unit_vector(array):
        return tf.divide(array, tf.norm(array, axis=1, keep_dims=True))

    unit_v1, unit_v2 = unit_vector(to_vector(angles1)), unit_vector(to_vector(angles2))

    return tf.acos(
        tf.clip_by_value(tf.reduce_sum(unit_v1 * unit_v2, axis=1), -1.0, 1.0),
        name='acos'
        ) * 180 / pi


def angle_accuracy(target, predicted):
    return tf.reduce_mean(calc_angle(predicted, target), name='mean_angle')


def custom_loss(y_true_and_weights, y_pred):
   y_true, y_weights = y_true_and_weights[:, :-1], y_true_and_weights[:, -1:]
   loss = K.reshape(K.sum(K.square(y_pred - y_true), axis=1), (1, -1))
   return K.dot(2 * pi / y_weights, loss)/K.cast(K.shape(y_true)[0], 'float32')


def create_model(learning_rate=1e-1, seed=None):

    # input
    input_img = Input(shape=(72, 120, 1), name='InputImage')
    input_pose = Input(shape=(2,), name='InputPose')

    regularizer = l2(1e-5)

    # convolutional
    conv1 = Conv2D(
        filters=32,
        activation='relu',
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.1, seed=seed),
        bias_initializer='zeros',
        # kernel_regularizer=regularizer,
        name='conv1'
        )(input_img)
    pool1 = MaxPool2D(
        pool_size=(4, 4),
        strides=(4, 4),
        padding='valid',
        name='maxpool1'
        )(conv1)
    conv2 = Conv2D(
        filters=64,
        activation='relu',
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.1, seed=seed),
        bias_initializer='zeros',
        # kernel_regularizer=regularizer,
        name='conv2'
        )(pool1)
    pool2 = MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid',
        name='maxpool2'
        )(conv2)
    conv3 = Conv2D(
        filters=96,
        activation='relu',
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=seed),
        bias_initializer='zeros',
        # kernel_regularizer=regularizer,
        name='conv3'
        )(pool2)
    pool3 = MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid',
        name='maxpool3'
        )(conv3)

    flatt = Flatten(name='flatt')(pool3)

    # concatanate with head pose
    cat = Concatenate(axis=-1, name='concat')([flatt, input_pose])

    batch_norm = BatchNormalization()(cat)

    # inner product 1
    dense1 = Dense(
        units=100,
        activation='tanh',
        kernel_initializer=glorot_uniform(seed=seed),
        bias_initializer='zeros',
        kernel_regularizer=regularizer,
        name='fc1'
        )(batch_norm)

    dense2 = Dense(
        units=50,
        activation='tanh',
        kernel_initializer=glorot_uniform(seed=seed),
        bias_initializer='zeros',
        kernel_regularizer=regularizer,
        name='fc2'
    )(dense1)

    drop = Dropout(0.3)(dense2)

    # inner product 2
    dense3 = Dense(
        units=2,
        kernel_initializer=glorot_uniform(seed=seed),
        activation='tanh',
        bias_initializer='zeros',
        name='fc3'
        )(drop)

    ### OPTIMIZER ###
    optimizer = SGD(
        lr=learning_rate,
        # decay=0.01,
        nesterov=True,
        momentum=0.9
        )

    ### COMPILE MODEL ###
    model = Model([input_img, input_pose], dense3)
    model.compile(optimizer=optimizer, loss='mse', metrics=[angle_accuracy])
    print(model.summary())
    return model


def create_callbacks(path_to_save, save_period=100):

    ### CALLBACKS ###
    tbCallBack = TensorBoard(
        log_dir=path_to_save,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_grads=False
        )
    checkpoint = ModelCheckpoint(
        Path.join(path_to_save, 'model_{epoch}_{val_loss:.4f}.h5'),
        monitor='val_loss',
        period=save_period
        )
    terminate = TerminateOnNaN()

    return [tbCallBack, checkpoint, terminate]
