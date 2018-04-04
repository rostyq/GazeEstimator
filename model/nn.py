from keras.layers import Input, Conv2D, MaxPool2D, Dense, Concatenate, Flatten, Dropout, Lambda
from keras.initializers import RandomNormal, glorot_normal, glorot_uniform
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf

def create_model():

    # functions
    def calc_angle(vector1, vector2):

        def to_vector(array):
            x = (-1) * tf.cos(array[:, 0]) * tf.sin(array[:, 1])
            y = (-1) * tf.sin(array[:, 0])
            z = (-1) * tf.cos(array[:, 0]) * tf.cos(array[:, 1])

            return tf.stack((x, y, z), axis=1)

        def unit_vector(array):
            return tf.divide(array, tf.norm(array, axis=1, keep_dims=True))

        unit_v1, unit_v2 = unit_vector(to_vector(vector1)), unit_vector(to_vector(vector2))
        angle_value = tf.matmul(unit_v1, unit_v2, transpose_b=True)[:, 0]
        # return angle_value
        return tf.clip_by_value(angle_value, -1.0, 1.0)


    def angle_loss(target, predicted):
        return tf.reduce_mean(1.0 - calc_angle(target, predicted))
    
    def loss(target, predicted):
        return tf.reduce_mean(tf.reduce_sum(tf.square(target - predicted), axis=1))

    def angle_accuracy(target, predicted):
        return tf.reduce_mean(tf.acos(calc_angle(target, predicted)) * 180 / 3.14159265)


    # input
    input_img = Input(shape=(36, 60, 1), name='InputNormalizedImage')
    input_pose = Input(shape=(2,), name='InputHeadPose')

    # convolutional
    conv1 = Conv2D(filters=20, activation='relu',
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
    conv2 = Conv2D(filters=50, activation='relu',
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
                   kernel_initializer=glorot_uniform(seed=42),
                   bias_initializer='zeros',
                   name='ip1'
                   )(flatt)

    # concatanate with head pose
    cat = Concatenate(axis=-1, name='concat')([dense1, input_pose])

    #dropout = Dropout(rate=0.1)(cat)

    # inner product 2
    dense2 = Dense(units=2,
                   kernel_initializer=glorot_uniform(seed=42),
                   bias_initializer='zeros',
                   name='ip2'
                   )(cat)

    ### OPTIMIZER ###
    optimizer = SGD(lr=1e-1)

    ### COMPILE MODEL ###
    model = Model([input_img, input_pose], dense2)
    model.compile(optimizer=optimizer, loss=loss, metrics=[angle_accuracy])
    return model

def create_callbacks():
    ### CALLBACKS ###
    tbCallBack = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True)
    checkpoint = ModelCheckpoint('./checkpoints/', monitor='val_loss', period=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=1e-4, patience=5, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10, verbose=1)
    terminate = TerminateOnNaN()
    return [tbCallBack, checkpoint, reduce_lr, earlystop, terminate]