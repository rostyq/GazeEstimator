from keras.models import load_model
<<<<<<< HEAD
from utils import gaze2Dto3D, pose3Dto2D
import tensorflow as tf
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

scope_dict = {'angle_accuracy': angle_accuracy}

model = load_model('../model_last.h5', custom_objects=scope_dict, compile=True)
=======
from transform import gaze2Dto3D, pose3Dto2D
from nn import angle_accuracy

scope_dict = {'angle_accuracy': angle_accuracy}
>>>>>>> keras_model

model = load_model('./model_master.h5', custom_objects=scope_dict, compile=True)

def estimate_gaze(eye_image, head_pose):

    # prepare and transform
    eye_image = eye_image.reshape((-1, 36, 60, 1)) / 255
    head_pose = head_pose.reshape((-1, 3))
    head_pose = pose3Dto2D(head_pose)

    # predict
    gaze = model.predict([eye_image, head_pose])

    # transform and return
    return gaze2Dto3D(gaze)
