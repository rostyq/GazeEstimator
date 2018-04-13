from keras.models import load_model
from .transform import gaze2Dto3D, pose3Dto2D
from .nn import angle_accuracy
from os import getcwd
from numpy import array, flip

scope_dict = {'angle_accuracy': angle_accuracy}
# '/estimator/model_master.h5'
model = load_model(getcwd() + '/estimator/checkpoints/model_last.h5', custom_objects=scope_dict, compile=True)

def estimate_gaze(eye_image, head_pose, eye='left'):

    # prepare and transform
    mirror_array = array([{'left': 1, 'right': -1}[eye], 1])
    eye_image = eye_image.reshape((-1, 36, 60, 1)) / 255
    eye_image = flip(eye_image, axis=2) if eye is 'right' else eye_image
    head_pose = pose3Dto2D(head_pose.reshape((-1, 3))) * mirror_array

    # predict
    gaze = model.predict([eye_image, head_pose]) * mirror_array

    # transform and return
    return gaze2Dto3D(gaze)
