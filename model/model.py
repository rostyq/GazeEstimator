from keras.models import load_model
from keras.utils import CustomObjectScope
from .utils import gaze2Dto3D, gaze3Dto2D, pose3Dto2D, angle_loss, angle_accuracy

scope_dict = {'angle_loss': angle_loss, 'angle_accuracy': angle_accuracy}

with CustomObjectScope(scope_dict):
    model = load_model('./model_epoch1.h5', custom_objects=None, compile=True)

# TODO estimator class or simple function?
