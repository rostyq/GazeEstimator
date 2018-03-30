from keras.models import load_model
from model.utils import gaze2Dto3D, gaze3Dto2D, pose3Dto2D, angle_loss, angle_accuracy

scope_dict = {'angle_loss': angle_loss, 'angle_accuracy': angle_accuracy}

model = load_model('./model_epoch1.h5', custom_objects=scope_dict, compile=True)


def estimate_gaze(eye_image, head_pose):
    eye_image = eye_image.reshape((-1, 36, 60, 1))
    head_pose = head_pose.reshape((-1, 3))
    head_pose = pose3Dto2D(head_pose)
    gaze = model.predict([eye_image, head_pose])
    return gaze2Dto3D(gaze)
