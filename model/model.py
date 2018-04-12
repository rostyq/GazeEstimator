from keras.models import load_model
from transform import gaze2Dto3D, pose3Dto2D
from nn import angle_accuracy

scope_dict = {'angle_accuracy': angle_accuracy}

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
