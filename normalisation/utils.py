from math import sqrt
import numpy as np

# x = qx / sqrt(1-qw*qw)
# y = qy / sqrt(1-qw*qw)
# z = qz / sqrt(1-qw*qw)

def quaternion_to_angle_axis(quaternion):
    t = sqrt(1-quaternion['W']*quaternion['W'])
    if t != 0:
        x = quaternion['X'] / t
        y = quaternion['Y'] / t
        z = quaternion['Z'] / t
        return np.array([[x], [y], [z]])
    else:
        return np.zeros((3,1))


def extract_normalized_eye_pictures(recognitor, image):

    recognitor.set_image(image)
    recognitor.decect_faces()

    if len(recognitor.rects) > 0:
        recognitor.detect_landmarks()
        recognitor.detect_faces_poses()
        recognitor.detect_gazes()
        return recognitor.produce_normalized_eye_frames()

    return None
