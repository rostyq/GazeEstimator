from keras.models import load_model
from .transform import gaze2Dto3D
from .transform import pose3Dto2D
from .nn import angle_accuracy
from os import getcwd
from numpy import reshape


def init_model(path):
    return load_model(
        path,
        custom_objects={'angle_accuracy': angle_accuracy},
        compile=True
    )

def prepare(eye_image, head_pose):
    """
    Reshape input data for tensorflow model.

    Parameters:
    -----------
    eye_image: Image 36x60, array-like with type uint8
    head_pose: Vector, ndarray[float, float, float]

    Returns:
    --------
    eye_image_tensor: Image 1x36x60x1, array-like with type float32
    head_pose_tensor: Vector, ndarray[[float, float, float]]
    """
    return [reshape(eye_image, (-1, 36, 60, 1)) / 255, pose3Dto2D(reshape(head_pose, (-1, 3)))]

def postprocess(predicted_gaze_tensor):
    """
    Reshape  and transform output from tensorflow model 2D to 3D.

    Parameters:
    -----------
    predicted_gaze_tensor: ndarray[[float, float]]

    Returns:
    --------
    predicted_gaze_vector: ndarray[float, float, float]
    """
    return reshape(gaze2Dto3D(predicted_gaze_tensor), (3,))

def estimate_gaze(eye_image, head_pose, estimator_model):
    """
    Predict gaze vector.

    Parameters:
    -----------
    eye_image: Image 36x60, array-like with type uint8
    head_pose: Vector, ndarray[float, float, float]

    Returns:
    --------
    gaze_vector: ndarray[float, float, float]
    """
    return postprocess(estimator_model.predict(prepare(eye_image, head_pose)))
