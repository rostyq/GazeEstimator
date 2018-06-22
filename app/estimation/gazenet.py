from keras.models import load_model
from .transform import gaze2Dto3D
from .transform import gaze3Dto2D
from .transform import pose3Dto2D
from .nn import angle_accuracy
from .nn import create_model
from .nn import create_callbacks
from numpy import reshape
import os


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
    result = [reshape(eye_image, (-1, 72, 120, 1)) / 255, gaze3Dto2D(reshape(head_pose, (-1, 3)))]
    print(gaze3Dto2D(reshape(head_pose, (-1, 3))))
    return result


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
    return reshape(gaze2Dto3D(predicted_gaze_tensor), (1, 3))


class GazeNet:

    def __init__(self):
        self.model = None

    def init(self, path_to_model):
        self.model = load_model(path_to_model,
                                custom_objects={'angle_accuracy': angle_accuracy},
                                compile=False)
        return self

    def train(self, path_to_save, create_new=False, create_dict=None, sess_name=None, save_period=100, **kwargs):
        if create_new:
            if create_dict is None:
                create_dict = {}
            self.model = create_model(**create_dict)

        path_to_save = os.path.join(path_to_save, sess_name)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        callbacks = create_callbacks(path_to_save=path_to_save, save_period=save_period)

        try:
            history = self.model.fit(callbacks=callbacks, **kwargs)
        finally:
            self.model.save(os.path.join(path_to_save, 'model_last.h5'))
        return history

    def score(self, input_data, gazes, batch_size):
        return self.model.evaluate(prepare(*input_data), gaze3Dto2D(gazes), batch_size=batch_size)

    def estimate_gaze(self, eye_image, head_pose):
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
        return postprocess(self.model.predict(prepare(eye_image, head_pose)))
