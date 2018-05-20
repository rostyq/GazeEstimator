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


class GazeNet:

    def __init__(self):
        self.model = None

    def init(self, path_to_model):
        self.model = load_model(path_to_model,
                                custom_objects={'angle_accuracy': angle_accuracy},
                                compile=True)
        return self

    def train(self, input_train, gazes_train, epochs, batch_size, create_new, path_to_save, validation_data=None, **kwargs):
        if create_new:
            self.model = create_model(**kwargs)
        callbacks = create_callbacks(path_to_save)

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        self.model.fit(x=input_train,
                       y=gazes_train,
                       batch_size=batch_size,
                       verbose=1,
                       epochs=epochs,
                       validation_data=validation_data,
                       callbacks=callbacks)

        self.model.save(f'{path_to_save}/model_last.h5')
        return self

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
