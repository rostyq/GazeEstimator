from keras.models import load_model
from .parser import DatasetParser
from .transform import gaze2Dto3D
from .transform import gaze3Dto2D
from .transform import pose3Dto2D
from .nn import angle_accuracy
from .nn import create_model
from .nn import create_callbacks
from os import getcwd
from numpy import reshape
from numpy import array


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


def read_data(path_to_dataset, json_name, parser_params, eye):
    parser = DatasetParser(**parser_params)
    with open(path_to_dataset+json_name, 'r') as file:
        parser.fit(file, path_to_dataset)
    images = array(parser.get_images_array(eye))
    poses = array(parser.get_poses_array())
    gazes = array(parser.get_gazes_array(eye))
    return prepare(images, poses), gaze3Dto2D(gazes)


class GazeNet:

    def __init__(self):
        self.model = None

    def init(self, path_to_model):
        self.model = load_model(path_to_model,
                                custom_objects={'angle_accuracy': angle_accuracy},
                                compile=True)
        return self

    def train(self, path_to_dataset, path_to_save, json_name, parser_params, create_new, eye, epochs, batch_size, **kwargs):
        if create_new:
            self.model = create_model(**kwargs)
        callbacks = create_callbacks()

        input_train, gazes = read_data(path_to_dataset, json_name, parser_params, eye)

        self.model.fit(x=input_train,
                       y=gazes,
                       batch_size=batch_size,
                       verbose=1,
                       epochs=epochs,
                       # validation_data=([test_images, test_poses], test_gazes),
                       callbacks=callbacks)

        self.model.save(path_to_save)
        return self

    def score(self, path_to_dataset, json_name, parser_params, eye, batch_size, **kwargs):
        input_test, gazes = read_data(path_to_dataset, json_name, parser_params, eye)
        return self.model.evaluate(input_test, gazes, batch_size=batch_size)

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
