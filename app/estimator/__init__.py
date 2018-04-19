from keras.models import load_model
from .transform import gaze2Dto3D, pose3Dto2D
from .nn import angle_accuracy
from os import getcwd


model = load_model(
    f'{getcwd()}/app/binaries/estimator.h5',
    custom_objects={'angle_accuracy': angle_accuracy},
    compile=True
    )


def estimate_gaze(eye_image, head_pose):
    return gaze2Dto3D(
        model.predict(
            [
                eye_image.reshape((-1, 36, 60, 1)) / 255,
                pose3Dto2D(head_pose.reshape((-1, 3)))
                ]
            )
        )
