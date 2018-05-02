import numpy as np

# path to gaze estimator model
PATH_TO_ESTIMATOR = './app/bin/estimator.h5'

# default config for coarse experiment
ESCAPE = 27
DEFAULT_WEBCAM_CAPTURE_TARGET = 0
TEST_TICKS_TRESHOLD = 10
DEFAULT_AVERAGE_DISTANCE = 12

# screen parameters
screen_diagonal = 13.3

cameras = {
    'colored_camera': {'matrix': np.array([[1.0487, 0., 0.9607],
                                           [0., 1.0505, 0.5409],
                                           [0., 0., 0.001]]) * 1.0e+03,
                       'distortion': np.zeros(4),
                       'rotation_vector': np.zeros((3, 1)),
                       'translation_vector': np.array([[0.0322945], [-0.0007767], [-0.0331078]])},

    'basler': {'matrix': np.array([[2.6155, -0.0035, 0.6576],
                                   [0., 2.6178, 0.4682],
                                   [0., 0., 0.001]]) * 1.0e+03,
               'distortion': np.array([-0.5195, 0.3594, -0.0022, -0.0004]),
               'rotation_vector': np.array([[-0.075], [0.005], [0.]]),
               'translation_vector': np.array([[0.137], [0.044], [0.]])}
}

