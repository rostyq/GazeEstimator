
# path to gaze estimator model
PATH_TO_ESTIMATOR = './app/bin/estimator.h5'

# default config for coarse experiment
ESCAPE = 27
DEFAULT_WEBCAM_CAPTURE_TARGET = 0
TEST_TICKS_TRESHOLD = 10
DEFAULT_AVERAGE_DISTANCE = 12

# screen parameters
screen_diagonal = 13.3

CAMERAS_PARAMETERS = {
    'colored_camera': {'matrix': [[1051.7, 0.,     957.6],
                                  [0.,     1051.7, 537.6],
                                  [0.,     0.,     1.   ]],
                       'distortion': [-0.0026, 0.1780, 0.0011, 0.0003],
                       'rotation_vector': [[0.00012], [-0.00052], [0.00035]],
                       'translation_vector': [[0.0517095], [0.0012225], [0.0013196]]},

    'basler': {'matrix': [[1896.6, 0.,     654.8],
                          [0.,     1897.9, 461.5],
                          [0.,     0.,     1.   ]],
               'distortion': [-0.6819, 0.3729, -0.0021, -0.0005],
               'rotation_vector': [[0.0497], [0.0409], [0.0621]],
               'translation_vector': [[0.1359533], [0.0489501], [0.0100126]]},

    'ir-camera': {'matrix': [[363.6643, 0.,       257.1285],
                             [0.,       363.7357, 208.6171],
                             [0.,       0.,       1.]],
                  'distortion': [0.1014, -0.2622, 0.0017, -0.0019],
                  'rotation_vector': [[0.], [0.], [0.]],
                  'translation_vector': [[0.], [0.], [0.]]}
}

SCREEN_PARAMETERS = {'screen_size': (1920, 1080),
                     'diagonal_in_meters': 23 * 0.0254,  # 23 inches in meters
                     'rotation_vector': [[0.3, -0.14, -0.04]],
                     'translation_vector': [[0.38], [-0.1], [-0.75]]}

