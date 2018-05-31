INTRINSIC_PARAMS = {
    'CAMERAS': {
        'color': {
            'matrix': [
                [1061.6129, 0,  960.0],
                [0,     1061.6129, 540.0],
                [0,     0,      1]
            ],
            'distortion': [0.038, -0.0388, 0.0010, 0.0004],
        },
        'basler': {
            'matrix': [
                [ 3802.7,        0,         648.0],
                [      0,   3804.0,         486.4],
                [      0,        0,       1.0]
            ],
            'distortion': [-0.3508,   0.5343, -0.0008929,  -0.0004769],
        },
        'ir': {
            'matrix': [
                [365.7,         0,   256],
                [       0,  365.7,   212],
                [       0,         0,   1]
            ],
            'distortion': [-6.510e-3, 1.205e-3, 1.377e-4, 1.589e-4],
        },
        'web_cam': {
            'matrix': [
                [2265.4 * 0.3125,   0,   1009.4 * 0.3125],
                [0,     2268.7 * 0.3125, 811.5 * 0.3125],
                [0,     0,      1]
            ],
            'distortion': [-0.0276, 0.1141, 0.0000871, -0.0002941],
        }
    },
    'SCREENS': {
        # 'screen': {
        #     'resolution': (1920, 1080),
        #     'diagonal': 23 * 0.0254,  # 23 inches in meters
        #  },
        'wall': {
            'resolution': (1080, 1920),
            'diagonal': 1.8537,
        },
    }
}


ORIGIN_CAM = 'ir'

DATASET_PATH = r'C:\Users\Valik\Documents\GitHub\dataroot\RETNNA\BAS\1527250535'

PATH_TO_ESTIMATOR = './app/bin/estimator.h5'
PATH_TO_FACE_MODEL = './app/bin/face_landmarks.dat'
PATH_TO_FACE_POINTS = './app/bin/face_points_tutorial.mat'
PATH_TO_HAARCASCADE_MODEL = './app/bin/haarcascade_frontalface_default.xml'
