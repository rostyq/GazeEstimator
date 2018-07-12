from json import load

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
                [3793.8, 0, 656.5],
                [0, 3795.0, 464.3],
                [0, 0, 1.0]
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
                [2265.4 * 0.625,   0,   1009.4 * 0.625],
                [0,     2268.7 * 0.625, 811.5 * 0.625],
                [0,     0,      1]
            ],
            'distortion': [-0.0276, 0.1141, 0.0000871, -0.0002941],
        }
    },
    'SCREENS': {
        'screen': {
            'resolution': (1080, 1920),
            'diagonal': 23 * 0.0254,  # 23 inches in meters
         },
        'wall': {
            'resolution': (1080, 1920),
            'diagonal': 1.8537,
        },
    }
}

GAZEPOINT_MARKERS = [
    (0.1, 0.1),
    (0.5, 0.1),
    (0.9, 0.1),
    (0.9, 0.5),
    (0.5, 0.5),
    (0.1, 0.5),
    (0.1, 0.9),
    (0.5, 0.9),
    (0.9, 0.9)
]

ORIGIN_CAM = 'ir'

DATASET_PATH = r''

PATH_TO_ESTIMATOR = './app/bin/estimator.h5'  # './checkpoints/model_200_0.0028.h5'  # './app/bin/estimator.h5'
PATH_TO_FACE_POINTS = './app/bin/face_landmarks.dat'
PATH_TO_FACE_MODEL = './app/bin/face_points_tutorial.mat'
PATH_TO_HAARCASCADE_MODEL = './app/bin/haarcascade_frontalface_default.xml'
PATH_TO_EXTRINSIC_PARAMS = './extrinsic_params.json'

ACTOR_DETECTOR = {
    'path_to_face_model': PATH_TO_FACE_MODEL,
    'path_to_face_points': PATH_TO_FACE_POINTS,
    'path_to_hc_model': PATH_TO_HAARCASCADE_MODEL,
    'factor': 1,
    'scale': 1.3,
    'minNeighbors': 5,
    'chin_nose_distance': 0.065
}

DATASET_PARSER = {
    'images': 'dataset/{index}/eyes/{eye}/image',
    'poses': 'dataset/{index}/rotation_norm',
    'gazes': 'dataset/{index}/eyes/{eye}/gaze_norm'
}

with open(PATH_TO_EXTRINSIC_PARAMS) as f:
    EXTRINSIC_PARAMS = load(f)

