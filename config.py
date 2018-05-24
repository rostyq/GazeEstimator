INTRINSIC_PARAMS = {
    'CAMERAS': {
        'color': {
            'matrix': [
                [1050.9, 0,  956.0],
                [0,     1050.0, 534.8],
                [0,     0,      1]
            ],
            'distortion': [0.038, -0.0388, 0.0010, 0.0004],
        },
        'basler': {
            'matrix': [
                [ 3802.7,        0,         0],
                [      0,   3804.0,         0],
                [  646.0,    444.4,       1.0]
            ],
            'distortion': [-0.6819, 0.3729, -0.0021, -0.0005],
        },
        'ir': {
            'matrix': [
                [363.1938,         0,   258.8109],
                [       0,  363.0246,   208.8607],
                [       0,         0,          1]
            ],
            'distortion': [0.0799, -0.1877, 0.0010, 0.0002],
        },
        'web_cam': {
            'matrix': [
                [2265.4,   0,   1009.4],
                [0,     2268.7, 811.5],
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
            'resolution': (1920, 1080),
            'diagonal': 185.37,
        },
    }
}

CAM_DIRS = {
    'color': 'cam_3',
    'basler': 'cam_0',
    'web_cam': 'cam_1',
    'ir': 'cam_5'
}

DATA_DIRS = {
    'face_poses': 'cam_8',
    'gazes': 'cam_2',
    'face_points': 'cam_9'
}

ORIGIN_CAM = 'ir'

DATASET_PATH = r'C:\Users\Valik\Documents\GitHub\dataroot\RETNNA\BAS\1527098095\DataSource'

PATH_TO_ESTIMATOR = './app/bin/estimator.h5'
PATH_TO_FACE_MODEL = './app/bin/face_landmarks.dat'
PATH_TO_FACE_POINTS = './app/bin/face_points_tutorial.mat'
PATH_TO_HAARCASCADE_MODEL = './app/bin/haarcascade_frontalface_default.xml'
