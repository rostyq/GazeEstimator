INTRINSIC_PARAMS = {
    'CAMERAS': {
        'color': {
            'matrix': [[1051.7, 0.,     957.6],
                       [0.,     1051.7, 537.6],
                       [0.,     0.,     1.   ]],
            'distortion': [-0.0026, 0.1780, 0.0011, 0.0003],
        },
        'basler': {
            'matrix': [[1896.6, 0.,     654.8],
                       [0.,     1897.9, 461.5],
                       [0.,     0.,     1.   ]],
            'distortion': [-0.6819, 0.3729, -0.0021, -0.0005],
        },
        'ir': {
            'matrix': [[363.6643, 0.,       257.1285],
                       [0.,       363.7357, 208.6171],
                       [0.,       0.,       1.]],
            'distortion': [0.1014, -0.2622, 0.0017, -0.0019],
        },
        'web_cam': {
            'matrix': [[363.6643, 0., 257.1285],
                       [0., 363.7357, 208.6171],
                       [0., 0., 1.]],
            'distortion': [0.1014, -0.2622, 0.0017, -0.0019],
        }
    },
    'SCREENS': {
        'wall': {
            'resolution': (1920, 1080),
            'diagonal': 23 * 0.0254,  # 23 inches in meters
         },
        'screen': {
            'resolution': (1920, 1080),
            'diagonal': 3.0,  # 23 inches in meters
        },
    }
}

CAM_DIRS = {
    'color': 'cam_1',
    'basler': 'cam_8',
    'web_cam': 'cam_0',
    'ir': 'cam_2'
}

DATA_DIRS = {
    'face_poses': 'cam_6',
    'gazes': 'cam_9',
    'face_points': 'cam_7'
}

ORIGIN_CAM = 'ir'

DATASET_PATH = r'C:\Users\Valik\Documents\GitHub\dataroot\RETNNA\BAS\1525974053\DataSource'

PATH_TO_ESTIMATOR = './app/bin/estimator.h5'
PATH_TO_FACE_MODEL = './app/bin/face_landmarks.dat'
