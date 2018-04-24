import json
from os import path, listdir

from app.calibration import Calibration
from app.normalisation import StandNormalizazer, DlibImageNormalizer
from app.normalisation.utils import *


def run_experiment():
    camera = Calibration(board_shape=(6, 4), path_to_dataset=path.join(path.dirname(__file__),
                                                                       r'../app/calibration/stand_dataset/kinect_victor'))
    # camera.calibrate(method='dataset')
    # camera.load_metadata()

    colored_camera = {'matrix': np.array([[1.0511, -0.0024, 0.6492],
                                          [0., 1.0522, 0.4767],
                                          [0., 0., 0.001]]) * 1.0e+03,
                      'distortion': np.zeros(4),
                      'rvec': np.zeros((3, 1)),
                      'tvec': np.array([[0.05], [0.], [0.]]),
                      'path_to_frames': [path.join(path.dirname(__file__),
                                                   r'../../11_04_18/1523433382/DataSource/cam_0/ColorFrame'),
                                         path.join(path.dirname(__file__),
                                                   r'../..\20_04_2018\20_04_18\1524238461\DataSource/cam_1/ColorFrame')],
                      'path_to_face_points': [path.join(path.dirname(__file__),
                                                        r'../../11_04_18/1523433382/DataSource/cam_0/FacePoints'),
                                              path.join(path.dirname(__file__),
                                                        r'../..\20_04_2018\20_04_18\1524238461/DataSource/cam_1/FacePoints')]}

    basler = {'matrix': np.array([[2.6155, -0.0035, 0.6576],
                                  [0., 2.6178, 0.4682],
                                  [0., 0., 0.001]]) * 1.0e+03,
                      'distortion': np.array([-0.5195, 0.3594, -0.0022, -0.0004]),
                      'rvec': np.array([[-0.075], [0.005], [0.]]),
                      'tvec': np.array([[0.137], [0.044], [0.]]),
                      'path_to_frames': [path.join(path.dirname(__file__),
                                                   r'../../11_04_18/1523433382/DataSource/cam_1/InfraredFrame'),
                                         path.join(path.dirname(__file__),
                                                   r'../../20_04_2018/20_04_18/1524238461/DataSource/cam_2/InfraredFrame')],
                      'path_to_face_points': [path.join(path.dirname(__file__),
                                                r'../../11_04_18/1523433382/DataSource/cam_0/FacePoints'),
                                              path.join(path.dirname(__file__),
                                                r'../../20_04_2018/20_04_18/1524238461/DataSource/cam_1/FacePoints')],
                      'path_to_face_poses': [path.join(path.dirname(__file__),
                                                r'../../11_04_18/1523433382/DataSource/cam_0/FaceFrame'),
                                             path.join(path.dirname(__file__),
                                                       r'../../20_04_2018/20_04_18/1524238461/DataSource/cam_1/FaceFrame')]}

    current_camera = basler
    dataset = 0
    path_to_frames = current_camera['path_to_frames'][dataset]
    path_to_face_points = current_camera['path_to_face_points'][dataset]
    path_to_face_poses = current_camera['path_to_face_poses'][dataset]
    camera.matrix = current_camera['matrix']
    camera.distortion = current_camera['distortion']

    i = 0
    for frame, face_points, face_poses in \
            zip(listdir(path_to_frames)[-5:], listdir(path_to_face_points)[-1:], listdir(path_to_face_poses)[-1:]):
        frame = path.join(path_to_frames, frame)
        image = cv2.imread(frame)

        face_points = open(path.join(path_to_face_points, face_points)).read()
        face_points = json.loads(face_points)

        face_poses = open(path.join(path_to_face_poses, face_poses)).read()
        face_poses = json.loads(face_poses)
        faces_rotation = [face_pose['FaceRotationQuaternion'] for face_pose in face_poses]

        dlib_normaliser = DlibImageNormalizer(image.shape, calibration=camera)
        normalizer = StandNormalizazer(image.shape, calibration=camera,
                                       rvec=current_camera['rvec'], tvec=current_camera['tvec'])

        eyes_dlib = dlib_normaliser.fit_transform(image)
        eyes = normalizer.fit_transform(image, face_points, faces_rotation)
        print(normalizer.faces[0].rvec)

        # if eyes_dlib:
        #     cv2.imshow('dlib - left', eyes_dlib[0][0])
        #     cv2.imshow('dlib - right', eyes_dlib[0][1])
        if eyes:
            cv2.imshow(__name__, normalizer.frame)
            cv2.imshow('kinect - left', eyes[0])
            cv2.imshow('kinect - right', eyes[1])
        #
        cv2.waitKey(0)
        # if eyes:
        #     print(i, frame)
        #     cv2.imwrite(f'result/L{i}.png', eyes[0])
        #     cv2.imwrite(f'result/R{i}.png', eyes[1])
        # i += 1

if __name__ == '__main__':
    run_experiment()
