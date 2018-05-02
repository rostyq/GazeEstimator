import json
from os import listdir

from app.calibration import Calibration
from app.normalisation import StandNormalizazer
from app.normalisation.utils import *
from config import *


def run_experiment():
    camera = Calibration(None)
    current_camera = basler
    dataset = 1
    path_to_frames = current_camera['path_to_frames'][dataset]
    path_to_face_points = current_camera['path_to_face_points'][dataset]
    path_to_face_poses = current_camera['path_to_face_poses'][dataset]
    camera.camera_matrix = current_camera['matrix']
    camera.distortion_vector = current_camera['distortion']

    i = 0
    for frame, face_points, face_poses in \
            zip(listdir(path_to_frames)[10:11],
                listdir(path_to_face_points)[10:11],
                listdir(path_to_face_poses)[10:11]):
        frame = path.join(path_to_frames, frame)
        image = cv2.imread(frame)

        # TODO with construction
        face_points = open(path.join(path_to_face_points, face_points)).read()
        face_points = json.loads(face_points)

        face_poses = open(path.join(path_to_face_poses, face_poses)).read()
        face_poses = json.loads(face_poses)
        faces_rotation = [face_pose['FaceRotationQuaternion'] for face_pose in face_poses]

        normalizer = StandNormalizazer(image.shape, calibration=camera,
                                       rotation_vector=current_camera['rotation_vector'],
                                       translation_vector=current_camera['translation_vector'])

        eyes = normalizer.fit_transform(image, face_points, faces_rotation)

        if eyes:
            draw_eye_centers(normalizer)

            cv2.imshow(__name__, normalizer.frame)
            cv2.imshow('kinect - left', eyes[0][0])
            cv2.imshow('kinect - right', eyes[0][1])
            cv2.waitKey(0)


if __name__ == '__main__':
    run_experiment()
