from config import *
from app import construct_scene_objects
import json
from app.parser import ExperimentParser
from app.frame import Frame
from app.actor import Actor
from app import ispressed
import cv2
import numpy as np


if __name__ == "__main__":

    with open('./extrinsic_params.json', 'r') as f:
        extrinsic_params = json.load(f)

    scene = construct_scene_objects(origin_name='ir',
                                    intrinsic_params=INTRINSIC_PARAMS,
                                    extrinsic_params=extrinsic_params)

    origin = scene['origin']
    cams = scene['cameras']
    screens = scene['screens']

    root_path = '/Users/rostyslav.db/Documents/beehiveor/datasets/1526380341/DataSource'
    path_to_frames = root_path + r'/cam_0'
    path_to_face_points = root_path + r'/cam_7'
    path_to_face_poses = root_path + r'/cam_6'
    path_to_gazes = root_path + r'/cam_9'
    parser = ExperimentParser(path_to_frames=path_to_frames,
                              path_to_face_points=path_to_face_points,
                              path_to_face_poses=path_to_face_poses,
                              path_to_gazes=path_to_gazes)

    image, face_points, faces_rotations, gaze_point = parser.read_sample(130)

    frame = Frame(cams['web_cam'], image)

    print('Frame shape', frame.image.shape)

    actor = Actor('tester', origin=origin, frame=frame)
    actor.set_landmarks(face_points)
    actor.set_rotation(faces_rotations[0])
    actor.set_translation('nose')
    print(actor.rotation, actor.translation)

    # eye, _ = frame.extract_eyes_from_actor(actor, shifts=(10, 20))
    # print(actor.landmarks['nose'])
    # frame.project_vectors(actor.landmarks['nose'].reshape(1, 3), radius=3)
    # frame.project_vectors(actor.landmarks['ReyeI'].reshape(1, 3), radius=3)
    # frame.project_vectors(actor.landmarks['ReyeO'].reshape(1, 3), radius=3)
    # frame.draw_points(np.array([[100, 100]]))

    # gp_vector = screens['screen'].point_to_origin(*gaze_point).reshape(1, 3)

    corners = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]

    gp_vector = screens['screen'].point_to_origin(0.0, 0.0).reshape(-1, 3)
    gp_vectors = np.array([screens['screen'].point_to_origin(*corner).reshape(3,) for corner in corners], subok=True)
    frame.project_vectors(gp_vectors)
    while not ispressed(27):
        cv2.imshow(__name__+'1', frame.image[::-1, :])
    cv2.destroyWindow(__name__+'1')
    cv2.destroyAllWindows()
