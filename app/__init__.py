from os import path as Path
import json
import os.path
# from app.estimation import GazeNet
from app.device.screen import Screen
from app.device.camera import Camera
from app.parser import SessionReader
from app.estimation.persondetector import PersonDetector
from app.frame import Frame
from app.actor import Person
from app import *
from tqdm import tqdm

import cv2


class Scene:
    def __init__(self, origin_name, intrinsic_params, extrinsic_params):
        params = {key: value for key, value in intrinsic_params['CAMERAS'].items() if key != origin_name}
        self.origin = Camera(name=origin_name, cam_dict=intrinsic_params['CAMERAS'][origin_name])

        self.cams = {
            name: Camera(
                name=name,
                cam_dict=cam_data,
                extrinsic_matrix=(extrinsic_params[f'{name}_{self.origin.name}']),
                origin=self.origin
            ) for name, cam_data in params.items()
        }
        self.cams[origin_name] = self.origin

        self.screens = {
            name: Screen(
                name=name,
                screen_dict=screen_data,
                extrinsic_matrix=extrinsic_params[f'{name}_{self.origin.name}'],
                origin=self.origin
            ) for name, screen_data in intrinsic_params['SCREENS'].items()
        }

    def to_dict(self):
        return {'screens':  {name: screen.to_dict() for name, screen in self.screens.items()},
                'cams': {name: cam.to_dict() for name, cam in self.cams.items()}}


def create_learning_dataset(save_path, sess_reader, face_detector, scene, indices=None, markers=None):
    save_path = Path.join(save_path, 'normalized_data', sess_reader.session_code)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print(save_path)

    learning_data = {'dataset': [], 'scene': scene.to_dict()}
    for ((frames, data), index), marker in zip(sess_reader.snapshots_iterate(indices=indices, progress_bar=True), markers):
        if data['face_points']:
            frame_basler = frames['basler']
            # actor_kinect = Person('kinect', origin=scene.origin)
            # actor_kinect.set_kinect_landmarks3d(data['face_points'])
            actors_basler = face_detector.detect_persons(frame_basler, scene.origin)
            if len(actors_basler) == 0:
                continue
            actor_basler = actors_basler[0]
            # actor_basler.set_landmarks3d_gazes(data['gazes'], scene.screens['wall'])
            # real_gazes = {'left': gaze,
                          # 'right': gaze}
            actor_basler.set_gazes_to_mark(marker)
            # actor_kinect.set_landmarks3d_gazes(gazes, scene.screens['wall'])

            right_eye_frame, left_eye_frame = frame_basler.extract_eyes_from_person(actor_basler,
                                                                                    resolution=(120, 72),
                                                                                    equalize_hist=True,
                                                                                    to_grayscale=True,
                                                                                    remove_specularity=True)

            cv2.imwrite(Path.join(save_path, f'{index}_left.png'), left_eye_frame)
            cv2.imwrite(Path.join(save_path, f'{index}_right.png'), right_eye_frame)

            learning_data['dataset'].append(actor_basler.to_learning_dataset(f'{index}_left.png',
                                                                              f'{index}_right.png',
                                                                              scene.cams['basler']))

    with open(Path.join(save_path, 'normalized_dataset.json'), mode='w') as outfile:
        json.dump(learning_data, fp=outfile, indent=2)
    print(f"Dataset saved to {save_path}. Number of useful snapshots: {len(learning_data['dataset'])}")
