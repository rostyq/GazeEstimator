from .normalisation import DlibImageNormalizer

from .estimator import estimate_gaze
from .estimator import init_model

from .cv2window import ExperimentWindow
from .cv2window import ispressed

from cv2 import VideoCapture
from numpy import array
from numpy.random import randint


def run_coarse_experiment(average_distance, screen_diagonal, path_to_estimator,
                          capture_target=0, test_ticks_treshold=10, button_code_to_stop=27):
    """
    Run coarse experiment for checking predictions of gaze estimator model.

    Parameters:
    -----------
    average_distance: Average distance from display to user's left eye.
    screen_diagonal: Screen diagonal in inches.
    path_to_estimator: path to gaze estimator model.
    capture_target: Parameters passed to opencv capture function. Default: 0 -- usually webcam.
    test_ticks_treshold: Every n-th tick change test circle. Default 10.
    button_code_to_stop: Code of button on keyboard which will stop the experiment. Default 27 -- ESCAPE.

    Returns:
    --------
    None

    """

    # helping functions
    def create_random_coordinates(screen_resolution):
        return randint(0, screen_resolution[0], size=1), randint(0, screen_resolution[1], size=1)

    def calc_pog_coordinates(distance, gaze_vector, screen_resolution, screen_inches):
        return tuple(map(
            int, [
                ((distance * gaze_vector[0] / gaze_vector[2]) / (0.5 * screen_inches[0]) + 1) * (0.5*screen_resolution[0]),
                (distance * gaze_vector[1] / (-gaze_vector[2])) / screen_inches[1] * screen_resolution[1]
                ]))

    # prepare working objects
    capture = VideoCapture(capture_target)
    window = ExperimentWindow(__name__, screen_diagonal)
    gaze_estimator = init_model(path_to_estimator)
    face_recognitor = DlibImageNormalizer(capture.read()[1].shape)

    # prepare experiment window
    window.open()
    window.draw_test_circle(create_random_coordinates(window.screen_resolution))
    window.set_frame_as_background()

    # dummy head pose
    dummy_head_pose = array([0]*3)

    # run cycles of experiment
    ticks = 0
    while not ispressed(button_code_to_stop):

        if not ticks % test_ticks_treshold:
            window.reset_frame()
            window.draw_test_circle(create_random_coordinates(window.screen_resolution))
            window.set_frame_as_background()
        ticks += 1

        try:
            left_eye_img = face_recognitor.fit_transform(capture.read()[1])[0][1]
            left_gaze_vector = estimate_gaze(left_eye_img,
                                             dummy_head_pose,
                                             gaze_estimator)
            pog_coordinates = calc_pog_coordinates(average_distance,
                                                   left_gaze_vector,
                                                   window.screen_resolution,
                                                   window.screen_inches)
            # checking data on the screen
            window.put_text(f'left gaze: {left_gaze_vector}',
                            (0, 40))
            window.put_text(f'coordinates: {pog_coordinates[0]}, {pog_coordinates[1]}',
                            (0, 80))
            window.put_image(left_eye_img.reshape((36, 60)),
                             (-37, -61))
            window.draw_pog(pog_coordinates)
        except TypeError:
            window.put_text('left gaze: No face - no eye :(',
                            (0, 40))
            continue
        finally:
            window.show()

    capture.release()
    window.close()

import json
import pickle
from os import listdir
from os import path

import cv2

from app.calibration import Calibration
from app.normalisation import StandNormalizazer, Face
from app.normalisation.utils import draw_eye_centers, POG_to_kinect_space
from config import *


class Experiment:
    def __init__(self, path_to_frames, path_to_face_points, path_to_face_poses, norm_camera='basler'):
        self.normalization_camera = Calibration(None)

        self.path_to_frames = path_to_frames
        self.frames = listdir(self.path_to_frames)
        self.path_to_face_points = path_to_face_points
        self.face_points = listdir(self.path_to_face_points)
        self.path_to_face_poses = path_to_face_poses
        self.face_poses = listdir(self.path_to_face_poses)

        self.normalization_camera.camera_matrix = cameras[norm_camera]['matrix']
        self.normalization_camera.distortion_vector = cameras[norm_camera]['distortion']
        self.normalization_camera.rotation_vector = cameras[norm_camera]['rotation_vector']

        self.dataset_size = min([len(self.frames), len(self.face_points), len(self.face_poses)])

        frame_example = cv2.imread(path.join(self.path_to_frames, self.frames[0]))
        self.normalizer = StandNormalizazer(frame_example.shape, calibration=self.normalization_camera,
                                            rotation_vector=cameras[norm_camera]['rotation_vector'],
                                            translation_vector=cameras[norm_camera]['translation_vector'])

    def generate_dataset(self, indices):
        """
        Dataset generator
        :param indices: indices of samples
        :return: yield tuple(frame, face_points, faces_rotations)
        """
        for current_sample in indices:
            frame = cv2.imread(path.join(self.path_to_frames, self.frames[current_sample]))
            with open(path.join(self.path_to_face_points, self.face_points[current_sample])) as face_points:
                face_points = json.load(face_points)
            with open(path.join(self.path_to_face_poses, self.face_poses[current_sample])) as face_poses:
                face_poses = json.load(face_poses)
                faces_rotations = [face_pose['FaceRotationQuaternion'] for face_pose in face_poses]
            yield frame, face_points, faces_rotations

    def validate_calibration(self, frame_indices):
        """
        Performs validation by projection eye landmarks on frames. Shows all frames
        :param frame_indices: indices of frames to project landmarks on
        :return: self
        """
        for k, (frame, face_points, faces_rotations) in enumerate(self.generate_dataset(frame_indices)):
            eyes = self.normalizer.fit_transform(frame, face_points, faces_rotations)
            if eyes:
                draw_eye_centers(self.normalizer)
                cv2.imshow(__name__ + str(k), cv2.resize(self.normalizer.frame,
                                                         (int(frame.shape[1]/2), int(frame.shape[0]/2))))
                cv2.imshow('kinect - left' + str(k), eyes[0][0])
                cv2.imshow('kinect - right' + str(k), eyes[0][1])
        cv2.waitKey(0)
        return self

    def create_learning_dataset(self, filename):
        """
        Creates pickle file with learning dataset
        :param filename: name of file, where to save
        :return: self
        """
        learning_dataset = []
        for frame, face_points, faces_rotations in self.generate_dataset(range(self.dataset_size)):
            self.normalizer.fit_transform(frame, face_points, faces_rotations)
            #real_gaze_vector
            learning_dataset.append(self.normalizer.faces)
        pickle.dump(learning_dataset, file=open(filename, mode='wb'))
        return self


if __name__ == '__main__':
    # path_to_frames = [path.join(path.dirname(__file__),
    #                              r'../../11_04_18/1523433382/DataSource/cam_0/ColorFrame'),
    #                    path.join(path.dirname(__file__),
    #                              r'../..\20_04_2018\20_04_18\1524238461\DataSource/cam_1/ColorFrame'),
    #                    ]
    # path_to_face_points = [path.join(path.dirname(__file__),
    #                                   r'../../11_04_18/1523433382/DataSource/cam_0/FacePoints'),
    #                         path.join(path.dirname(__file__),
    #                                   r'../..\20_04_2018\20_04_18\1524238461/DataSource/cam_1/FacePoints'),
    #                         ]

    path_to_frames = [path.join(path.dirname(__file__),
                                 r'../../11_04_18/1523433382/DataSource/cam_1/InfraredFrame'),
                       path.join(path.dirname(__file__),
                                 r'../../20_04_2018/20_04_18/1524238461/DataSource/cam_2/InfraredFrame')]
    path_to_face_points = [path.join(path.dirname(__file__),
                                      r'../../11_04_18/1523433382/DataSource/cam_0/FacePoints'),
                            path.join(path.dirname(__file__),
                                      r'../20_04_2018/20_04_18/1524238461/DataSource/cam_1/FacePoints')]
    path_to_face_poses = [path.join(path.dirname(__file__),
                                     r'../../11_04_18/1523433382/DataSource/cam_0/FaceFrame'),
                           path.join(path.dirname(__file__),
                                     r'../../20_04_2018/20_04_18/1524238461/DataSource/cam_1/FaceFrame')]

    experiment = Experiment(path_to_frames=path_to_frames[0],
                            path_to_face_points=path_to_face_points[0],
                            path_to_face_poses=path_to_face_poses[0])

    # validating camera calibration on random samples
    experiment.validate_calibration([0, 100, 110])
    experiment.create_learning_dataset('dataset.pickle')

    pickle.load(open('dataset.pickle', mode='rb'))




