import json
import pickle
from cv2 import VideoCapture
from os import listdir, path
from pprint import pprint as print

from numpy import array
from numpy.random import randint

from app.calibration import Calibration
from app.cv2window import ExperimentWindow
from app.cv2window import ispressed
from app.estimator import estimate_gaze
from app.estimator import init_model
from app.normalisation import StandNormalizer, Face, DlibImageNormalizer
from app.normalisation.utils import *
from config import *


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


class Experiment:
    def __init__(self, path_to_frames, path_to_face_points, path_to_face_poses, path_to_gazes, norm_camera='basler'):
        self.normalization_camera = Calibration(None)

        self.path_to_frames = path_to_frames
        self.path_to_face_points = path_to_face_points
        self.path_to_face_poses = path_to_face_poses
        self.path_to_gazes = path_to_gazes
        self.indices = [path.splitext(frame_index)[0] for frame_index in listdir(self.path_to_frames)]
        self.dataset_size = len(self.indices)

        self.normalization_camera.camera_matrix = np.array(CAMERAS_PARAMETERS[norm_camera]['matrix'])
        self.normalization_camera.distortion_vector = np.array(CAMERAS_PARAMETERS[norm_camera]['distortion'])
        self.normalization_camera.rotation_vector = np.array(CAMERAS_PARAMETERS[norm_camera]['rotation_vector'])

        frame_example = cv2.imread(path.join(self.path_to_frames, self.indices[0] + '.png'))
        self.normalizer = StandNormalizer(frame_example.shape, calibration=self.normalization_camera,
                                          rotation_vector=np.array(CAMERAS_PARAMETERS[norm_camera]['rotation_vector']),
                                          translation_vector=np.array(CAMERAS_PARAMETERS[norm_camera]['translation_vector']))

    def generate_dataset(self, indices):
        """
        Dataset generator
        :param indices: indices of samples, according to BRS
        :return: yield tuple(frame, face_points, faces_rotations)
        """
        for current_sample in indices:
            frame_file = path.join(self.path_to_frames, self.indices[current_sample] + '.png')
            face_points_file = path.join(self.path_to_face_points, self.indices[current_sample] + '.txt')
            face_poses_file = path.join(self.path_to_face_poses, self.indices[current_sample] + '.txt')
            gazes_file = path.join(self.path_to_gazes, self.indices[current_sample] + '.txt')

            frame = None
            face_points = None
            faces_rotations = None
            gaze = None

            if path.isfile(frame_file):
                frame = cv2.imread(frame_file)
            if path.isfile(face_points_file):
                with open(face_points_file) as face_points:
                    face_points = json.load(face_points)
            if path.isfile(face_poses_file):
                with open(face_poses_file) as face_poses:
                    face_poses = json.load(face_poses)
                    faces_rotations = [face_pose['FaceRotationQuaternion'] for face_pose in face_poses]
            if path.isfile(gazes_file):
                with open(gazes_file) as gaze:
                    gaze = json.load(gaze)
                    if gaze['REC']['FPOGV'] or True:
                        gaze = POG_to_kinect_space(gaze['REC']['FPOGX'], gaze['REC']['FPOGY'])
            yield frame, face_points, faces_rotations, gaze

    def validate_camera_calibration(self, frame_indices, camera='basler'):
        """
        Performs validation by projection of eye landmarks on frame. Shows all frames
        :param frame_indices: indices of frames to project landmarks on
        :return: self
        """
        for k, (frame, face_points, faces_rotations, gaze) in enumerate(self.generate_dataset(frame_indices)):
            if face_points is not None and faces_rotations is not None:
                eyes = self.normalizer.fit_transform(frame, face_points, faces_rotations, gaze)
                if eyes:
                    draw_eye_landmarks(self.normalizer, self.normalizer.frame, camera)
                    cv2.imshow(__name__ + str(k), cv2.resize(self.normalizer.frame,
                                                             (int(frame.shape[1]), int(frame.shape[0]))))
                    cv2.imshow('kinect - left' + str(k), eyes[0][0])
                    cv2.imshow('kinect - right' + str(k), eyes[0][1])
            else:
                print('Not full sample!')
        cv2.waitKey(0)
        return self

    def validate_screen_calibration(self, frame_indices, camera='basler'):
        """
        Performs validation by projection of screen rectangle on frame. Shows all frames
        :param frame_indices: indices of frames to project rectangle on
        :return: self
        """
        for k, (frame, _, _, _) in enumerate(self.generate_dataset(frame_indices)):
            draw_screen(frame, camera)
            cv2.imshow(__name__ + str(k), cv2.resize(frame,
                                                     (int(frame.shape[1]/2), int(frame.shape[0]/2))))
        cv2.waitKey(0)
        return self


    def validate_Gazepoint_gaze(self, frame_indices, camera='basler'):
        """
        Performs validation by projection of gaze vector. Shows all frames
        :param frame_indices: indices of frames to project gaze vector on
        :return: self
        """
        for k, (frame, face_points, faces_rotations, gaze) in enumerate(self.generate_dataset(frame_indices)):
            if face_points is not None and faces_rotations is not None:
                eyes = self.normalizer.fit_transform(frame, face_points, faces_rotations, gaze)
                if eyes:
                    draw_real_gazes(self.normalizer, frame, camera)
                    cv2.imshow(__name__ + str(k),
                               cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2))))
        cv2.waitKey(0)
        return self


    def create_learning_dataset(self, filename):
        """
        Creates pickle file with learning dataset
        :param filename: name of file, where to save
        :return: self
        """
        learning_dataset = []
        for k, (frame, face_points, faces_rotations, gaze) in enumerate(self.generate_dataset(range(self.dataset_size))):
            if face_points is not None and faces_rotations is not None and gaze is not None:
                self.normalizer.fit_transform(frame, face_points, faces_rotations, gaze)
                learning_dataset.append(self.normalizer.faces)
            else:
                print(f'Not full sample #{k}!')
        pickle.dump(learning_dataset, file=open(filename, mode='wb'))
        return self

    @staticmethod
    def load_learning_dataset(filename):
        with open(filename, mode='rb') as dataset:
            return pickle.load(dataset)


if __name__ == '__main__':

    root_path = path.join(path.dirname(__file__), r'..\..\03_05_18__18-00\1525358247\DataSource')
    camera = 'ir-camera'

    path_to_frames = [root_path + r'\cam_2\InfraredFrame']
    # path_to_frames = [root_path + r'\cam_1\InfraredFrame']
    # path_to_frames = [root_path + r'\cam_1\ColorFrame']
    path_to_face_points = [root_path + r'\cam_1\FacePoints']
    path_to_face_poses = [root_path + r'\cam_1\FaceFrame']
    path_to_gazes = [root_path + r'\cam_3\GazepointData']

    experiment = Experiment(path_to_frames=path_to_frames[0],
                            path_to_face_points=path_to_face_points[0],
                            path_to_face_poses=path_to_face_poses[0],
                            path_to_gazes=path_to_gazes[0],
                            norm_camera=camera)

    print('Dataset size: ' + str(experiment.dataset_size))

    # validating data on random samples
    experiment.validate_camera_calibration([48], camera=camera)
    experiment.validate_screen_calibration([48], camera=camera)
    experiment.validate_Gazepoint_gaze([48], camera=camera)
    # experiment.create_learning_dataset(root_path + r'\normalized_dataset.pickle')




