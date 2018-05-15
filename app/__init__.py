import json
from cv2 import VideoCapture
from cv2 import imread
from cv2 import resize
from cv2 import imshow

from logging import basicConfig
from logging import getLogger
from logging import DEBUG
from logging import StreamHandler

from os import listdir, path
import os

from numpy import array
from numpy import full
from numpy import uint8
from numpy.random import randint

from app.calibration import Calibration

from app.cv2window import ExperimentWindow
from app.cv2window import ispressed

from app.estimator import GazeNet

from app.normalisation import StandNormalizer
from app.normalisation import Face
from app.normalisation import DlibImageNormalizer
from app.normalisation.utils import *
from config import *


def show_charuco(path_to_image, screen_diagonal, square_length_cm, shift):
    """
    This function shows charuco board on the screen.

    Parameters:
    -----------
    path_to_image: Path to image with CHARUCO board in .png format.
    screen_diagonal: Screen diagonal in inches.
    square_length_cm: Length of charuco-square in cm.
    shift tuple(int, int): Shift of charuco from left-upper corner of the screen.

    Returns:
    --------
    None
    """
    charuco_window = ExperimentWindow('CHARUCO', screen_diagonal)
    charuco_image = imread(path_to_image, 0)

    y, x = charuco_image.shape
    square_length = y // 4
    square_length_inch = square_length_cm * 0.3937007874
    dpi = charuco_window.screen_resolution[0] / charuco_window.screen_inches[0]
    ynew = int(y*square_length_inch*dpi/square_length)
    xnew = int(ynew*(x/y))

    charuco_image = resize(charuco_image, (xnew, ynew))
    new_charuco = full(charuco_window.screen_resolution[::-1], 255, dtype=uint8)

    new_charuco[shift[0]:charuco_image.shape[0]+shift[0], shift[1]:charuco_image.shape[1]+shift[1]] = charuco_image
    charuco_window.background = new_charuco

    charuco_window.open()
    while not ispressed(27):
        charuco_window.show()
    charuco_window.close()

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
    gaze_estimator = GazeNet().init(path_to_estimator)
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
            print(left_eye_img.shape)
            left_gaze_vector = gaze_estimator.estimate_gaze(left_eye_img, dummy_head_pose)
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

    basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                filename='log/{}.log'.format(__name__),
                level=DEBUG)
    logger = getLogger(__name__)
    logger.addHandler(StreamHandler())

    def __init__(self, path_to_frames, path_to_face_points, path_to_face_poses, path_to_gazes, norm_camera='basler'):
        self.normalization_camera = Calibration(None)

        self.path_to_frames = path_to_frames
        self.path_to_face_points = path_to_face_points
        self.path_to_face_poses = path_to_face_poses
        self.path_to_gazes = path_to_gazes
        self.indices = [path.splitext(frame_index)[0] for frame_index in listdir(self.path_to_frames)]
        self.dataset_size = len(self.indices)

        self.logger.info('Dataset size: ' + str(self.dataset_size))

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
                    if int(gaze['REC']['FPOGV']) or True:
                        gaze = POG_to_kinect_space(gaze['REC']['FPOGX'], gaze['REC']['FPOGY'])
                    else:
                        gaze = None
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
                    cv2.imshow(__name__ + str(k),
                               cv2.resize(self.normalizer.frame, (int(frame.shape[1]/2), int(frame.shape[0]/2))))
                    cv2.imshow('left' + str(k), eyes[0][0])
                    cv2.imshow('right' + str(k), eyes[0][1])
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


    def create_learning_dataset(self, path, display=False):
        """
        Creates pickle file with learning dataset
        :param path: name of file, where to save
        :return: self
        """
        learning_dataset = []
        if not os.path.exists(path + '/normalized_data'):
            os.mkdir(path + '/normalized_data')

        for k, (frame, face_points, faces_rotations, gaze) in enumerate(self.generate_dataset(range(self.dataset_size))):
            if face_points is not None and faces_rotations is not None and gaze is not None:
                eyes = self.normalizer.fit_transform(frame, face_points, faces_rotations, gaze)

                if display:
                    cv2.imshow('left' + str(k), eyes[0][0])
                    cv2.imshow('right' + str(k), eyes[0][1])
                    cv2.waitKey(1)
                if k % 50 == 0:
                    cv2.destroyAllWindows()

                learning_dataset.append({'faces': [face.__dict__() for face in self.normalizer.faces],
                                         'frames': [f'{k}_left.png', f'{k}_right.png']})
                cv2.imwrite(path + f'/normalized_data/{k}_left.png', eyes[0][0])
                cv2.imwrite(path + f'/normalized_data/{k}_right.png', eyes[0][1])
                self.logger.info(f'Sample#{k} saved')

            else:
                absent_data = 'face_points, ' if face_points is None else '' +\
                              'faces_rotations, ' if faces_rotations is None else '' +\
                              'gaze, ' if gaze is None else ''
                self.logger.warning(f'Not full sample #{k}! Missing {absent_data}')

        self.dump_dataset(path+'/normalized_data/normalized_dataset.json', learning_dataset, camera)
        self.logger.info(f'{len(learning_dataset)}/{self.dataset_size} samples saved to {path}')
        return path+'/normalized_data/normalized_dataset.json'

    @staticmethod
    def dump_dataset(filename, dataset, camera='basler'):
        with open(filename, mode='w') as outfile:
            json.dump({'dataset': dataset,
                       'camera': CAMERAS_PARAMETERS[camera]},
                      fp=outfile, indent=2)

    @staticmethod
    def load_learning_dataset(filename):
        with open(filename, mode='rb') as dataset:
            return json.load(dataset)


if __name__ == '__main__':

    root_path = path.join(path.dirname(__file__), r'..\..\03_05_18__18-00\1525358247\DataSource')
    camera = 'basler'

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

    # validating data on random samples
    experiment.validate_camera_calibration([1], camera=camera)
    experiment.validate_screen_calibration([1], camera=camera)
    experiment.validate_Gazepoint_gaze([1], camera=camera)
    cv2.destroyAllWindows()

    dataset = experiment.create_learning_dataset(root_path, display=True)
    print(Experiment.load_learning_dataset(dataset))
