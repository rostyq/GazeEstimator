from os import path

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.io import loadmat
from sklearn.base import BaseEstimator

from app.normalisation.utils import quaternion_to_angle_axis
from ..calibration import Calibration


class Face:
    def __init__(self, rect):
        self.landmarks = []
        self.rect = rect
        self.rvec = None
        self.tvec = None
        self.gaze = None
        self.left_eye_center = None
        self.right_eye_center = None
        self.real_right_gaze = None
        self.real_left_gaze = None
        self.norm_eye_frames = tuple()


class ImageNormalizer(BaseEstimator):
    """
    Create normalized images of eyes for every face. Is connected to one of the cameras
    """

    def __init__(self, frame_size, calibration=None):
        self.faces = []
        self.frame = None
        self.gray = None
        self.calibration = calibration
        self.frame_size = frame_size

        # Camera intrinsic parameters
        if self.calibration is None:
            self._default_intrinsic_calibration()

        # focal_length = self.frame_size[0]
        # center = (self.frame_size[1] / 2, self.frame_size[0] / 2)
        # self.calibration.camera_matrix[0, 2] = center[0]
        # self.calibration.camera_matrix[1, 2] = center[1]

        # Other settings
        self.eye_frame_size = 2
        np.set_printoptions(precision=2)

    def _default_intrinsic_calibration(self):
        self.calibration = Calibration(board_shape=None)
        focal_length = self.frame_size[0]
        center = (self.frame_size[1] / 2, self.frame_size[0] / 2)
        self.calibration.camera_matrix = np.array([[focal_length, 0, center[0]],
                                            [0, focal_length, center[1]],
                                            [0, 0, 1]], dtype="double")
        self.calibration.distortion_vector = np.zeros((4, 1))
        return self

    def _set_frame(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)


class DlibImageNormalizer(ImageNormalizer):
    def __init__(self, frame_size, model='tutorial', calibration=None):
        super().__init__(frame_size, calibration)
        # init of generic face model
        self.model = model
        path_to_models = path.dirname(__file__)
        matfile = loadmat(path.join(path_to_models, f'../bin/face_points_{self.model}.mat'))
        if model == 'dataset':
            self.model_points = matfile['model'].T
            self.model_points = self.model_points * np.array([1, 1, -1])
            self.landmarks_to_model = {37: 0,  # Left eye left corner
                                       40: 1,  # Left eye right corner
                                       43: 2,  # Right eye left corner
                                       46: 3,  # Right eye right corner
                                       49: 4,  # Left Mouth corner
                                       55: 5  # Right mouth corner
                                       }
        elif model == 'tutorial':
            self.eye_height = 70
            self.model_points = matfile['model']
            self.model_points = self.model_points * np.array([1, -1, -1])
            self.landmarks_to_model = {31: 0,  # Nose tip
                                       9: 1,  # Chin
                                       37: 2,  # Left eye left corner
                                       46: 3,  # Right eye right corner
                                       49: 4,  # Left Mouth corner
                                       55: 5  # Right mouth corner
                                       }
        else:
            raise Exception('Not supported model!')

        # init dlib face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path.join(path_to_models,
                                                        '../bin/face_landmarks.dat'))

    def _detect_faces(self):
        self.faces = [Face(rect) for rect in self.detector(self.gray, 1)]
        return self

    def _detect_landmarks(self):
        for face in self.faces:
            face.landmarks = face_utils.shape_to_np(self.predictor(self.gray, face.rect))
        return self

    def _detect_faces_poses(self):
        """
        Uses solvePnp to get all face poses (rvec and tvec)
        :return: None
        """
        for (i, person_face) in enumerate(self._extract_face_landmarks()):
            person_points = np.array(person_face, dtype="double")
            (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points,
                                                                          person_points,
                                                                          self.calibration.camera_matrix,
                                                                          self.calibration.distortion_vector,
                                                                          flags=cv2.SOLVEPNP_ITERATIVE)
            self.faces[i].rvec = rotation_vector
            self.faces[i].tvec = translation_vector
        return self

    def _extract_eye_landmarks(self):
        """
        Extract all eye landmarks from self.faces
        :param draw:
        :return: Eye landmarks for every face
        """
        # TODO It works only for model == 'tutorial'
        landmarks = []
        for (k, face) in enumerate(self.faces):
            landmarks.append([])
            for (j, (x, y)) in enumerate(face.landmarks):
                if (j + 1) in np.arange(37, 49):
                    landmarks[k].append((x, y))
        return landmarks

    def _extract_face_landmarks(self):
        """
        Extract 6 face landmarks from self.landmarks which corresponds to generic face model
        :return: None
        """
        faces_landmarks = [[None] * 6] * len(self.faces)
        for (k, face) in enumerate(self.faces):
            for (j, (x, y)) in enumerate(face.landmarks):
                if j + 1 in self.landmarks_to_model.keys():
                    faces_landmarks[k][self.landmarks_to_model[j + 1]] = (x, y)
        return faces_landmarks

    def _extract_normalized_eye_frames(self, equalize=True):
        """
        Generates normalized eye images of fixed shape (60, 36) * self.eye_frame_size
        :return: List of tuples: (left_eye_frame, right_eye_frame)
        """
        # TODO Remove hardcode
        # generic canvas for eyes
        model_left_eye, model_right_eye = self.model_points[2], self.model_points[3]

        four_points_plane = np.array([(model_left_eye[0], model_left_eye[1] - self.eye_height, model_left_eye[2]),
                                      (model_right_eye[0], model_left_eye[1] - self.eye_height, model_left_eye[2]),
                                      (model_right_eye[0], model_left_eye[1] + self.eye_height, model_left_eye[2]),
                                      (model_left_eye[0], model_left_eye[1] + self.eye_height, model_left_eye[2])],
                                     dtype="double")
        translation_to_eyes = - np.array([model_left_eye[0], model_left_eye[1] - self.eye_height])

        for face in self.faces:
            # Drawing plane in front of face on frame
            four_points_plane_proj, _ = cv2.projectPoints(four_points_plane, face.rvec,
                                                          face.tvec, self.calibration.camera_matrix,
                                                          self.calibration.distortion_vector)

            # Calculate Homography
            homography, status = cv2.findHomography(four_points_plane_proj[:, 0, 0:2],
                                           four_points_plane[:, 0:2] + translation_to_eyes)

            eyes = cv2.warpPerspective(self.frame, homography, (225 * 2, 70 * 2))

            # Cropping, resizing, grayscaling, equalizing
            left_eye_frame, right_eye_frame = eyes[25:-25, :150], eyes[25:-25, -150:]
            left_eye_frame, right_eye_frame = cv2.resize(left_eye_frame,
                                                         (60 * self.eye_frame_size, 36 * self.eye_frame_size)), \
                                              cv2.resize(right_eye_frame,
                                                         (60 * self.eye_frame_size, 36 * self.eye_frame_size))
            left_eye_frame, right_eye_frame = cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(
                right_eye_frame, cv2.COLOR_BGR2GRAY)
            if equalize:
                left_eye_frame, right_eye_frame = cv2.equalizeHist(left_eye_frame), cv2.equalizeHist(right_eye_frame)
            face.norm_eye_frames = (left_eye_frame, right_eye_frame)

        return [face.norm_eye_frames for face in self.faces]

    def fit_transform(self, image):
        self._set_frame(image)
        self._detect_faces()
        if len(self.faces):
            self._detect_landmarks()
            self._detect_faces_poses()
            return self._extract_normalized_eye_frames()


class StandNormalizer(ImageNormalizer):
    def __init__(self, frame_size, calibration, rotation_vector, translation_vector):
        super().__init__(frame_size, calibration)
        self.camera_rotation_vector = rotation_vector
        self.camera_translation_vector = translation_vector

        # eye landmarks
        self.right_eye_rect = [1084, 847, 947, 772]
        self.left_eye_rect = [1080, 201, 289, 151]

        # landmarks for defining eye centers
        self.right_eye_center = [876, 1094, 728, 1091, 975, 1012]
        self.left_eye_center = [238, 121, 1103, 1106, 210, 1116]

        # eye planes
        self.left_norm_image_plane = np.array([[60.0, 0.0],
                                               [0.0, 0.0],
                                               [0., 36.0],
                                               [60.0, 36.0]]) * self.eye_frame_size
        self.right_norm_image_plane = np.array([[0.0, 0.0],
                                                [60.0, 0.0],
                                                [60.0, 36.0],
                                                [0., 36.0]]) * self.eye_frame_size

    def _set_faces(self, face_points_json, faces_quaternions, gaze_in_kinect_space):
        landmarks_kinect_space = np.array([list(face_points_json[i].values())\
                                           for i in self.right_eye_rect + self.left_eye_rect]).T * np.array([[1], [-1], [1]])

        landmarks_eye_centers = np.array([list(face_points_json[i].values())\
                                              for i in self.right_eye_center + self.left_eye_center]).T * np.array([[1], [-1], [1]])

        # TODO landmars for all faces
        self.faces = [Face(None)]
        for i, face in enumerate(self.faces):
            face.landmarks = landmarks_kinect_space
            face.rvec = quaternion_to_angle_axis(faces_quaternions[i])
            face.left_eye_center = (landmarks_eye_centers[:, 6:].sum(axis=1)/6).reshape((3, 1))
            face.right_eye_center = (landmarks_eye_centers[:, :6].sum(axis=1)/6).reshape((3, 1))

        # setting real gaze vector for training dataset only for first face
        if gaze_in_kinect_space is not None:
            real_right_gaze = gaze_in_kinect_space - self.faces[0].right_eye_center
            self.faces[0].real_right_gaze = real_right_gaze/np.linalg.norm(real_right_gaze)
            real_left_gaze = gaze_in_kinect_space - self.faces[0].left_eye_center
            self.faces[0].real_left_gaze = real_left_gaze/np.linalg.norm(real_left_gaze)

        return self

    def _extract_normalized_eye_frames(self, equalize=False):
        """
        Generates normalized eye images of fixed shape (60, 36) * self.eye_frame_size
        :return: List of tuples: (left_eye_frame, right_eye_frame)
        """

        for face in self.faces:
            # 3D eye landmarks -> 2D eye landmarks on frame
            end_points, _ = cv2.projectPoints(face.landmarks.T,
                                              self.camera_rotation_vector,
                                              self.camera_translation_vector,
                                              self.calibration.camera_matrix,
                                              self.calibration.distortion_vector)
            end_points = end_points.reshape((8, 2))

            # TODO remove duplication
            # left eye homography
            homography, status = cv2.findHomography(end_points[4:], self.left_norm_image_plane)
            left_eye_frame = cv2.warpPerspective(self.frame, homography, (60 * self.eye_frame_size, 36 * self.eye_frame_size))

            # right eye homography
            homography, status = cv2.findHomography(end_points[:4], self.right_norm_image_plane)
            right_eye_frame = cv2.warpPerspective(self.frame, homography, (60 * self.eye_frame_size, 36 * self.eye_frame_size))

            # Grayscaling, equalizing
            left_eye_frame, right_eye_frame = cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2GRAY), \
                                              cv2.cvtColor(right_eye_frame, cv2.COLOR_BGR2GRAY)
            if equalize:
                left_eye_frame, right_eye_frame = cv2.equalizeHist(left_eye_frame), cv2.equalizeHist(right_eye_frame)
            face.norm_eye_frames = (left_eye_frame, right_eye_frame)


        return [face.norm_eye_frames for face in self.faces]

    def fit(self, face_points_json, faces_quaternions, gaze_in_kinect_space=None):
        self._set_faces(face_points_json, faces_quaternions, gaze_in_kinect_space)
        return self

    def transform(self, image):
        self._set_frame(image)
        if len(self.faces):
            return self._extract_normalized_eye_frames()

    def fit_transform(self, image, face_points_json, faces_quaternions, gaze_in_kinect_space=None):
        self.fit(face_points_json, faces_quaternions, gaze_in_kinect_space)
        return self.transform(image)
