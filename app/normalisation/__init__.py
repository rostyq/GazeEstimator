from os import path

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.io import loadmat

from ..calibration import Calibration


class Face:
    def __init__(self, rect):
        self.landmarks = []
        self.rect = rect
        self.rvec = None
        self.tvec = None
        self.gaze = None
        self.norm_eye_frames = tuple()

class ImageNormalizer:
    '''
    Create normalized images of eyes for every face. Is connected to one of the cameras
    '''
    def __init__(self, frame_size, calibration=None):
        self.faces = []
        self.frame = None
        self.gray = None
        self.calibration = calibration
        self.frame_size = frame_size

        # Camera intrinsic parameters
        if self.calibration is None:
            self._default_intrinsic_calibration()

        # Other settings
        np.set_printoptions(precision=2)

    def _default_intrinsic_calibration(self):
        self.calibration = Calibration(board_shape=None)
        focal_length = self.frame_size[1]
        center = (self.frame_size[1] / 2, self.frame_size[0] / 2)
        self.calibration.matrix = np.array([[focal_length, 0, center[0]],
                                       [0, focal_length, center[1]],
                                       [0, 0, 1]], dtype="double")
        self.calibration.distortion = np.zeros((4, 1))


    def set_frame(self, frame):
        '''
        Set image for feature extraction
        :param frame: image for analysis
        :return: None
        '''
        self.frame = frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def get_normalized_eye_frames(self, image, face_points_json):
        '''
        Generates normalized eye images of fixed shape (60, 36)
        :return: List of tuples: (left_eye_frame, right_eye_frame)
        '''
        pass

class DlibImageNormalizer(ImageNormalizer):
    def __init__(self, frame_size, model='tutorial', calibration=None):
        super().__init__(frame_size)
        # init of generic face model
        self.model = model
        path_to_models = path.dirname(__file__)
        matfile = loadmat(path.join(path_to_models, f'../binaries/6_points_face_model_{self.model}.mat'))
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
                                                        '../binaries/shape_predictor_68_face_landmarks.dat'))

    def detect_faces(self):
        self.faces = [Face(rect) for rect in self.detector(self.gray, 1)]

    def detect_landmarks(self):
        for face in self.faces:
            face.landmarks = face_utils.shape_to_np(self.predictor(self.gray, face.rect))

    def detect_faces_poses(self):
        '''
        Uses solvePnp to get all face poses (rvec and tvec)
        :return: None
        '''
        for (i, person_face) in enumerate(self.extract_face_landmarks()):
            person_points = np.array(person_face, dtype="double")
            (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points,
                                                                          person_points,
                                                                          self.calibration.matrix,
                                                                          self.calibration.distortion,
                                                                          flags=cv2.SOLVEPNP_ITERATIVE)
            self.faces[i].rvec = rotation_vector
            self.faces[i].tvec = translation_vector

    def extract_eye_landmarks(self):
        '''
        Extract all eye landmarks from self.faces
        :param draw:
        :return: Eye landmarks for every face
        '''
        # TODO It works only for model == 'tutorial'
        landmarks = []
        for (k, face) in enumerate(self.faces):
            landmarks.append([])
            for (j, (x, y)) in enumerate(face.landmarks):
                if (j + 1) in np.arange(37, 49):
                    landmarks[k].append((x, y))
        return landmarks

    def extract_face_landmarks(self):
        '''
        Extract 6 face landmarks from self.landmarks which corresponds to generic face model
        :return: None
        '''
        faces_landmarks = [[None] * 6] * len(self.faces)
        for (k, face) in enumerate(self.faces):
            for (j, (x, y)) in enumerate(face.landmarks):
                if j + 1 in self.landmarks_to_model.keys():
                    faces_landmarks[k][self.landmarks_to_model[j + 1]] = (x, y)
        return faces_landmarks

    def extract_normalized_eye_frames(self):
        '''
        Generates normalized eye images of fixed shape (60, 36)
        :return: List of tuples: (left_eye_frame, right_eye_frame)
        '''
        # TODO Remove hardcode
        # generic canvas for eyes
        l_eye, r_eye = self.model_points[2], self.model_points[3]

        four_points_plane = np.array([(l_eye[0], l_eye[1] - self.eye_height, l_eye[2]),
                                      (r_eye[0], l_eye[1] - self.eye_height, l_eye[2]),
                                      (r_eye[0], l_eye[1] + self.eye_height, l_eye[2]),
                                      (l_eye[0], l_eye[1] + self.eye_height, l_eye[2])], dtype="double")
        translation_to_eyes = - np.array([l_eye[0], l_eye[1] - self.eye_height])

        for face in self.faces:

            # Drawing plane in front of face
            four_points_plane_proj, _ = cv2.projectPoints(four_points_plane, face.rvec,
                                                          face.tvec, self.calibration.matrix, self.calibration.distortion)

            # Calculate Homography
            h, status = cv2.findHomography(four_points_plane_proj[:, 0, 0:2],
                                           four_points_plane[:, 0:2] + translation_to_eyes)

            eyes = cv2.warpPerspective(self.frame, h, (225 * 2, 70 * 2))
            left_eye_frame, right_eye_frame = eyes[25:-25, :150], eyes[25:-25, -150:]
            left_eye_frame, right_eye_frame = cv2.resize(left_eye_frame, (60, 36)), cv2.resize(right_eye_frame,
                                                                                               (60, 36))
            left_eye_frame, right_eye_frame = cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(
                right_eye_frame, cv2.COLOR_BGR2GRAY)
            left_eye_frame, right_eye_frame = cv2.equalizeHist(left_eye_frame), cv2.equalizeHist(right_eye_frame)
            face.norm_eye_frames = (left_eye_frame, right_eye_frame)

        return [face.norm_eye_frames for face in self.faces]

    def get_normalized_eye_frames(self, image, face_points_json=None):
        self.set_frame(image)
        self.detect_faces()
        if len(self.faces):
            self.detect_landmarks()
            self.detect_faces_poses()
            return self.extract_normalized_eye_frames()

class StandNormalizazer(ImageNormalizer):
    def __init__(self, frame_size, calibration):
        super().__init__(frame_size, calibration)
        self.camera_rvec = None
        self.camera_tvec = np.array([0.05, 0., 0.])

        #eye landmarks
        self.right_eye_rect = [803, 828, 719, 969]
        self.left_eye_rect = [346, 339, 115, 314]

        #eye planes
        self.left_norm_image_plane = np.array([[0.0, 0.0],
                                             [120.0, 0.0],
                                             [120.0, 72.0],
                                             [0., 72.0]])
        self.right_norm_image_plane = np.array([[120.0, 0.0],
                                             [0.0, 0.0],
                                             [0., 72.0],
                                             [120.0, 72.0]])

    def set_face_landmarks(self, face_points_json):
        self.landmarks_camera_space = np.array([list(face_points_json[i].values()) for i in self.right_eye_rect + self.left_eye_rect])
        self.landmarks_colored_space = (self.landmarks_camera_space + self.camera_tvec) * np.array([-1, 1, -1])

    def extract_normalized_eye_frames(self):
        end_points, _ = cv2.projectPoints(self.landmarks_colored_space, np.zeros((3, 1)),
                                          np.zeros((3, 1)), self.calibration.matrix, self.calibration.distortion)
        end_points = end_points.reshape((8, 2))

        #left eye
        h, status = cv2.findHomography(np.array(end_points)[:4], self.left_norm_image_plane)
        left_eye_frame = cv2.warpPerspective(self.frame, h, (120, 72))

        #right eye
        h, status = cv2.findHomography(np.array(end_points)[4:], self.right_norm_image_plane)
        right_eye_frame = cv2.warpPerspective(self.frame, h, (120, 72))

        left_eye_frame, right_eye_frame = cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(
            right_eye_frame, cv2.COLOR_BGR2GRAY)
        left_eye_frame, right_eye_frame = cv2.equalizeHist(left_eye_frame), cv2.equalizeHist(right_eye_frame)

        return left_eye_frame, right_eye_frame

    def get_normalized_eye_frames(self, image, face_points_json):
        self.set_frame(image)
        self.set_face_landmarks(face_points_json)

        if len(self.landmarks_camera_space):
            return self.extract_normalized_eye_frames()

