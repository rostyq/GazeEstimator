import cv2
import numpy as np
from imutils import face_utils
from scipy.io import loadmat
from os import getcwd, path
from estimator import estimate_gaze
import dlib


class FacesRecognition:
    def __init__(self, size, camera_matrix=None, dist_coeffs=None, model='tutorial'):
        # init generic face model
        self.model = model
        path_to_models = path.join(getcwd(), 'normalisation')
        matfile = loadmat(path.join(path_to_models, f'6_points_face_model_{self.model}.mat'))
        if model == 'dataset':
            self.model_points = matfile['model'].T
            self.model_points = self.model_points * np.array([1, 1, -1])
        else:
            self.model_points = matfile['model']
            self.model_points = self.model_points * np.array([1, -1, -1])

        # init face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path.join(path_to_models, 'shape_predictor_68_face_landmarks.dat'))

        # Camera internals
        self.size = size
        self.focal_length = size[1]
        self.center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = camera_matrix if camera_matrix is not None else \
                            np.array([[self.focal_length, 0, self.center[0]],
                                      [0, self.focal_length, self.center[1]],
                                      [0, 0, 1]], dtype="double")
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

        np.set_printoptions(precision=2)

    def set_image(self, frame):
        self.frame = frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def decect_faces(self):
        self.rects = self.detector(self.gray, 1)

    def draw_faces_rectangles(self):
        for (k, rect) in enumerate(self.rects):
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(self.frame, "Face #{}".format(k + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def detect_landmarks(self):
        self.landmarks = []
        for (k, rect) in enumerate(self.rects):
            shape = self.predictor(self.gray, rect)
            self.landmarks.append(face_utils.shape_to_np(shape))

    def get_eye_landmarks(self, draw=False):
        landmarks = []
        for (k, shape) in enumerate(self.landmarks):
            landmarks.append([])
            for (j, (x, y)) in enumerate(shape):
                if (j + 1) in np.arange(37, 49):
                    if draw:
                        cv2.circle(self.frame, (x, y), 1, (0, 0, 255), -1)
                    landmarks[k].append((x, y))
        return landmarks

    def get_face_landmarks(self):
        faces_landmarks = np.empty((len(self.landmarks), 6)).tolist()
        for (k, shape) in enumerate(self.landmarks):

            j = 1
            for (x, y) in shape:
                if self.model == 'dataset':
                    if j == 37:  # Left eye left corner
                        faces_landmarks[k][0] = (x, y)
                    elif j == 40:  # Left eye right corner
                        faces_landmarks[k][1] = (x, y)
                    elif j == 43:  # Right eye left corner
                        faces_landmarks[k][2] = (x, y)
                    elif j == 46:  # Right eye right corner
                        faces_landmarks[k][3] = (x, y)
                    elif j == 49:  # Left Mouth corner
                        faces_landmarks[k][4] = (x, y)
                    elif j == 55:  # Right mouth corner
                        faces_landmarks[k][5] = (x, y)
                else:
                    if j == 31:  # Nose tip
                        faces_landmarks[k][0] = (x, y)
                    elif j == 9:  # Chin
                        faces_landmarks[k][1] = (x, y)
                    elif j == 37:  # Left eye left corner
                        faces_landmarks[k][2] = (x, y)
                    elif j == 46:  # Right eye right corner
                        faces_landmarks[k][3] = (x, y)
                    elif j == 49:  # Left Mouth corner
                        faces_landmarks[k][4] = (x, y)
                    elif j == 55:  # Right mouth corner
                        faces_landmarks[k][5] = (x, y)
                j += 1
        return faces_landmarks

    def draw_eye_centeres(self):

        for eye_landmark in self.get_eye_landmarks():
            # drawing eyes centers
            eyes = np.array(eye_landmark)
            l_eye, r_eye = eyes[:6], eyes[6:]
            l_eye_c, r_eye_c = (l_eye.sum(axis=0) / l_eye.shape[0]), (r_eye.sum(axis=0) / r_eye.shape[0])

            cv2.circle(self.frame, tuple(l_eye_c.astype(int)), 1, (0, 0, 255), -1)
            cv2.circle(self.frame, tuple(r_eye_c.astype(int)), 1, (0, 0, 255), -1)

    def detect_faces_poses(self):
        self.faces_poses = []
        for person_face in self.get_face_landmarks():

            person_points = np.array(person_face, dtype="double")

            (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points,
                                                                          person_points,
                                                                          self.camera_matrix,
                                                                          self.dist_coeffs,
                                                                          flags=cv2.SOLVEPNP_ITERATIVE)
            self.faces_poses.append((rotation_vector, translation_vector))

    def draw_eye_borders(self):
        l_eye, r_eye = self.model_points[2], self.model_points[3]
        self.eye_height = 70
        four_points_plane = np.array([(l_eye[0], l_eye[1] - self.eye_height, l_eye[2]),
                                      (r_eye[0], l_eye[1] - self.eye_height, l_eye[2]),
                                      (r_eye[0], l_eye[1] + self.eye_height, l_eye[2]),
                                      (l_eye[0], l_eye[1] + self.eye_height, l_eye[2])], dtype="double")

        for face_pose in self.faces_poses:
            (end_points2D, jacobian) = cv2.projectPoints(four_points_plane, face_pose[0], face_pose[1],
                                                         self.camera_matrix, self.dist_coeffs)
            cv2.line(self.frame, (int(end_points2D[0][0][0]), int(end_points2D[0][0][1])),
                     (int(end_points2D[1][0][0]), int(end_points2D[1][0][1])), (255,0,0), 2)
            cv2.line(self.frame, (int(end_points2D[1][0][0]), int(end_points2D[1][0][1])),
                     (int(end_points2D[2][0][0]), int(end_points2D[2][0][1])), (255,0,0), 2)
            cv2.line(self.frame, (int(end_points2D[2][0][0]), int(end_points2D[2][0][1])),
                     (int(end_points2D[3][0][0]), int(end_points2D[3][0][1])), (255,0,0), 2)
            cv2.line(self.frame, (int(end_points2D[3][0][0]), int(end_points2D[3][0][1])),
                     (int(end_points2D[0][0][0]), int(end_points2D[0][0][1])), (255,0,0), 2)

    def _normalized_eye_frames(self, rotation_vector, translation_vector, four_points_plane, translation_to_eyes):
        # Drawing plane in front of face
        four_points_plane_proj, _ = cv2.projectPoints(four_points_plane, rotation_vector,
                                                      translation_vector, self.camera_matrix, self.dist_coeffs)

        # Calculate Homography
        h, status = cv2.findHomography(four_points_plane_proj[:, 0, 0:2],
                                       four_points_plane[:, 0:2] + translation_to_eyes)

        eyes = cv2.warpPerspective(self.frame, h, (225 * 2, 70 * 2))
        left_eye_frame, right_eye_frame = eyes[25:-25, :150], eyes[25:-25, -150:]
        left_eye_frame, right_eye_frame = cv2.resize(left_eye_frame, (60, 36)), cv2.resize(right_eye_frame, (60, 36))
        left_eye_frame, right_eye_frame = cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(
            right_eye_frame, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(left_eye_frame), cv2.equalizeHist(right_eye_frame)

    def produce_normalized_eye_frames(self):

        #generic canvas for eyes
        l_eye, r_eye = self.model_points[2], self.model_points[3]
        self.eye_height = 70
        four_points_plane = np.array([(l_eye[0], l_eye[1] - self.eye_height, l_eye[2]),
                                      (r_eye[0], l_eye[1] - self.eye_height, l_eye[2]),
                                      (r_eye[0], l_eye[1] + self.eye_height, l_eye[2]),
                                      (l_eye[0], l_eye[1] + self.eye_height, l_eye[2])], dtype="double")
        translation_to_eyes = - np.array([l_eye[0], l_eye[1] - self.eye_height])

        self.norm_eye_frames = []
        for face_pose in self.faces_poses:
            left_eye_frame, right_eye_frame = self._normalized_eye_frames(face_pose[0], face_pose[1],
                                                                            four_points_plane, translation_to_eyes)
            self.norm_eye_frames.append((left_eye_frame, right_eye_frame))

        return self.norm_eye_frames


    def detect_gazes(self):

        #creating eye frames
        self.produce_normalized_eye_frames()

        for (i, face_pose) in enumerate(self.faces_poses):
            left_eye_frame, right_eye_frame = self.norm_eye_frames[i]

            # gaze estimation
            gaze = estimate_gaze(np.array([left_eye_frame, np.flip(right_eye_frame, axis=1)]),
                                 np.array([np.zeros((3, 1)), np.zeros((3, 1)) * np.array([[-1], [1], [1]])]))

            cv2.putText(self.frame, str(gaze), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # drawing left eye vector
            left_eye_center = np.array([150.0, -170.0, 135.0])
            left_eye_gaze = left_eye_center + 750 * gaze[0]
            end_points, _ = cv2.projectPoints(np.array([left_eye_center, left_eye_gaze]), face_pose[0],
                                              face_pose[1], self.camera_matrix, self.dist_coeffs)
            cv2.line(self.frame, (int(end_points[0][0][0]), int(end_points[0][0][1])),
                     (int(end_points[1][0][0]), int(end_points[1][0][1])), (255, 0, 0), 2)

            # drawing right eye vector
            right_eye_center = np.array([-150.0, -170.0, 135.0])
            right_eye_gaze = right_eye_center + 750 * gaze[1]
            end_points, _ = cv2.projectPoints(np.array([right_eye_center, right_eye_gaze*np.array([-1, 1, 1])]),
                                              face_pose[0], face_pose[1], self.camera_matrix, self.dist_coeffs)
            cv2.line(self.frame, (int(end_points[0][0][0]), int(end_points[0][0][1])),
                     (int(end_points[1][0][0]), int(end_points[1][0][1])), (255, 0, 0), 2)


