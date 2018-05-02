from math import sqrt

import cv2
import numpy as np
from imutils import face_utils


def quaternion_to_angle_axis(quaternion):
    """
    Convert angle-axis to quaternion
    :param quaternion: dict {'X': , 'Y': , 'Z': , 'W': }
    :return: angle-axis rotation vector
    """
    t = sqrt(1-quaternion['W']*quaternion['W'])
    if t:
        x = quaternion['X'] / t
        y = quaternion['Y'] / t
        z = quaternion['Z'] / t
        return np.array([[x], [y], [z]])
    else:
        return np.zeros((3, 1))

def POG_to_kinect_space(POGX, POGY, screen_size, diagonal_in_meters, rvec, tvec):
    """
    :param POGX: Gazepoint FPOGX/BPOGX
    :param POGY: Gazepoint FPOGY/BPOGY
    :param screen_size: tuple (width, height) in pixels
    :param diagonal_in_meters: screen diagonal in meters
    :param rvec: screen rotation vector (extrinsic parameter)
    :param tvec: screen translation vector (extrinsic parameter)
    :return: POG in 3D kinect space
    """
    meters_per_pixel = diagonal_in_meters / np.sqrt(screen_size[0]**2 + screen_size[1]**2)
    POG_meters = np.array([[POGX * screen_size[0] * meters_per_pixel],
                           [POGY * screen_size[1] * meters_per_pixel],
                           [0.0]])
    rot_matr, _ = cv2.Rodrigues(rvec)
    return POG_meters @ rot_matr + tvec

def draw_faces_rectangles(normalisator):
    """
    Draw all faces rectangle and numbers all rectangles
    :return: None
    """
    for (k, face) in enumerate(normalisator.faces):
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face.rect)
        cv2.rectangle(normalisator.frame,
                      (x, y),
                      (x + w, y + h),
                      (0, 255, 0), 2)

        # show the face number
        cv2.putText(normalisator.frame,
                    "Face #{}".format(k + 1),
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

def draw_eye_centeres(normalisator):
    """
    Draw all eye centers
    :return: None
    """
    for eye_landmark in normalisator._extract_eye_landmarks():
        # drawing eyes centers
        eyes = np.array(eye_landmark)
        left_eye = eyes[:6]
        right_eye = eyes[6:]
        left_eye_center = (left_eye.sum(axis=0) / left_eye.shape[0])
        right_eye_center = (right_eye.sum(axis=0) / right_eye.shape[0])

        cv2.circle(normalisator.frame,
                   tuple(left_eye_center.astype(int)),
                   1, (0, 0, 255), -1)
        cv2.circle(normalisator.frame,
                   tuple(right_eye_center.astype(int)),
                   1, (0, 0, 255), -1)

def draw_eye_borders(normalisator):
    """
    Draws rectangles around eyes
    :return: None
    """
    left_eye = normalisator.model_points[3]
    right_eye = normalisator.model_points[2]
    normalisator.eye_height = 70
    ### !
    four_points_plane = np.array([(left_eye[0], left_eye[1] - normalisator.eye_height, left_eye[2]),
                                  (right_eye[0], left_eye[1] - normalisator.eye_height, left_eye[2]),
                                  (right_eye[0], left_eye[1] + normalisator.eye_height, left_eye[2]),
                                  (left_eye[0], left_eye[1] + normalisator.eye_height, left_eye[2])],
                                 dtype="double")

    for face in normalisator.faces:
        (end_points2D, jacobian) = cv2.projectPoints(four_points_plane,
                                                     face.rvec,
                                                     face.tvec,
                                                     normalisator.camera_matrix,
                                                     normalisator.dist_coeffs)
        cv2.line(normalisator.frame,
                 (int(end_points2D[0][0][0]), int(end_points2D[0][0][1])),
                 (int(end_points2D[1][0][0]), int(end_points2D[1][0][1])),
                 (255, 0, 0), 2)
        cv2.line(normalisator.frame,
                 (int(end_points2D[1][0][0]), int(end_points2D[1][0][1])),
                 (int(end_points2D[2][0][0]), int(end_points2D[2][0][1])),
                 (255, 0, 0), 2)
        cv2.line(normalisator.frame,
                 (int(end_points2D[2][0][0]), int(end_points2D[2][0][1])),
                 (int(end_points2D[3][0][0]), int(end_points2D[3][0][1])),
                 (255, 0, 0), 2)
        cv2.line(normalisator.frame,
                 (int(end_points2D[3][0][0]), int(end_points2D[3][0][1])),
                 (int(end_points2D[0][0][0]), int(end_points2D[0][0][1])),
                 (255, 0, 0), 2)

def draw_gazes(normalisator):
    """
        Draws vector of gazes on image
        :return: None
        """
    for (i, face) in enumerate(normalisator.faces):
        left_eye_frame, right_eye_frame = normalisator.norm_eye_frames[i]

        # drawing left eye vector
        #
        ## ! this is a same code. should be a function
        #
        left_eye_center = np.array([150.0, -170.0, 135.0])## !!!
        left_eye_gaze = left_eye_center + 750 * face.gaze[0]
        end_points, _ = cv2.projectPoints(np.array([left_eye_center, left_eye_gaze]), face.rvec,
                                          face.tvec, normalisator.calibration.camera_matrix,
                                          normalisator.calibration.distortion_vector)
        cv2.line(normalisator.frame, (int(end_points[0][0][0]), int(end_points[0][0][1])),
                 (int(end_points[1][0][0]), int(end_points[1][0][1])), (255, 0, 0), 2)

        # drawing right eye vector
        right_eye_center = np.array([-150.0, -170.0, 135.0])
        right_eye_gaze = right_eye_center + 750 * face.gaze[1]
        end_points, _ = cv2.projectPoints(np.array([right_eye_center, right_eye_gaze * np.array([-1, 1, 1])]),
                                          face.rvec, face.tvec, normalisator.calibration.camera_matrix,
                                          normalisator.calibration.distortion_vector)
        cv2.line(normalisator.frame, (int(end_points[0][0][0]), int(end_points[0][0][1])),
                 (int(end_points[1][0][0]), int(end_points[1][0][1])), (255, 0, 0), 2)

def draw_face_norm(self, screen):
    self.background = np.zeros((screen), dtype=np.uint8)

    for (i, face_pose) in enumerate(self.faces_poses):
        A = np.array([150.0, -170.0, 135.0])
        left_eye_frame, right_eye_frame = self.norm_eye_frames[i]
        gaze = None #estimate_gaze(np.array([left_eye_frame]), np.zeros((3, 1)))
        B = A + gaze[0]

        rmat, _ = (cv2.Rodrigues(face_pose[0]))

        A = (rmat) @ A + face_pose[1].reshape(3)
        B = (rmat) @ B + face_pose[1].reshape(3)

        x = (int(screen[1] / 2) - int((A[0] - A[2] * (A[0] - B[0]) / (A[2] - B[2]))))
        y = int((A[1] - A[2] * (A[1] - B[1]) / (A[2] - B[2])))

        cv2.circle(self.background,
                   (x, y),
                   20,
                   100,
                   -1)
        cv2.putText(self.background,
                    str((x, y)),
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    100,
                    2)

def draw_eye_centers(normalizer):
    for face in normalizer.faces:
        end_points, _ = cv2.projectPoints(np.array([face.left_eye_center, face.right_eye_center]), np.zeros((3, 1)),
                                          np.zeros((3, 1)), normalizer.calibration.camera_matrix,
                                          normalizer.calibration.distortion_vector)
        end_points = end_points.reshape((-1, 2))
        for end_point in end_points:
            _ = cv2.circle(normalizer.frame, (int(end_point[0]), int(end_point[1])), 5, (255, 0, 0), -1)

def draw_gazes(normalizer):
    for face in normalizer.faces:
        left_eye_center = face.left_eye_center
        face_rot, _ = cv2.Rodrigues(face.rvec)
        gaze_in_kinect_space = np.linalg.inv(face_rot) @ (face.gaze * 1) + left_eye_center
        end_points, _ = cv2.projectPoints(np.array([left_eye_center, gaze_in_kinect_space]), np.zeros((3, 1)),
                                          np.zeros((3, 1)), normalizer.calibration.camera_matrix,
                                          normalizer.calibration.distortion_vector)
        cv2.line(normalizer.frame, (int(end_points[0][0][0]), int(end_points[0][0][1])),
                 (int(end_points[1][0][0]), int(end_points[1][0][1])), (255, 0, 0), 2)
