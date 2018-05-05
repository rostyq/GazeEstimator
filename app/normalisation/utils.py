from math import sqrt

import cv2
import numpy as np
from imutils import face_utils

from config import *


def quaternion_to_angle_axis(quaternion):
    """
    Converts angle-axis to quaternion
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


def POG_to_kinect_space(POGX, POGY):
    """
    Converts POG 2D point to 3D kinect space
    :param POGX: Gazepoint FPOGX/BPOGX
    :param POGY: Gazepoint FPOGY/BPOGY
    :param screen_size: tuple (width, height) in pixels
    :param diagonal_in_meters: screen diagonal in meters
    :param screen_rotation_vector: screen rotation vector (extrinsic parameter) (upper left corner)
    :param screen_translation_vector: screen translation vector (extrinsic parameter) (upper left corner)
    :return: POG in 3D kinect space
    """
    screen_size = np.array(SCREEN_PARAMETERS['screen_size'])
    diagonal_in_meters = np.array(SCREEN_PARAMETERS['diagonal_in_meters'])
    meters_per_pixel = diagonal_in_meters / np.sqrt(screen_size[0]**2 + screen_size[1]**2)
    POG_in_meters = np.array([[float(POGX) * screen_size[0] * meters_per_pixel],
                              [float(POGY) * screen_size[1] * meters_per_pixel],
                              [0.0]])
    rot_matr, _ = cv2.Rodrigues(np.array(SCREEN_PARAMETERS['rotation_vector']))
    return rot_matr @ POG_in_meters - np.array(SCREEN_PARAMETERS['translation_vector'])


def draw_faces_rectangles(dlib_normalizer, frame):
    """
    Draws all faces rectangle and numbers all rectangles. Works only for normalization camera
    :return: None
    """
    for (k, face) in enumerate(dlib_normalizer.faces):
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face.rect)
        cv2.rectangle(frame,
                      (x, y),
                      (x + w, y + h),
                      (0, 255, 0), 2)

        # show the face number
        cv2.putText(frame,
                    "Face #{}".format(k + 1),
                    (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)


def draw_eye_borders(dlib_normalizer, frame):
    """
    Draws two rectangles in front of eyes
    :return: None
    """
    # TODO Rewrite code for all cameras and remove hardcode
    # Initialization of eye borders
    left_eye = dlib_normalizer.model_points[3]
    right_eye = dlib_normalizer.model_points[2]
    dlib_normalizer.eye_height = 70
    four_points_plane = np.array([(left_eye[0], left_eye[1] - dlib_normalizer.eye_height, left_eye[2]),
                                  (right_eye[0], left_eye[1] - dlib_normalizer.eye_height, left_eye[2]),
                                  (right_eye[0], left_eye[1] + dlib_normalizer.eye_height, left_eye[2]),
                                  (left_eye[0], left_eye[1] + dlib_normalizer.eye_height, left_eye[2])],
                                 dtype="double")

    for face in dlib_normalizer.faces:
        # Projection
        end_points, _ = cv2.projectPoints(four_points_plane,
                                          face.rvec,
                                          face.tvec,
                                          dlib_normalizer.camera_matrix,
                                          dlib_normalizer.dist_coeffs)
        end_points = end_points.reshape((-1, 2))
        
        # Drawing
        cv2.line(frame,
                 (int(end_points[0][0]), int(end_points[0][1])),
                 (int(end_points[1][0]), int(end_points[1][1])),
                 (255, 0, 0), 2)
        cv2.line(frame,
                 (int(end_points[1][0]), int(end_points[1][1])),
                 (int(end_points[2][0]), int(end_points[2][1])),
                 (255, 0, 0), 2)
        cv2.line(frame,
                 (int(end_points[2][0]), int(end_points[2][1])),
                 (int(end_points[3][0]), int(end_points[3][1])),
                 (255, 0, 0), 2)
        cv2.line(frame,
                 (int(end_points[3][0]), int(end_points[3][1])),
                 (int(end_points[0][0]), int(end_points[0][1])),
                 (255, 0, 0), 2)


def draw_gazes(dlib_normalizer, frame):
    """
    Draws estimated gaze vectors
    :return: None
    """
    # TODO Rewrite code for all cameras and remove hardcode
    for (i, face) in enumerate(dlib_normalizer.faces):
        # Projection
        left_eye_center = np.array([150.0, -170.0, 135.0])
        left_eye_gaze = left_eye_center + 750 * face.gaze[0]
        right_eye_center = np.array([-150.0, -170.0, 135.0])
        right_eye_gaze = right_eye_center + 750 * face.gaze[1]
        gaze_points = np.array([left_eye_center, left_eye_gaze, right_eye_center, right_eye_gaze])
        
        end_points, _ = cv2.projectPoints(gaze_points, 
                                          face.rvec,
                                          face.tvec, 
                                          dlib_normalizer.calibration.camera_matrix,
                                          dlib_normalizer.calibration.distortion_vector)
        end_points = end_points.reshape((-1, 2))

        # Drawing
        _draw_endpoints(end_points, frame)
        _draw_gaze_lines(end_points, frame)


def draw_eye_landmarks(stand_normalizer, frame, camera='basler'):
    """
    Draws 4 corner landmarks for each eye
    :return: None
    """
    for face in stand_normalizer.faces:
        # Projection
        landmarks = np.append(face.landmarks.T, 
                              np.array([face.left_eye_center, face.right_eye_center]).reshape(-1, 3),
                              axis=0)
        end_points, _ = cv2.projectPoints(landmarks,
                                          np.array(CAMERAS_PARAMETERS[camera]['rotation_vector']),
                                          np.array(CAMERAS_PARAMETERS[camera]['translation_vector']),
                                          np.array(CAMERAS_PARAMETERS[camera]['matrix']),
                                          np.array(CAMERAS_PARAMETERS[camera]['distortion']))
        end_points = end_points.reshape((-1, 2))

        # Drawing
        _draw_endpoints(end_points, frame)


def draw_screen(frame, camera='basler'):
    """
    Draws 8 points on screen edges
    :return: None
    """
    # Projection
    screen_edges_GAZEPOINT = [[0., 0.], [0., 1.], [0., 0.5], [0.5, 0], [1., 0], [1., 0.5], [1., 1.], [0.5, 1.]]
    screen_edges_kinect = np.array([POG_to_kinect_space(edge[0], edge[1]) for edge in screen_edges_GAZEPOINT])
    end_points, _ = cv2.projectPoints(screen_edges_kinect,
                                      np.array(CAMERAS_PARAMETERS[camera]['rotation_vector']),
                                      np.array(CAMERAS_PARAMETERS[camera]['translation_vector']),
                                      np.array(CAMERAS_PARAMETERS[camera]['matrix']),
                                      np.array(CAMERAS_PARAMETERS[camera]['distortion']))
    end_points = end_points.reshape((-1, 2))

    # Drawing
    _draw_endpoints(end_points, frame)


def draw_real_gazes(stand_normalizer, frame, camera='basler'):
    """
    Draws gaze vectors, estimated by Gazepoint
    :return: None
    """
    for face in stand_normalizer.faces:
        # Projection
        gaze_points = np.array([face.left_eye_center,
                                face.left_eye_center + face.real_left_gaze,
                                face.right_eye_center,
                                face.right_eye_center + face.real_right_gaze]).reshape((-1, 3))
        end_points, _ = cv2.projectPoints(gaze_points,
                                          np.array(CAMERAS_PARAMETERS[camera]['rotation_vector']),
                                          np.array(CAMERAS_PARAMETERS[camera]['translation_vector']),
                                          np.array(CAMERAS_PARAMETERS[camera]['matrix']),
                                          np.array(CAMERAS_PARAMETERS[camera]['distortion']))
        end_points = end_points.reshape((-1, 2))
        
        # Drawing
        _draw_endpoints(end_points, frame)
        _draw_gaze_lines(end_points, frame)


def _draw_endpoints(end_points, frame):
    for end_point in end_points:
        _ = cv2.circle(frame, (int(end_point[0]), int(end_point[1])), 10, (0, 255, 0), -1)
        
def _draw_gaze_lines(end_points, frame):
    cv2.line(frame, (int(end_points[0][0]), int(end_points[0][1])),
             (int(end_points[1][0]), int(end_points[1][1])), (255, 0, 0), 2)
    cv2.line(frame, (int(end_points[2][0]), int(end_points[2][1])),
             (int(end_points[3][0]), int(end_points[3][1])), (255, 0, 0), 2)