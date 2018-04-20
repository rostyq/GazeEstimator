from math import sqrt

import cv2
import numpy as np
from imutils import face_utils


# x = qx / sqrt(1-qw*qw)
# y = qy / sqrt(1-qw*qw)
# z = qz / sqrt(1-qw*qw)

def quaternion_to_angle_axis(quaternion):
    t = sqrt(1-quaternion['W']*quaternion['W'])
    if t != 0:
        x = quaternion['X'] / t
        y = quaternion['Y'] / t
        z = quaternion['Z'] / t
        return np.array([[x], [y], [z]])
    else:
        return np.zeros((3,1))

def draw_faces_rectangles(normalisator):
    '''
    Draw all faces rectangle and numbers all rectangles
    :return: None
    '''
    for (k, face) in enumerate(normalisator.faces):
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face.rect)
        cv2.rectangle(normalisator.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(normalisator.frame, "Face #{}".format(k + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_eye_centeres(normalisator):
    '''
    Draw all eye centers
    :return: None
    '''
    for eye_landmark in normalisator.extract_eye_landmarks():
        # drawing eyes centers
        eyes = np.array(eye_landmark)
        l_eye, r_eye = eyes[:6], eyes[6:]
        l_eye_c, r_eye_c = (l_eye.sum(axis=0) / l_eye.shape[0]), (r_eye.sum(axis=0) / r_eye.shape[0])

        cv2.circle(normalisator.frame, tuple(l_eye_c.astype(int)), 1, (0, 0, 255), -1)
        cv2.circle(normalisator.frame, tuple(r_eye_c.astype(int)), 1, (0, 0, 255), -1)

def draw_eye_borders(normalisator):
    '''
    Draws rectangles around eyes
    :return: None
    '''
    l_eye, r_eye = normalisator.model_points[2], normalisator.model_points[3]
    normalisator.eye_height = 70
    four_points_plane = np.array([(l_eye[0], l_eye[1] - normalisator.eye_height, l_eye[2]),
                                  (r_eye[0], l_eye[1] - normalisator.eye_height, l_eye[2]),
                                  (r_eye[0], l_eye[1] + normalisator.eye_height, l_eye[2]),
                                  (l_eye[0], l_eye[1] + normalisator.eye_height, l_eye[2])], dtype="double")

    for face in normalisator.faces:
        (end_points2D, jacobian) = cv2.projectPoints(four_points_plane, face.rvec, face.tvec,
                                                     normalisator.calibration.matrix,
                                                     normalisator.calibration.distortion)
        cv2.line(normalisator.frame, (int(end_points2D[0][0][0]), int(end_points2D[0][0][1])),
                 (int(end_points2D[1][0][0]), int(end_points2D[1][0][1])), (255, 0, 0), 2)
        cv2.line(normalisator.frame, (int(end_points2D[1][0][0]), int(end_points2D[1][0][1])),
                 (int(end_points2D[2][0][0]), int(end_points2D[2][0][1])), (255, 0, 0), 2)
        cv2.line(normalisator.frame, (int(end_points2D[2][0][0]), int(end_points2D[2][0][1])),
                 (int(end_points2D[3][0][0]), int(end_points2D[3][0][1])), (255, 0, 0), 2)
        cv2.line(normalisator.frame, (int(end_points2D[3][0][0]), int(end_points2D[3][0][1])),
                 (int(end_points2D[0][0][0]), int(end_points2D[0][0][1])), (255, 0, 0), 2)

def draw_gazes(normalisator):

    for (i, face) in enumerate(normalisator.faces):
        left_eye_frame, right_eye_frame = normalisator.norm_eye_frames[i]

        # drawing left eye vector
        left_eye_center = np.array([150.0, -170.0, 135.0])
        left_eye_gaze = left_eye_center + 750 * face.gaze[0]
        end_points, _ = cv2.projectPoints(np.array([left_eye_center, left_eye_gaze]), face.rvec,
                                          face.tvec, normalisator.calibration.matrix,
                                          normalisator.calibration.distortion)
        cv2.line(normalisator.frame, (int(end_points[0][0][0]), int(end_points[0][0][1])),
                 (int(end_points[1][0][0]), int(end_points[1][0][1])), (255, 0, 0), 2)

        # drawing right eye vector
        right_eye_center = np.array([-150.0, -170.0, 135.0])
        right_eye_gaze = right_eye_center + 750 * face.gaze[1]
        end_points, _ = cv2.projectPoints(np.array([right_eye_center, right_eye_gaze * np.array([-1, 1, 1])]),
                                          face.rvec, face.tvec, normalisator.calibration.matrix,
                                          normalisator.calibration.distortion)
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

        cv2.circle(self.background, (x, y), 20, 100, -1)
        cv2.putText(self.background, str((x, y)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 100, 2)
