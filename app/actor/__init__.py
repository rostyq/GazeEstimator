from numpy import cross
from numpy import array
from numpy import sum
from numpy import abs
from numpy.linalg import norm
from numpy.linalg import inv

from app.device import SceneObj
from app.parser import quaternion_to_angle_axis

from scipy.optimize import minimize


class Person(SceneObj):

    right_eye_center = [843, 1097, 1095, 1096, 1091, 1090, 1092, 1099, 1094, 1065, 1100, 1101, 1102, 992, 846, 777,
                        776, 728, 731, 873, 733, 876, 749, 752, 992]

    left_eye_center = [210, 1111, 1109, 1108, 1103, 1104, 1105, 1112, 1106, 1107, 1113, 1114, 1115, 1116, 188, 211,
                       137, 238, 244, 241, 121, 153, 187, 316]

    def __init__(self, name, origin):
        super().__init__(name=name, origin=origin)

        # landmarks in real space
        self.landmarks_3d = {
            'eyes': {
                'left': {
                    'rectangle': None,
                    'center': None,
                    'gaze': None
                },
                'right': {
                    'rectangle': None,
                    'center': None,
                    'gaze': None
                }
            },
            'nose': None,
            'chin': None
        }

        # raw data from dlib
        self.raw_dlib_landmarks = None

        # average parameters of face
        self.nose_chin_distance = None
        self.eyeball_radius = 0.012

    def get_eye_rectangle(self, eye):
        return self.landmarks_3d['eyes'][eye]['rectangle']

    def get_eye_gaze(self, eye):
        return self.landmarks_3d['eyes'][eye]['gaze']

    def get_eye_center(self, eye):
        return self.landmarks_3d['eyes'][eye]['center']

    def get_nose(self):
        return self.landmarks_3d['nose']

    def get_chin(self):
        return self.landmarks_3d['chin']

    def set_eye_rectangle(self, eye, rectangle):
        self.landmarks_3d['eyes'][eye]['rectangle'] = rectangle

    def set_eye_gaze(self, eye, gaze_vector):
        self.landmarks_3d['eyes'][eye]['gaze'] = gaze_vector

    def set_eye_center(self, eye, landmark):
        self.landmarks_3d['eyes'][eye]['center'] = landmark

    def set_nose(self, landmark):
        self.landmarks_3d['nose'] = landmark

    def set_chin(self, landmark):
        self.landmarks_3d['chin'] = landmark

    def get_norm_gaze(self, eye, camera):

        eye_gaze = self.get_eye_gaze(eye=eye)
        eye_gaze = eye_gaze / norm(eye_gaze)

        return inv(camera.get_rotation_matrix()) @ eye_gaze.reshape(3, -1)

    def get_norm_rotation(self, camera):
        return inv(camera.get_rotation_matrix()) @ self.get_face_gaze().reshape(3, -1)

    def to_learning_dataset(self, img_left_name, img_right_name, camera):
        return {
            'eyes': {
                eye: {
                    'gaze_norm': self.get_norm_gaze(eye, camera).tolist(),
                    'image': img_left_name if eye is 'left' else img_right_name,
                    'center': camera.vectors_to_self(self.get_eye_center(eye)).tolist()
                } for eye in ['left', 'right']
            },
            'rotation_norm': self.get_norm_rotation(camera).tolist(),
            'nose_chin_distance': self.nose_chin_distance,
            'name': self.name
        }

    def set_kinect_landmarks3d(self, face_points):
        face_points = array(face_points)

        # Function for OLS
        def objective_function(center, eye):
            return sum(abs(norm(center - face_points[eye, :], axis=1) - self.eyeball_radius))

        self.set_eye_rectangle('left', face_points[[1080, 201, 289, 151]])
        self.set_eye_rectangle('right', face_points[[1084, 847, 947, 772]])
        self.set_eye_center('left', minimize(lambda x: objective_function(x, self.left_eye_center),
                                             x0=face_points[self.left_eye_center[0]]).x)
        self.set_eye_center('right', minimize(lambda x: objective_function(x, self.right_eye_center),
                                              x0=face_points[self.right_eye_center[0]]).x)
        self.set_nose(face_points[18])
        self.set_chin(face_points[4])
        return self

    def set_dlib_landmarks3d(self, face_model_origin_space):

        nose_origin_space = face_model_origin_space[0].reshape(3)
        chin_origin_space = face_model_origin_space[1].reshape(3)

        right_eye_center_origin_space = face_model_origin_space[2].reshape(3)
        left_eye_center_origin_space = face_model_origin_space[3].reshape(3)

        right_eye_rectangle_origin_space = face_model_origin_space[4:8]
        left_eye_rectangle_origin_space = face_model_origin_space[8:12]

        self.set_eye_rectangle('left', left_eye_rectangle_origin_space)
        self.set_eye_rectangle('right', right_eye_rectangle_origin_space)

        self.set_nose(nose_origin_space)
        self.set_chin(chin_origin_space)

        self.set_eye_center('left', left_eye_center_origin_space)
        self.set_eye_center('right', right_eye_center_origin_space)

    def set_landmarks3d_gazes(self, gazes, screen):

        left_gaze_point_origin_space = screen.point_to_origin(*gazes['left']).reshape(3)
        right_gaze_point_origin_space = screen.point_to_origin(*gazes['right']).reshape(3)

        self.set_eye_gaze('left', left_gaze_point_origin_space - self.get_eye_center('left'))
        self.set_eye_gaze('right', right_gaze_point_origin_space - self.get_eye_center('right'))

    def set_rotation(self, face_rotation_quaternion):
        self.rotation = quaternion_to_angle_axis(face_rotation_quaternion)
        return self

    def set_translation(self, key='nose'):
        self.translation = self.landmarks_3d[key]
        return self

    def get_face_gaze(self):

        cross_vec = cross(self.get_chin() - self.get_eye_center('left'),
                          self.get_chin() - self.get_eye_center('right'))

        return cross_vec / norm(cross_vec)

    def get_gaze_line(self, gaze_vector, key):
        return self.get_eye_center(key) + gaze_vector.reshape(3), self.get_eye_center(key)

    def set_gazes_to_mark(self, point):
        for eye in ['left', 'right']:
            gaze = point - self.get_eye_center(eye)
            gaze = gaze / norm(gaze)
            self.set_eye_gaze(eye=eye, gaze_vector=gaze)
        return self
