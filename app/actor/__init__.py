from app.device import SceneObj
from numpy import array
from numpy import sqrt
from numpy import zeros
from numpy import cross


def face_point_to_array(dct):
    return array([dct['X'], dct['Y'], dct['Z']]).reshape(1, 3)


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
        return array([[x], [y], [z]])
    else:
        return zeros((3, 1))


class Actor(SceneObj):

    kinect_idx = {
        'LeyeO': 469,
        'LeyeI': 210,
        'ReyeO': 469,
        'ReyeI': 210,
        'nose': 18,
        'chin': 4
    }

    def __init__(self, name, origin, frame, rectangle=None, landmarks2D=None):
        super().__init__(name=name, origin=origin)
        self.frame = frame
        self.rectangle = rectangle
        self.landmarks3D = {}
        self.landmarks2D = landmarks2D
        self.gazes = None

    def set_landmarks3d(self, face_points_json):
        self.landmarks3D = {
            key: face_point_to_array(face_points_json[value])
            for key, value in self.kinect_idx.items()
        }
        self.landmarks3D['LeyeC'] = (self.landmarks3D['LeyeI'] + self.landmarks3D['LeyeO']) / 2
        self.landmarks3D['ReyeC'] = (self.landmarks3D['ReyeI'] + self.landmarks3D['ReyeO']) / 2
        return self

    def set_landmarks2d(self, shape_from_dlib):
        self.landmarks2D = shape_from_dlib
        return self

    def get_eye_landmarks2d(self):
        return self.landmarks2D[[37, 40] + [43, 46]].reshape(2, -1, 2)

    def get_eye_rectangle_coordinates(self, out_shape):
        return self.get_eye_landmarks2d().mean(axis=1, dtype=int) - out_shape[::-1].reshape(1, 2) // 2

    def set_rotation(self, face_rotation_quaternion):
        self.rotation = quaternion_to_angle_axis(face_rotation_quaternion)
        return self

    def set_translation(self, key='nose'):
        self.translation = self.landmarks3D[key]
        return self

    def get_norm_vector_to_face(self):
        return cross(self.landmarks3D['chin'] - self.landmarks3D['LeyeO'],
                     self.landmarks3D['chin'] - self.landmarks3D['ReyeO'])
