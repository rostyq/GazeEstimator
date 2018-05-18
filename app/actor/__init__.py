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

    idx = {
        'LeyeO': 469,
        'LeyeI': 210,
        'ReyeO': 469,
        'ReyeI': 210,
        'nose': 18,
        'chin': 4
    }

    def __init__(self, name, origin, frame):
        super().__init__(name=name, origin=origin)
        self.frame = frame
        self.rectangle = None
        self.landmarks = None
        self.gazes = None

    def set_landmarks(self, face_points_json):
        self.landmarks = {
            key: face_point_to_array(face_points_json[value])
            for key, value in self.idx.items()
        }
        self.landmarks['LeyeC'] = (self.landmarks['LeyeI'] + self.landmarks['LeyeO']) / 2
        self.landmarks['ReyeC'] = (self.landmarks['ReyeI'] + self.landmarks['ReyeO']) / 2
        return self

    def set_rotation(self, FaceRotationQuaternion):
        self.rotation = quaternion_to_angle_axis(FaceRotationQuaternion)
        return self

    def set_translation(self, key='nose'):
        self.translation = self.landmarks[key]
        return self

    def get_norm_vector_to_face(self):
        v1 = self.landmarks['chin'] - self.landmarks['LeyeO']
        v2 = self.landmarks['chin'] - self.landmarks['ReyeO']
        return cross(v1, v2)
