from numpy import array
from numpy import hstack
from numpy import vstack
from cv2 import Rodrigues
from numpy.linalg import inv


class SceneObj:

    def __init__(self, name, origin, translation=None, rotation=None):
        self.name = name
        self.origin = origin
        self.translation = translation
        self.rotation = rotation

    def get_rotation_matrix(self):
        return Rodrigues(self.rotation)[0]

    def get_extrinsic(self):
        return {'rotation': self.rotation, 'translation': self.translation}

    def restore_extrinsic_matrix(self):
        return vstack(
            (
                hstack(
                    (
                        Rodrigues(self.rotation)[0],
                        self.translation.reshape(3, 1)
                    )
                ),
                array([0.0, 0.0, 0.0, 1.0])
            )
        )

    def set_extrinsic_from_matrix(self, matrix):
        matrix = array(matrix).reshape(4, 4)
        self.translation = matrix[:3, 3]
        self.rotation = Rodrigues(matrix[:3, :3])[0]
        return self

    def vector_to_origin(self, vector):
        return inv(self.get_rotation_matrix()) @ (vector - self.translation)

    def vector_to_self(self, vector):
        return inv(self.get_rotation_matrix()) @ (vector + self.translation)

