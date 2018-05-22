from numpy import array
from numpy import hstack
from numpy import vstack
from numpy import zeros
from cv2 import Rodrigues
from numpy.linalg import inv


class SceneObj:

    to_m = {
        'mm': 1000
    }

    def __init__(self, name, origin, extrinsic_matrix=None, scale='mm'):
        self.name = name
        self.origin = origin
        if extrinsic_matrix is None:
            self.translation = zeros((3, 1))
            self.rotation = zeros((3, 1))
        else:
            extrinsic_matrix = array(extrinsic_matrix)
            self.translation = (extrinsic_matrix[:3, 3] / self.to_m[scale]).reshape(3, 1)
            self.rotation = (Rodrigues(extrinsic_matrix[:3, :3])[0]).reshape(3, 1)
        self.rotation_matrix = Rodrigues(self.rotation)[0]

    def get_rotation_matrix(self):
        # if self.rotation_matrix is None:
        #     self.rotation_matrix = Rodrigues(self.rotation)[0]
        return self.rotation_matrix

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

    # def set_extrinsic_from_matrix(self, matrix, scale='mm'):
    #     matrix = array(matrix)
    #     self.translation = (matrix[:3, 3] / self.to_m[scale]).reshape(3, 1)
    #     self.rotation = (Rodrigues(matrix[:3, :3])[0]).reshape(3, 1)
    #     return self

    def vectors_to_origin(self, vectors):
        return inv(self.get_rotation_matrix()) @ (vectors.reshape(3, -1) - self.translation.reshape(3, 1))

    def vectors_to_self(self, vectors):
        return self.get_rotation_matrix() @ (vectors.reshape(3, -1) + self.translation.reshape(3, 1))

