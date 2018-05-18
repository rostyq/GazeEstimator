from object import SceneObj
from numpy import array
from numpy import sqrt
import numpy as np


def plane_line_intersection(line_points, plane_points):
    ''' Compute intersection point of plane and lineself.

    Parameter line_points consists of two points and stands to determine
    line's equesion:
        (x - x_1)/(x_2 - x_1) =
       =(y - y_1)/(y_2 - y_1) =
       =(z - z_1)/(z_2 - z_1).
    Parameter plane_points consists of three points and stands to determine
    plane's equasion:
        A*x + B*y + C*z = D.
    This function returns 3D coordinates of intersection point.
    '''

    line_point_1 = np.array(line_points[0]).reshape(3)
    line_point_2 = np.array(line_points[1]).reshape(3)
    plane_point_1 = np.array(plane_points[0]).reshape(3)
    plane_point_2 = np.array(plane_points[1]).reshape(3)
    plane_point_3 = np.array(plane_points[2]).reshape(3)

    # These two vectors are in the plane.
    vector_1 = plane_point_3 - plane_point_1
    vector_2 = plane_point_2 - plane_point_1

    # The cross prodaction is a normal vector to the plane.
    cross_product = np.cross(vector_1, vector_2)
    a, b, c = cross_product
    d = np.dot(cross_product, plane_point_3)

    # Compute the solution of equasion A*x = B.
    # Compute matrix A.
    A11 = 1 / (line_point_2[0] - line_point_1[0])
    A12 = -1 / (line_point_2[1] - line_point_1[1])
    A13 = 0
    A21 = 0
    A22 = 1 / (line_point_2[1] - line_point_1[1])
    A23 = -1 / (line_point_2[2] - line_point_1[2])
    A31 = a
    A32 = b
    A33 = c
    A = np.array([[A11, A12, A13],
                  [A21, A22, A23],
                  [A31, A32, A33]])

    # Compute vector B.
    B1 = line_point_1[0] * A11 + line_point_1[1] * A12
    B2 = line_point_1[1] * A22 + line_point_1[2] * A23
    B3 = d
    B = np.array([B1, B2, B3])
    # Compute intersection point.
    return np.linalg.solve(A, B)


class Screen(SceneObj):
    def __init__(self, name, origin=None, matrix=None, distortion=None,
                 diagonal=None, resolution=None, height=None, width=None):
        super().__init__(name=name, origin=origin)
        self.matrix = matrix
        self.distortion = distortion
        self.diagonal = diagonal
        self.resolution = resolution
        self.height = height
        self.width = width
        self.mpp = None

    def calc_mpp(self):
        """
        mpp - Meters per pixels
        :return:
        """
        self.mpp = self.diagonal / sqrt(self.resolution[0]**2+self.resolution[1]**2)

    def get_point_in_pixels(self, x, y):
        if not self.mpp:
            self.calc_mpp()
        return array([int(coord*axis*self.mpp) for coord, axis in zip((x, y), self.resolution)])

    def point_to_origin(self, x, y):
        point = array([coord*axis*self.mpp for coord, axis in zip((x, y), self.resolution)]+[0.0])
        return self.get_rotation_matrix() @ point - self.translation



if __name__ == '__main__':
    print(plane_line_intersection(((1, 3, -1), (3,4,2)), ((2,0,0), (0,-4,0), (0,0,4))))
