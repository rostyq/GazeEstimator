from .object import SceneObj
from numpy import array
from numpy import sqrt
from numpy import cross, dot
from numpy.linalg import solve

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

    line_point_1 = array(line_points[0]).reshape(3)
    line_point_2 = array(line_points[1]).reshape(3)
    plane_point_1 = array(plane_points[0]).reshape(3)
    plane_point_2 = array(plane_points[1]).reshape(3)
    plane_point_3 = array(plane_points[2]).reshape(3)

    # These two vectors are in the plane.
    vector_1 = plane_point_3 - plane_point_1
    vector_2 = plane_point_2 - plane_point_1

    # The cross prodaction is a normal vector to the plane.
    cross_product = cross(vector_1, vector_2)
    a, b, c = cross_product
    d = dot(cross_product, plane_point_3)

    # Compute the solution of equasion A*x = B.
    # Compute matrix A.
    A11 = 1 / (line_point_2[0] - line_point_1[0])
    A12 = -1 / (line_point_2[1] - line_point_1[1])
    A13 = 0
    A21 = 0
    A22 = 1 / (line_point_2[1] - line_point_1[1])
    A23 = -1 / (line_point_2[2] - line_point_1[2])
    A31, A32, A33 = a, b, c
    A = array([[A11, A12, A13],
                  [A21, A22, A23],
                  [A31, A32, A33]])

    # Compute vector B.
    B1 = line_point_1[0] * A11 + line_point_1[1] * A12
    B2 = line_point_1[1] * A22 + line_point_1[2] * A23
    B3 = d
    B = array([B1, B2, B3])
    # Compute intersection point.
    return solve(A, B)

class Screen(SceneObj):

    def __init__(self, name, screen_dict, extrinsic_matrix, origin=None, height=None, width=None):
        super().__init__(name=name, origin=origin, extrinsic_matrix=extrinsic_matrix)
        self.resolution = screen_dict['resolution']
        self.diagonal = screen_dict['diagonal']
        self.mpp = self.diagonal / sqrt(self.resolution[0] ** 2 + self.resolution[1] ** 2)
        self.height = height
        self.width = width
        if self.height is not None:
            self.pixel_height = self.height / self.resolution[0]
        else:
            self.pixel_height = None
        if self.width is not None:
            self.pixel_width = self.width / self.resolution[1]
        else:
            self.pixel_width = None

    # def calc_mpp(self):
    #     """
    #     mpp - Meters per pixels
    #     :return:
    #     """
    #     self.mpp = self.diagonal / sqrt(self.resolution[0]**2+self.resolution[1]**2)

    def get_point_in_pixels(self, x, y):
        if not self.mpp:
            self.calc_mpp()
        return array([int(coord*axis*self.mpp) for coord, axis in zip((x, y), self.resolution)])

    def point_to_origin(self, x, y):
        point = array([coord*axis*self.mpp for coord, axis in zip((x, y), self.resolution)]+[0.0]).reshape((3, 1))
        return self.get_rotation_matrix() @ point - self.translation

    def point_in_pixels(self, x, y):
        if self.width is not None and self.height is not None:
            pixel_x, pixel_y = round(x / self.pixel_height), round(y/ self.pixel_width)
        else:
            pixel_x, pixel_y = None, None
        return pixel_x, pixel_y

if __name__ == '__main__':
    print('Plane intersection example:\n'
          'Plane = ((2,0,0), (0,-4,0), (0,0,4))\n'
          'Line = ((1, 3, -1), (3,4,2))\n'
          'Right answer: [3, 4, 2]\n')
    print(plane_line_intersection(((1, 3, -1), (3,4,2)), ((2,0,0), (0,-4,0), (0,0,4))))

    print('Point in pixels:\n'
          'height = 100 mm\n'
          'width = 200 mm\n'
          'resolution = (1000, 2000)\n'
          'point = (10.123, 20,123)\n')

    new_screen = Screen('Dell', origin=None, matrix=None, distortion=None,
                 diagonal=sqrt(5) * 100, resolution=(1000, 2000), height=100, width=200)
    print(new_screen.point_in_pixels(*(10.123, 20.123)))
