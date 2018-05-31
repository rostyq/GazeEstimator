from .object import SceneObj
from numpy import array
from numpy import sqrt
from numpy import cross, dot
from numpy import zeros
from numpy import uint8
from numpy.linalg import solve
from cv2 import copyMakeBorder, BORDER_CONSTANT
from app.frame import Frame

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

    def to_dict(self):
        result = super().to_dict()
        result['resolution'] = self.resolution
        result['diagonal'] = self.diagonal
        return result

    def get_point_in_pixels(self, x, y):
        return array([int(coord*axis) for coord, axis in zip((x, y), self.resolution)])

    def point_to_origin(self, x, y):
        point = array([coord*axis*self.mpp for coord, axis in zip((y, x), self.resolution)]+[0.0]).reshape((3, 1))
        return self.vectors_to_origin(point)

    def get_intersection_point_in_pixels(self, line_points_origin):
        intersection_origin = self.get_intersection_point_origin(line_points_origin)
        intersection_self = self.vectors_to_self(intersection_origin).reshape(3)
        return (intersection_self[1]/self.mpp, intersection_self[0]/self.mpp)

    def get_intersection_point_origin(self, line_points_origin):
        wall_points_origin = array([self.point_to_origin(0, 0), self.point_to_origin(1, 1), self.point_to_origin(1, 0)])
        intersection_origin = plane_line_intersection(line_points_origin, wall_points_origin)
        return intersection_origin

    def generate_image_with_circles(self, points, padding=10, labels=None, colors=None):
        image = zeros((self.resolution[0], self.resolution[1], 3), dtype=uint8)
        image = copyMakeBorder(image, padding, padding, padding, padding, BORDER_CONSTANT)
        image[:, padding:padding+3], \
        image[:, -3-padding:-padding], \
        image[padding:padding+3, :], \
        image[-3-padding:-padding, :] = [255] * 4
        Frame.draw_points(image, points+padding, radius=40)
        if labels:
            Frame.draw_labels(image, labels, points+padding, colors)
        return image
