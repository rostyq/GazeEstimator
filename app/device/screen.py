from .object import SceneObj
from numpy import array
from numpy import sqrt


class Screen(SceneObj):

    def __init__(self, name, origin=None, matrix=None, distortion=None, diagonal=None, resolution=None):
        super().__init__(name=name, origin=origin, extrinsic_matrix=extrinsic_matrix)
        self.resolution = screen_dict['resolution']
        self.diagonal = screen_dict['diagonal']
        self.mpp = self.diagonal / sqrt(self.resolution[0] ** 2 + self.resolution[1] ** 2)
        self.resolution = resolution
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
        point = array([coord*axis*self.mpp for coord, axis in zip((x, y), self.resolution)]+[0.0])
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
