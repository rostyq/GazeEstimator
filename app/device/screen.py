from .object import SceneObj
from numpy import array
from numpy import sqrt


class Screen(SceneObj):

    def __init__(self, name, origin=None, matrix=None, distortion=None, diagonal=None, resolution=None):
        super().__init__(name=name, origin=origin)
        self.matrix = matrix
        self.distortion = distortion
        self.diagonal = diagonal
        self.resolution = resolution
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
