from .object import SceneObj
from numpy import array
from numpy import sqrt


class Screen(SceneObj):

    def __init__(self, name, screen_dict, extrinsic_matrix, origin=None):
        super().__init__(name=name, origin=origin, extrinsic_matrix=extrinsic_matrix)
        self.resolution = screen_dict['resolution']
        self.diagonal = screen_dict['diagonal']
        self.mpp = self.diagonal / sqrt(self.resolution[0] ** 2 + self.resolution[1] ** 2)

        # for key, value in screen_dict.items():
        #     self.__setattr__(key, array(value) if isinstance(value, list) else value)


    # def calc_mpp(self):
    #     """
    #     mpp - Meters per pixels
    #     :return:
    #     """
    #     self.mpp = self.diagonal / sqrt(self.resolution[0]**2+self.resolution[1]**2)
    #     return self

    def get_point_in_meters(self, x, y):
        return [coord*axis*self.mpp for coord, axis in zip((x, y), self.resolution)]

    def point_to_origin(self, x, y):
        return self.get_rotation_matrix() @ array(self.get_point_in_meters(x, y) + [0.0]).reshape(3, 1) - self.translation.reshape(3, 1)
