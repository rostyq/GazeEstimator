from .camera import *
from .object import *
from .screen import *
from numpy import zeros


def create_origin_cam(name, matrix, distortion):
    origin = Camera(name=name, origin=None)
    origin.rotation = zeros((3,), dtype='float')
    origin.translation = zeros((3,), dtype='float')
    origin.matrix = matrix
    origin.distortion = distortion
    return origin


def from_extrinsic_matrix(name, matrix, relative=None):
    matrix = array(matrix)
    obj = SceneObj(name=name, origin=relative)
    obj.translation = matrix[:3, 3]
    obj.rotation = Rodrigues(matrix[:3, :3])[0]
    return obj


# def cam_from_dict(name, cam_dict, origin=None):
#     cam = Camera(name=name, origin=origin)
#     # TODO check rotation and translation
#     for key, value in cam_dict.items():
#         cam.__setattr__(key, array(value))
#     return cam


# def screen_from_dict(name, screen_dict, origin=None):
#     screen = Screen(name=name, screen_dict=screen_dict, origin=origin)
#     # TODO check rotation and translation
#
#     return screen
