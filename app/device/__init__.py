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


def create_screen(name, extrinsic_matrix, diagonal, resolution, relative=None):
    extrinsic_matrix = array(extrinsic_matrix).reshape(4, 4)
    screen = Screen(name=name, origin=relative)
    screen.translation = extrinsic_matrix[:3, 3]
    screen.rotation = Rodrigues(extrinsic_matrix[:3, :3])[0]
    screen.resolution = resolution
    screen.diagonal = diagonal
    return screen


def from_extrinsic_matrix(name, matrix, relative=None):
    matrix = array(matrix).reshape(4, 4)
    obj = SceneObj(name=name, origin=relative)
    obj.translation = matrix[:3, 3]
    obj.rotation = Rodrigues(matrix[:3, :3])[0]
    return obj


def cam_from_dict(name, cam_dict, relative=None):
    cam = Camera(name=name, origin=relative)
    # TODO check rotation and translation
    cam.rotation = array(cam_dict['rotation']).reshape((3,))
    cam.translation = array(cam_dict['translation']).reshape((3,))
    cam.matrix = array(cam_dict['matrix']).reshape((3, 3))
    cam.distortion = array(cam_dict['distortion']).reshape((4,))
    return cam