from .object import SceneObj
from numpy import array


class Camera(SceneObj):

    def __init__(self, name, cam_dict, extrinsic_matrix=None, origin=None):
        super().__init__(name=name, origin=origin, extrinsic_matrix=extrinsic_matrix)
        self.matrix = array(cam_dict['matrix'])
        self.distortion = array(cam_dict['distortion'])

    def to_dict(self):
        result = super().to_dict()
        result['matrix'] = self.matrix.tolist()
        result['distortion'] = self.distortion.tolist()
        return result

    def get_intrinsic(self):
        return {'matrix': self.matrix, 'distortion': self.distortion}