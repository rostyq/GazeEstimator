from .object import SceneObj


class Camera(SceneObj):

    def __init__(self, name, origin=None):
        super().__init__(name=name, origin=origin)
        self.matrix = None
        self.distortion = None

    def get_intrinsic(self):
        return {'matrix': self.matrix, 'distortion': self.distortion}