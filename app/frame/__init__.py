from numpy import copy
from numpy import copy
from cv2 import projectPoints
from cv2 import imshow
from cv2 import circle


class Frame:

    def __init__(self, camera, image):
        self.camera = camera
        self.image = image

    def draw_points(self, points, radius=10, color=(0, 255, 0)):
        for point in points:
            circle(self.image, tuple(point.astype(int)), radius, color, -1)

    def get_projected_coordinates(self, vectors):
        return projectPoints(vectors,
                             self.camera.rotation,
                             self.camera.translation,
                             self.camera.matrix,
                             self.camera.distortion)[0].reshape(-1, 2)

    def project_vectors(self, vectors, **kwargs):
        self.draw_points(self.get_projected_coordinates(vectors), **kwargs)
        return self

    def extract_eyes_from_actor(self, actor, shifts=(18, 60)):
        Lstart = self.get_projected_coordinates(actor.landmarks['LeyeI'])
        Rstart = self.get_projected_coordinates(actor.landmarks['ReyeI'])
        return self.image[Lstart - shifts[0]:Lstart + shifts[0], Lstart:Lstart + shifts[1]], \
               self.image[Rstart - shifts[0]:Rstart + shifts[0], Rstart:Rstart - shifts[1]]

    def show(self):
        imshow(__name__, self.image)
