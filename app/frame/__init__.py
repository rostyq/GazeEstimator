from numpy import copy
from numpy import array
from cv2 import projectPoints
from cv2 import circle


class Frame:

    flip_y_array = array([[1, -1, 1]])

    def __init__(self, camera, image):
        self.camera = camera
        self.image = image.astype('uint8')

    def draw_points(self, points, radius=10, color=(255, 0, 0)):
        for point in points:
            circle(self.image, tuple(point.astype(int)), radius, color, -1)

    def get_projected_coordinates(self, vectors):
        return projectPoints(vectors*self.flip_y_array,
                             self.camera.rotation,
                             self.camera.translation,
                             self.camera.matrix,
                             self.camera.distortion)[0].reshape(-1, 2).astype(int)

    def project_vectors(self, vectors, **kwargs):
        self.draw_points(self.get_projected_coordinates(vectors), **kwargs)
        return self

    def extract_rectangle(self, coord, shape):
        """

        :param coord: Left-upper corner.
        :param shape: Tuple (height, width)
        :return:
        """
        return self.image[coord[0]:coord[0]+shape[0], coord[1]:coord[1]+shape[1]]

    def extract_eyes_from_actor(self, actor, shifts=(18, 60)):
        Lstart = self.get_projected_coordinates(actor.landmarks['LeyeI']).flatten()[::-1]
        Rstart = self.get_projected_coordinates(actor.landmarks['ReyeI']).flatten()[::-1]
        return self.image[Lstart[0]:Lstart[0] + shifts[0], Lstart[1]-shifts[1]:Lstart[1]], \
               self.image[Rstart[0]:Rstart[0] + shifts[0], Rstart[1]:Rstart[1]+ shifts[1]]
