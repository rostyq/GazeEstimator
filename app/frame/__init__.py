from numpy import copy
from numpy import array
from cv2 import projectPoints
from cv2 import circle
from cv2 import findHomography
from cv2 import warpPerspective

class Frame:

    flip_y_array = array([[1, -1, 1]])

    def __init__(self, camera, image):
        self.camera = camera
        self.image = image.astype('uint8')

    def draw_points(self, points, radius=10, color=(255, 0, 0)):
        for point in points:
            circle(self.image, tuple(point.astype(int)), radius, color, -1)
        return self

    def get_projected_coordinates(self, vectors):
        return projectPoints(vectors, #*self.flip_y_array,
                             self.camera.rotation,
                             self.camera.translation,
                             self.camera.matrix,
                             self.camera.distortion)[0].reshape(-1, 2)

    def project_vectors(self, vectors, **kwargs):
        self.draw_points(self.get_projected_coordinates(vectors).astype(int), **kwargs)
        return self

    def extract_rectangle(self, coord, shape):
        """

        :param coord: Left-upper corner.
        :param shape: Tuple (height, width)
        :return:
        """
        return self.image[coord[0]:coord[0]+shape[0], coord[1]:coord[1]+shape[1]]

    def extract_eyes_from_actor(self, actor, resolution=(60, 36)):
        # eye planes
        left_norm_image_plane = array([[resolution[0], 0.0          ],
                                        [0.0,           0.0          ],
                                        [0.,            resolution[1]],
                                        [resolution[0], resolution[1]]])
        right_norm_image_plane = array([[0.0,          0.0          ],
                                         [resolution[0], 0.0         ],
                                         [resolution[0], resolution[1]],
                                         [0.,            resolution[1]]])
        left_eye_projection = self.get_projected_coordinates(actor.landmarks3D['eyes']['left']['rectangle'])
        right_eye_projection = self.get_projected_coordinates(actor.landmarks3D['eyes']['right']['rectangle'])

        homography, status = findHomography(left_eye_projection, left_norm_image_plane)
        left_eye_frame = warpPerspective(self.image, homography, resolution)

        homography, status = findHomography(right_eye_projection, right_norm_image_plane)
        right_eye_frame = warpPerspective(self.image, homography, resolution)

        return left_eye_frame, right_eye_frame
