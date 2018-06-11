from app.actor import Actor
from dlib import shape_predictor
from dlib import get_frontal_face_detector
from dlib import rectangle as DlibRectangle
from dlib import rectangles as DlibRectangles
from scipy.io import loadmat
from cv2 import resize
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from cv2 import solvePnP, SOLVEPNP_ITERATIVE
from cv2 import Rodrigues
from cv2 import CascadeClassifier
from numpy import array
from numpy import tile
from numpy import concatenate
from numpy import copy


class ActorDetector:

    # landmarks_to_model = {31: 0,  # Nose tip
    #                            9: 1,  # Chin
    #                            37: 2,  # Right eye right corner
    #                            46: 3,  # Left eye left corner
    #                            49: 4,  # Right mouth corner
    #                            55: 5  # Left Mouth corner
    #                            }

    landmarks_to_model = [30, 8, 36, 45, 48, 54]

    def __init__(self, path_to_face_model, path_to_face_points, path_to_hc_model, factor, scale=1.3, minNeighbors=5,
                 chin_nose_distance=0.065):
        # self.detector = get_frontal_face_detector()
        self.detector = CascadeClassifier(path_to_hc_model).detectMultiScale
        self.scale = scale
        self.minNeighbors = minNeighbors
        self.predictor = shape_predictor(path_to_face_model)
        self.factor = factor
        self.model_points = loadmat(path_to_face_points)['model'] * array([-1, -1, 1])
        scale = chin_nose_distance / self.model_points[1, 1]
        self.model_points = self.model_points * scale
        self.eye_height = 60 * scale
        self.eye_width = 160 * scale
        self.nose_chin_distance = chin_nose_distance

    def rescale_coordinates(self, coords):
        return (coords * self.factor).astype(int)

    def downscale(self, image, **kwargs):
        return resize(image, (image.shape[1] // self.factor, image.shape[0] // self.factor), **kwargs)

    @staticmethod
    def to_grayscale(image):
        if image.ndim == 2:
            return image
        return cvtColor(image, COLOR_BGR2GRAY)

    @staticmethod
    def shape_to_np(shape, dtype='int'):
        return array([[shape.part(i).x, shape.part(i).y] for i in range(0, 68)], dtype=dtype)

    @staticmethod
    def cvface2dlibrects(cvfaces):
        return DlibRectangles([DlibRectangle(*cvface[:2], *(cvface[:2] + cvface[2:]))
                               for cvface in cvfaces])

    @classmethod
    def _extract_face_landmarks(cls, landmarks2D):
        """
        Extract 6 face landmarks from self.landmarks which corresponds to generic face model
        :return: None
        """
        if not landmarks2D:
            return []
        result = [face_landmarks2D[cls.landmarks_to_model] for face_landmarks2D in landmarks2D]
        return result

    @staticmethod
    def _vectors_from_model_to_origin(vectors, rotation_vector, translation_vector, camera):
        return camera.vectors_to_origin(Rodrigues(rotation_vector)[0] @ vectors.reshape(3, -1) + translation_vector)

    def _eye_rectangles(self, right_eye_model_space, left_eye_model_space):
        left_eye_rectangle_model_space = tile(left_eye_model_space, (4, 1))
        left_eye_rectangle_model_space[0:2, 1] = left_eye_rectangle_model_space[0:2, 1] - self.eye_height
        left_eye_rectangle_model_space[2:4, 1] = left_eye_rectangle_model_space[2:4, 1] + self.eye_height
        left_eye_rectangle_model_space[1:3, 0] = left_eye_rectangle_model_space[1:3, 0] + self.eye_width

        right_eye_rectangle_model_space = tile(right_eye_model_space, (4, 1))
        right_eye_rectangle_model_space[0:2, 1] = right_eye_rectangle_model_space[0:2, 1] - self.eye_height
        right_eye_rectangle_model_space[2:4, 1] = right_eye_rectangle_model_space[2:4, 1] + self.eye_height
        right_eye_rectangle_model_space[1:3, 0] = right_eye_rectangle_model_space[1:3, 0] - self.eye_width
        return array([right_eye_rectangle_model_space, left_eye_rectangle_model_space]).reshape(-1, 3)

    def detect_actors(self, frame, origin):
        image_for_detector = self.downscale(self.to_grayscale(frame.image))
        rectangles = self.cvface2dlibrects(self.detector(image_for_detector,
                                                         scaleFactor=self.scale,
                                                         minNeighbors=self.minNeighbors))

        landmarks2D = [self.rescale_coordinates(self.shape_to_np(self.predictor(image_for_detector, rectangle)))
                       for rectangle in rectangles]
        face_landmarks2D = self._extract_face_landmarks(landmarks2D)

        actors = []
        for i, face in enumerate(face_landmarks2D):
            actor_face_landamrks2D = array(face, dtype="double")
            success, rotation_vector, translation_vector = solvePnP(self.model_points,
                                                                    actor_face_landamrks2D,
                                                                    frame.camera.matrix,
                                                                    frame.camera.distortion,
                                                                    flags=SOLVEPNP_ITERATIVE)

            eye_centers_model_space = self.model_points[2:4] + array([[-self.eye_width/2, 0., 0.],
                                                                      [self.eye_width/2, 0., 0.]])
            eye_rectangles_model_space = self._eye_rectangles(*self.model_points[2:4])
            face_model_space = concatenate([self.model_points[:2],
                                            eye_centers_model_space,
                                            eye_rectangles_model_space], axis=0).T
            face_model_origin_space = self._vectors_from_model_to_origin(face_model_space,
                                                                         rotation_vector,
                                                                         translation_vector,
                                                                         frame.camera).T

            actor = Actor(name=f'Actor{i}', origin=origin)
            actor.set_dlib_landmarks3d(face_model_origin_space)
            actor.rotation = frame.camera.rotation + rotation_vector
            actor.nose_chin_distance = self.nose_chin_distance
            actors.append(actor)

        return actors
