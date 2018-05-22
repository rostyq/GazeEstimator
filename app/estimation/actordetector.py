from app.actor import Actor
from dlib import get_frontal_face_detector
from dlib import shape_predictor
from scipy.io import loadmat
from cv2 import resize
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from cv2 import solvePnP, SOLVEPNP_ITERATIVE
from cv2 import Rodrigues
from numpy import array
from numpy.linalg import inv


class ActorDetector:

    def __init__(self, path_to_face_model, path_to_face_points, factor):
        self.detector = get_frontal_face_detector()
        self.predictor = shape_predictor(path_to_face_model)
        self.factor = factor
        self.eye_height = 70 * 0.0001
        self.eye_width = 200 * 0.0001
        self.model_points = loadmat(path_to_face_points)['model'] * array([1, -1, 1]) * 0.0001
        self.landmarks_to_model = {31: 0,  # Nose tip
                                   9: 1,  # Chin
                                   37: 2,  # Left eye left corner
                                   46: 3,  # Right eye right corner
                                   49: 4,  # Left Mouth corner
                                   55: 5  # Right mouth corner
                                   }

    def rescale_coordinates(self, coords):
        return (coords * self.factor).astype(int)

    def downscale(self, image, **kwargs):
        return resize(image, (image.shape[1] // self.factor, image.shape[0] // self.factor), **kwargs)

    @staticmethod
    def to_grayscale(image):
        return cvtColor(image, COLOR_BGR2GRAY)

    @staticmethod
    def shape_to_np(shape, dtype='int'):
        return array([[shape.part(i).x, shape.part(i).y] for i in range(0, 68)], dtype=dtype)

    def _extract_face_landmarks(self, landmarks2D):
        """
        Extract 6 face landmarks from self.landmarks which corresponds to generic face model
        :return: None
        """
        if len(landmarks2D) == 0:
            return []
        faces_landmarks = [[None] * 6] * len(landmarks2D)
        for (k, face_landmarks2D) in enumerate(landmarks2D):
            for (j, (x, y)) in enumerate(face_landmarks2D):
                if j + 1 in self.landmarks_to_model.keys():
                    faces_landmarks[k][self.landmarks_to_model[j + 1]] = (x, y)
        return faces_landmarks

    def _vectors_from_model_to_origin(self, vectors, rotation_vector, translation_vector, camera):
        rotation_matrix = Rodrigues(rotation_vector)[0]
        vectors_camera_space = inv(rotation_matrix) @ (vectors - translation_vector)
        return camera.vectors_to_origin(vectors_camera_space)

    def detect_actors(self, frame, origin):
        image_for_detector = self.downscale(self.to_grayscale(frame.image))
        rectangles = self.detector(image_for_detector)
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
            left_eye_model_space, right_eye_model_space = self.model_points[2], self.model_points[3]
            nose_model_space, chin_model_space = self.model_points[0], self.model_points[1]

            left_eye_rectangle_model_space = array(
                [[left_eye_model_space[0],                  left_eye_model_space[1] - self.eye_height, left_eye_model_space[2]],
                 [left_eye_model_space[0] + self.eye_width, left_eye_model_space[1] - self.eye_height, left_eye_model_space[2]],
                 [left_eye_model_space[0] + self.eye_width, left_eye_model_space[1] + self.eye_height, left_eye_model_space[2]],
                 [left_eye_model_space[0],                  left_eye_model_space[1] + self.eye_height, left_eye_model_space[2]]]).T
            right_eye_rectangle_model_space = array(
                [[right_eye_model_space[0],                  right_eye_model_space[1] - self.eye_height, right_eye_model_space[2]],
                 [right_eye_model_space[0] - self.eye_width, right_eye_model_space[1] - self.eye_height, right_eye_model_space[2]],
                 [right_eye_model_space[0] - self.eye_width, right_eye_model_space[1] + self.eye_height, right_eye_model_space[2]],
                 [right_eye_model_space[0],                  right_eye_model_space[1] + self.eye_height, right_eye_model_space[2]]]).T

            left_eye_rectangle_origin_space = self._vectors_from_model_to_origin(left_eye_rectangle_model_space,
                                                                                 rotation_vector,
                                                                                 translation_vector,
                                                                                 frame.camera)
            right_eye_rectangle_origin_space = self._vectors_from_model_to_origin(right_eye_rectangle_model_space,
                                                                                  rotation_vector,
                                                                                  translation_vector,
                                                                                  frame.camera)
            nose_origin_space = self._vectors_from_model_to_origin(nose_model_space.T,
                                                                   rotation_vector,
                                                                   translation_vector,
                                                                   frame.camera)
            chin_origin_space = self._vectors_from_model_to_origin(chin_model_space.T,
                                                                   rotation_vector,
                                                                   translation_vector,
                                                                   frame.camera)

            actor = Actor(name=f'Actor{i}', origin=origin)
            actor.set_landmarks3d_eye_rectangles(left_eye_rectangle_origin_space.T, right_eye_rectangle_origin_space.T)
            actor.set_landmarks3d_nose_chin(nose_origin_space, chin_origin_space)
            actor.rotation = rotation_vector
            actors.append(actor)

        return actors
