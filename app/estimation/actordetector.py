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


class ActorDetector:

    # landmarks_to_model = {31: 0,  # Nose tip
    #                            9: 1,  # Chin
    #                            37: 2,  # Right eye right corner
    #                            46: 3,  # Left eye left corner
    #                            49: 4,  # Right mouth corner
    #                            55: 5  # Left Mouth corner
    #                            }

    landmarks_to_model = [31, 9, 37, 46, 49, 55]

    def __init__(self, path_to_face_model, path_to_face_points, factor, chin_nose_distance=0.065):
        self.detector = get_frontal_face_detector()
#     def __init__(self, path_to_face_model, path_to_face_points, path_to_hc_model, factor, scale=1.3, minNeighbors=5):
#         # self.detector = get_frontal_face_detector()
#         self.detector = CascadeClassifier(path_to_hc_model).detectMultiScale
#         self.scale = scale
#         self.minNeighbors = minNeighbors
        self.predictor = shape_predictor(path_to_face_model)
        self.factor = factor
        self.model_points = loadmat(path_to_face_points)['model'] * array([-1, -1, 1])
        scale = chin_nose_distance / self.model_points[1, 1]
        self.model_points = self.model_points * scale
        self.eye_height = 60 * scale
        self.eye_width = 160 * scale

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
        # faces_landmarks = [[None] * 6] * len(landmarks2D)
        # for (k, face_landmarks2D) in enumerate(landmarks2D):
        #     for (j, (x, y)) in enumerate(face_landmarks2D):
        #         if j + 1 in cls.landmarks_to_model.keys():
        #             faces_landmarks[k][cls.landmarks_to_model[j + 1]] = (x, y)
        # return faces_landmarks
        return [face_landmarks2D[cls.landmarks_to_model] for face_landmarks2D in landmarks2D]

    @staticmethod
    def _vectors_from_model_to_origin(vectors, rotation_vector, translation_vector, camera):
        return camera.vectors_to_origin(Rodrigues(rotation_vector)[0] @ vectors.reshape(3, -1) + translation_vector)

    def _eye_rectagle(self, eye_model_space, right=True):
        eye_rectangle_model_space = tile(eye_model_space, (4, 1))
        eye_rectangle_model_space[0:2, 1] = eye_rectangle_model_space[0:2, 1] - self.eye_height
        eye_rectangle_model_space[2:4, 1] = eye_rectangle_model_space[2:4, 1] + self.eye_height
        if right:
            eye_rectangle_model_space[1:3, 0] = eye_rectangle_model_space[1:3, 0] - self.eye_width
        else:
            eye_rectangle_model_space[1:3, 0] = eye_rectangle_model_space[1:3, 0] + self.eye_width
        return eye_rectangle_model_space.T

    def detect_actors(self, frame, origin):
        image_for_detector = self.downscale(self.to_grayscale(frame.image))
        rectangles = self.detector(image_for_detector)
        # rectangles = self.cvface2dlibrects(self.detector(image_for_detector,
        #                                                  scaleFactor=self.scale,
        #                                                  minNeighbors=self.minNeighbors))

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
            left_eye_model_space, right_eye_model_space = self.model_points[3], self.model_points[2]
            nose_model_space, chin_model_space = self.model_points[0], self.model_points[1]
            left_eye_center_model_space = left_eye_model_space + array([self.eye_width/2, 0., 0.])
            right_eye_center_model_space = right_eye_model_space + array([-self.eye_width/2, 0., 0.])

            left_eye_rectangle_model_space = self._eye_rectagle(left_eye_model_space, right=False)
            right_eye_rectangle_model_space = self._eye_rectagle(right_eye_model_space, right=True)

            left_eye_rectangle_origin_space = self._vectors_from_model_to_origin(left_eye_rectangle_model_space,
                                                                                 rotation_vector,
                                                                                 translation_vector,
                                                                                 frame.camera)
            right_eye_rectangle_origin_space = self._vectors_from_model_to_origin(right_eye_rectangle_model_space,
                                                                                  rotation_vector,
                                                                                  translation_vector,
                                                                                  frame.camera)
            left_eye_center_origin_space = self._vectors_from_model_to_origin(left_eye_center_model_space,
                                                                             rotation_vector,
                                                                             translation_vector,
                                                                             frame.camera)
            right_eye_center_origin_space = self._vectors_from_model_to_origin(right_eye_center_model_space,
                                                                             rotation_vector,
                                                                             translation_vector,
                                                                             frame.camera)
            nose_origin_space = self._vectors_from_model_to_origin(nose_model_space,
                                                                   rotation_vector,
                                                                   translation_vector,
                                                                   frame.camera)
            chin_origin_space = self._vectors_from_model_to_origin(chin_model_space,
                                                                   rotation_vector,
                                                                   translation_vector,
                                                                   frame.camera)

            actor = Actor(name=f'Actor{i}', origin=origin)
            actor.set_landmarks3d_eye_rectangles(left_eye_rectangle_origin_space.T, right_eye_rectangle_origin_space.T)
            actor.set_landmarks3d_nose_chin(nose_origin_space.reshape(3), chin_origin_space.reshape(3))
            actor.set_landmarks3d_eye_centers(left_eye_center_origin_space.reshape(3),
                                              right_eye_center_origin_space.reshape(3))
            actor.rotation = - frame.camera.rotation + rotation_vector
            actors.append(actor)

        return actors
