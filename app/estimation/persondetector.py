from app.actor import Person

from dlib import shape_predictor
from dlib import rectangle as DlibRectangle
from dlib import rectangles as DlibRectangles

from scipy.io import loadmat

from cv2 import resize
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from cv2 import solvePnP
from cv2 import SOLVEPNP_ITERATIVE
from cv2 import Rodrigues
from cv2 import CascadeClassifier

from numpy import array
from numpy import tile
from numpy import concatenate


class PersonDetector:

    # 30 -- nose tip
    # 8 -- chin
    # 36 -- right eye outer corner
    # 45 -- left eye outer corner
    # 48 -- right mouth corner
    # 54 -- left mouth corner

    landmarks_to_model = [30, 8, 36, 45, 48, 54]

    def __init__(self, path_to_face_model, path_to_face_points, path_to_hc_model, factor, scale=1.3, minNeighbors=5,
                 chin_nose_distance=0.065):

        # init face detector model
        self.detector = CascadeClassifier(path_to_hc_model).detectMultiScale

        # parameters for face detector model
        self.scale = scale
        self.minNeighbors = minNeighbors
        self.factor = factor

        # init face landmarks detector model
        self.predictor = shape_predictor(path_to_face_points)

        # parameters for face landmarks model
        self.model_points = loadmat(path_to_face_model)['model'] * array([-1, -1, 1])
        face_scale = chin_nose_distance / self.model_points[1, 1]
        self.eye_height = 60 * face_scale
        self.eye_width = 160 * face_scale
        self.model_points = self.model_points * face_scale
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
    def _extract_face_landmarks(cls, raw_dlib_faces):
        """
        Extract 6 face landmarks from self.landmarks which corresponds to generic face model
        :return: None
        """
        if not raw_dlib_faces:
            return []
        else:
            return [face_landmarks_2d[cls.landmarks_to_model] for face_landmarks_2d in raw_dlib_faces]

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

    def extract_faces(self, image):

        # downscale image for faster detection
        image_for_detector = self.downscale(self.to_grayscale(image))

        # detect faces and return cv2-friendly rectangles
        rectangles = self.cvface2dlibrects(self.detector(image_for_detector,
                                                         scaleFactor=self.scale,
                                                         minNeighbors=self.minNeighbors))

        # raw 2d dlib landmarks
        raw_dlib_faces = [self.rescale_coordinates(self.shape_to_np(self.predictor(image_for_detector, rectangle)))
                          for rectangle in rectangles]

        return raw_dlib_faces, self._extract_face_landmarks(raw_dlib_faces=raw_dlib_faces)

    def detect_person(self, name, extracted_face, camera, origin, raw_dlib_face=None):

        person_face_landmarks_2d = array(extracted_face, dtype="double")
        success, rotation_vector, translation_vector = solvePnP(self.model_points,
                                                                person_face_landmarks_2d,
                                                                camera.matrix,
                                                                camera.distortion,
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
                                                                     camera).T

        person = Person(name=name, origin=origin)

        # save raw data from dlib to person object
        person.raw_dlib_landmarks = raw_dlib_face

        person.set_dlib_landmarks3d(face_model_origin_space)
        person.rotation = camera.rotation + rotation_vector
        person.nose_chin_distance = self.nose_chin_distance

        return person

    def detect_persons(self, frame, origin):

        # find faces on image
        raw_dlib_faces, extracted_faces_2d = self.extract_faces(frame.image)

        persons = [self.detect_person(name=f'Person{i}',
                                      extracted_face=face,
                                      camera=frame.camera,
                                      origin=origin,
                                      raw_dlib_face=raw_dlib_faces[i])
                   for i, face in enumerate(extracted_faces_2d)]

        return persons
