from app.actor import Actor
from dlib import get_frontal_face_detector
from dlib import shape_predictor
from cv2 import resize
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from numpy import array


class ActorDetector:

    def __init__(self, path_to_face_model, factor):
        self.detector = get_frontal_face_detector()
        self.predictor = shape_predictor(path_to_face_model)
        self.factor = factor

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

    def detect_actors(self, frame, origin):
        image_for_detector = self.downscale(self.to_grayscale(frame.image))
        rectangles = self.detector(image_for_detector)
        return [Actor(name=f'Actor{i}',
                      frame=frame,
                      origin=origin,
                      rectangle=rectangle,
                      landmarks2D=self.rescale_coordinates(self.predictor(image_for_detector, rectangle)))
                for i, rectangle in enumerate(rectangles)]
