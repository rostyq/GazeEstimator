from time import strftime
from os import path, listdir, getcwd
from yaml import dump, load
from logging import basicConfig, getLogger, DEBUG
import numpy as np
import cv2
import json
from PIL import Image

basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='log/{}.log'.format(__name__),
            level=DEBUG)
logger = getLogger(__name__)

class Calibration:

    def __init__(self, board_shape, path_to_dataset = None):
        self.path = path_to_dataset if path is not None else path.join(getcwd(), 'calibration/dataset')
        self.metadata = path.join(self.path, '{}.yaml'.format(__name__))
        self.object_points = list()
        self.frame_points = list()
        self.board_shape = board_shape
        self.counter = 0

    def calibrate(self, method='capture'):
        if method == 'capture':
            self.capture_calibration()
        elif method == 'dataset':
            self.dataset_calibration()
        if self.counter:
            self.calibrate_camera()
            self.dump_metadata()
        else:
            logger.error('Take more images')

    def capture_calibration(self):
        capture = cv2.VideoCapture(cv2.CAP_ANY)
        while capture.isOpened():
            self.retrieval, self.frame = capture.read()
            self.display_message('Saved frames: {}'.format(self.counter))
            cv2.imshow(__name__, self.frame)
            if cv2.waitKey(33) % 256 == 27:
                logger.warning('ESC pressed, closing...')
                break
            if cv2.waitKey(33) % 256 == 32:
                self.retrieval, corners = self.find_corners()
                if self.retrieval:
                    self.counter += 1
                    cv2.imwrite(self.frame_path(), self.frame)
                    logger.info('Frame was saved to {}'.format(self.frame_path()))
                    self.object_points.append(self.zero_points())
                    self.draw_corners(corners)
                    print(1)
                    cv2.imshow(__name__, self.frame)
        capture.release()
        cv2.destroyAllWindows()

    def dataset_calibration(self, dataset_path=''):
        dataset_path = self.path if not dataset_path else dataset_path
        for filename in listdir(dataset_path):
            frame = path.join(dataset_path, filename)
            if path.isfile(frame) and (frame.endswith('.png') or frame.endswith('.jpg')):
                self.frame = cv2.imread(frame)
                self.retrieval, corners = self.find_corners()
                if self.retrieval:
                    self.counter += 1
                    self.object_points.append(self.zero_points())
                    self.frame_points.append(self.corners_subpixel(corners))

    def calibrate_camera(self):
        self.retrieval, self.matrix, self.distortion, self.rotation, \
        self.translation = cv2.calibrateCamera(self.object_points,
                                           self.frame_points,
                                           self.frame_to_grey().shape[::-1],
                                           None, None)

    def dump_metadata(self):
        metadata = {'camera_matrix':np.asarray(self.matrix).tolist(),
                    'distortion_coefficient':np.asarray(self.distortion).tolist(),
                    'rotation_vector':np.asarray(self.rotation).tolist(),
                    'translation_vector':np.asarray(self.translation).tolist()}
        with open(self.metadata, 'w') as _file:
            dump(metadata, _file)

    def load_metadata(self):
        with open(self.metadata) as _file:
            metadata = load(_file)
        self.matrix = np.array(metadata.get('camera_matrix'))
        self.distortion = np.array(metadata.get('distortion_coefficient'))
        self.rotation = [np.array(vector) for vector in metadata.get('rotation_vector')]
        self.translation = [np.array(vector) for vector in metadata.get('translation_vector')]

    def metadata_logger(self):
        logger.info('Camera Matrix: {}'.format(self.matrix))
        logger.info('Distortion factor: {}'.format(self.distortion))
        logger.info('Rotation vectors: {}'.format(self.rotation))
        logger.info('Translation vectors: {}'.format(self.translation))

    def find_corners(self):
        return cv2.findChessboardCorners(self.frame, self.board_shape)

    def corners_subpixel(self, corners):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        return cv2.cornerSubPix(self.frame_to_grey(), corners, (11, 11), (-1, -1), criteria)

    def draw_corners(self, corners):
        corners_subpixel = self.corners_subpixel(corners)
        self.frame_points.append(corners_subpixel)
        cv2.drawChessboardCorners(self.frame, self.board_shape, corners_subpixel, self.retrieval)
        Image.fromarray(self.frame).show()

    def display_message(self, message):
        cv2.putText(self.frame, message, (20, 40), 0, 0.6, (0, 0, 0), 2)

    def zero_points(self):
        points = np.zeros((self.board_shape[0]*self.board_shape[1], 3), np.float32)
        points[:,:2] = np.mgrid[0:self.board_shape[0], 0:self.board_shape[1]].T.reshape(-1, 2)
        return points

    def frame_path(self):
        return path.join(self.path, strftime('%Y%m%d%H%M%S.png'))

    def frame_to_grey(self):
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

################################################################################
