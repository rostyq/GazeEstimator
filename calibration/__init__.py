from time import strftime
from os import path, getcwd, listdir
from yaml import dump, load
from logging import basicConfig, getLogger, DEBUG
from numpy import zeros, mgrid, float32, array, asarray
from cv2 import *

basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='log/{}.log'.format(__name__),
            level=DEBUG)
logger = getLogger(__name__)

class Calibration:

    def __init__(self):
        self.path = path.join(getcwd(), 'calibration/dataset')
        self.metadata = path.join(self.path, '{}.yaml'.format(__name__))
        self.object_points = list()
        self.frame_points = list()
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
        capture = VideoCapture(CAP_ANY)
        while capture.isOpened():
            self.retrieval, self.frame = capture.read()
            self.display_message('Saved frames: {}'.format(self.counter))
            imshow(__name__, self.frame)
            if waitKey(33) % 256 == 27:
                logger.warning('ESC pressed, closing...')
                break
            if waitKey(33) % 256 == 32:
                self.retrieval, corners = self.find_corners()
                if self.retrieval:
                    self.counter += 1
                    imwrite(self.frame_path(), self.frame)
                    logger.info('Frame was saved to {}'.format(self.frame_path()))
                    self.object_points.append(self.zero_points())
                    self.draw_corners(corners)
                    imshow(__name__, self.frame)
        capture.release()
        destroyAllWindows()

    def dataset_calibration(self, dataset_path=''):
        dataset_path = self.path if not dataset_path else dataset_path
        for filename in listdir(dataset_path):
            frame = path.join(dataset_path, filename)
            if path.isfile(frame):
                self.frame = imread(frame)
                self.retrieval, corners = self.find_corners()
                if self.retrieval:
                    self.counter += 1
                    self.object_points.append(self.zero_points())
                    self.frame_points.append(self.corners_subpixel(corners))

    def calibrate_camera(self):
        self.retrieval, self.matrix, self.distortion, self.rotation, \
        self.translation = calibrateCamera(self.object_points,
                                           self.frame_points,
                                           self.frame_to_grey().shape[::-1],
                                           None, None)

    def dump_metadata(self):
        metadata = {'camera_matrix':asarray(self.matrix).tolist(),
                    'distortion_coefficient':asarray(self.distortion).tolist(),
                    'rotation_vector':asarray(self.rotation).tolist(),
                    'translation_vector':asarray(self.translation).tolist()}
        with open(self.metadata, 'w') as _file:
            dump(metadata, _file)

    def load_metadata(self):
        with open(self.metadata) as _file:
            metadata = load(_file)
        self.matrix = array(metadata.get('matrix'))
        self.distortion = array(metadata.get('distortion'))
        self.rotation = [array(vector) for vector in metadata.get('rotation')]
        self.translation = [array(vector) for vector in metadata.get('translation')]

    def metadata_logger(self):
        logger.info('Camera Matrix: {}'.format(self.matrix))
        logger.info('Distortion factor: {}'.format(self.distortion))
        logger.info('Rotation vectors: {}'.format(self.rotation))
        logger.info('Translation vectors: {}'.format(self.translation))

    def find_corners(self):
        return findChessboardCorners(self.frame_to_grey(), (6,9))

    def corners_subpixel(self, corners):
        criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
        return cornerSubPix(self.frame_to_grey(), corners, (11,11), (-1,-1), criteria)

    def draw_corners(self, corners):
        corners_subpixel = self.corners_subpixel(corners)
        self.frame_points.append(corners_subpixel)
        drawChessboardCorners(self.frame, (6,9), corners_subpixel, self.retrieval)

    def display_message(self, message):
        putText(self.frame, message, (20, 40), 0, 0.6, (0, 0, 0), 2)

    def zero_points(self):
        points = zeros((6*9,3), float32)
        points[:,:2] = mgrid[0:6,0:9].T.reshape(-1,2)
        return points

    def frame_path(self):
        return path.join(self.path, strftime('%Y%m%d%H%M%S.png'))

    def frame_to_grey(self):
        return cvtColor(self.frame, COLOR_BGR2GRAY)

################################################################################
