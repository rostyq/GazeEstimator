from numpy import zeros, mgrid, float32
from cv2 import *
import yaml
import os
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Calibration:

    def __init__(self):
        self.points = self.points()
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

    def term_criteria(self):
        '''
            termination criteria
        '''
        return (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def points(self):
        '''
        prepare object points
        example: (0,0,0), (1,0,0), (2,0,0) ....,(5,8,0)
        '''
        points = zeros((6*9,3), float32)
        points[:,:2] = mgrid[0:6,0:9].T.reshape(-1,2)
        return points

    def get_param(self, save_path='', dataset_path=''):
        # Arrays to store object points and image points from all the images.
        # 3d point in real world space.
        object_points = []
        # 2d points in image plane.
        img_points = []
        frame_counter = 0
        if dataset_path:
            for filename in os.listdir(dataset_path):
                if os.path.isfile(os.path.join(dataset_path, filename)):
                    img = imread(os.path.join(dataset_path, filename))
                    gray = cvtColor(img, COLOR_BGR2GRAY)
                    # Find the chess board corners
                    corners = findChessboardCorners(gray, (8), self.retrieval_stage)
                    if self.retrieval_stage:
                        frame_counter += 1
                        # Certainly, every loop points is the same, in 3D.
                        object_points.append(self.points)
                        corners2 = cornerSubPix(gray, corners, (11,11), (-1,-1),
                                                self.term_criteria)
                        img_points.append(corners2)
        else:
            cap = VideoCapture(CAP_ANY)
            print(Mat)
            while True:
                self.retrieval_stage, image = cap.read()
                text = 'frame counter: {}'.format(frame_counter)
                putText(image, text, (10, 10), FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                imshow('capture', image)
                k = waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    logger.warning('Escape hit, closing...')
                    break
                elif k % 256 == 32:
                    # SPACE pressed
                    gray = cvtColor(image, COLOR_BGR2GRAY)
                    # Find the chess board corners
                    self.retrieval_stage, corners = findChessboardCorners(gray, (8), self.retrieval_stage)
                    # If found, add object points, image points (after refining them)
                    if self.retrieval_stage:
                        frame_counter += 1
                        img_name = 'frame_{}.png'.format(frame_counter)
                        logger.info('{} was read!'.format(img_name))
                        if save_img:
                            imwrite(os.path.join(save_path, img_name), image)
                            logger.info('image_{} was saved to {}'.format(frame_counter, img_name))
                        # Certainly, every loop points is the same, in 3D.
                        object_points.append(self.points)
                        corners2 = cornerSubPix(gray, corners, (11,11), (-1,-1), self.term_criteria())
                        img_points.append(corners2)
                        # Draw and display the corners
                        image = drawChessboardCorners(image, (6,9), corners2, self.retrieval_stage)
                        imshow('image', image)
                        waitKey(10)
            # When everything done, release the capture
            cap.release()
            destroyAllWindows()
        if frame_counter:
            self.retrieval_stage, self.mtx, self.dist, self.rvecs, self.tvecs = \
            calibrateCamera(object_points, img_points, gray.shape[::-1], None, None)
        else:
            logger.warning('Take more images')

    def param_to_yaml(self, yaml_path='calibration.yaml'):
        data = {'camera_matrix'    : asarray(self.mtx).tolist(),
                'dist_coeff'       : asarray(self.dist).tolist(),
                'rotation_vecs'    : asarray(self.rvecs).tolist(),
                'translation_vecs' : asarray(self.tvecs).tolist()}
        with open(yaml_path, 'w') as _file:
            yaml.dump(data, _file)

    def param_from_yaml(self, yaml_path='calibration.yaml'):
        with open(yaml_path) as _file:
            params = yaml.load(_file)
        self.mtx = array(params.get('camera_matrix'))
        self.dist = array(params.get('dist_coeff'))
        self.rvecs = [array(rvec) for rvec in params.get('rotation_vecs')]
        self.tvecs = [array(tvec) for tvec in params.get('translation_vecs')]


    def param_log(self):
        logger.info('Camera Matrix :\n {}'.format(self.mtx))
        logger.info('Distortion Coefficients :\n {}'.format(self.dist))
        logger.info('Rotation vectors :\n {}'.format(self.rvecs))
        logger.info('Translation vectors :\n {}'.format(self.tvecs))
