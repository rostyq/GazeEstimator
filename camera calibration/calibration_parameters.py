#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import yaml
import os


class Calibration:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

    def get_param(self, save_img=False,
                        save_path='', read_img=False, read_path=''):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(5,8,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        # 3d point in real world space.
        obj_points = []
        # 2d points in image plane.
        img_points = []
        img_counter = 0
        if read_img:
            for img_path in os.listdir(read_path):
                if os.path.isfile(os.path.join(read_path,img_path)):
                    img = cv2.imread(os.path.join(read_path, img_path))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Find the chess board corners
                    ret, corners = cv2.findChessboardCorners(gray, (6,9),None)
                    if ret:
                        img_counter += 1
                        # Certainly, every loop objp is the same, in 3D.
                        obj_points.append(objp)
                        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                        img_points.append(corners2)
        else:
            cap = cv2.VideoCapture(0)
            while True:
                ret, img = cap.read()
                text = 'frame counter: {}'.format(img_counter)
                cv2.putText(img, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                cv2.imshow("Camera", img)
                k = cv2.waitKey(1)   
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                elif k % 256 == 32:
                    # SPACE pressed
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Find the chess board corners
                    ret, corners = cv2.findChessboardCorners(gray, (6,9),None)
                    # If found, add object points, image points (after refining them)
                    if ret:
                        img_counter += 1
                        img_name = "frame_{}.png".format(img_counter)
                        print("{} was read!".format(img_name))
                        if save_img:
                            cv2.imwrite(os.path.join(save_path, img_name), img)
                            msg = 'image_{} was saved to {}'.format(
                                    img_counter,img_name)
                            print(msg)
                        # Certainly, every loop objp is the same, in 3D.
                        obj_points.append(objp)
                        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                        img_points.append(corners2)
                        # Draw and display the corners
                        img = cv2.drawChessboardCorners(img, (6,9), corners2, ret)
                        cv2.imshow('img', img)
                        cv2.waitKey(10)
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
        if img_counter:
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        else:
            print('Take more images')

    def param_to_yaml(self, yaml_path="calibration.yaml"):
        data = {'camera_matrix'     : np.asarray(self.mtx).tolist(), 
                'dist_coeff'        : np.asarray(self.dist).tolist(),
                'rotation_vecs'     : np.asarray(self.rvecs).tolist(),
                'translation_vecs'  : np.asarray(self.tvecs).tolist()}
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

    def param_from_yaml(self, yaml_path='calibration.yaml'):
        with open(yaml_path) as f:
            loaded_dict = yaml.load(f)
        self.mtx = np.array(loaded_dict.get('camera_matrix'))
        self.dist = np.array(loaded_dict.get('dist_coeff'))
        self.rvecs = []
        for rvec in loaded_dict.get('rotation_vecs'):
            self.rvecs.append(np.array(rvec))
        self.tvecs = []
        for tvec in loaded_dict.get('translation_vecs'):
            self.tvecs.append(np.array(tvec))

    def print_param(self):
        print('Camera Matrix :\n {}'.format(self.mtx))
        print('Distortion Coefficients :\n {}'.format(self.dist))
        print('Rotation vectors :\n {}'.format(self.rvecs))
        print('Translation vectors :\n {}'.format(self.tvecs))

if __name__ == '__main__':
    pass

