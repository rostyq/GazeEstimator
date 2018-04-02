#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import yaml
from PIL import Image


class IntrCalibration:
    def __init__(self):
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None

    def get_calibration(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(5,8,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        cap = cv2.VideoCapture(0)
        img_counter = 0
        while(img_counter < 10):         
            ret, img = cap.read()
            cv2.putText(img, 'frame counter: ' + str(img_counter), (10, 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1) 
            cv2.imshow("test", img)
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
                if ret == True:
                    img_counter += 1
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    print("{} self.ret, self.mtx, self.dist, self.rvecs, self.tvecswritten!".format(img_name))
                    # Certainly, every loop objp is the same, in 3D.
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    imgpoints.append(corners2)
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (6,9), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(10)
        
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def param_to_yaml(self):
        data = {'camera_matrix' : np.asarray(self.mtx).tolist(), 
                'dist_coeff'    : np.asarray(self.dist).tolist()}
        with open("calibration.yaml", "w") as f:
            yaml.dump(data, f)
    
    def param_from_yaml(self):
        with open('calibration.yaml') as f:
            loadeddict = yaml.load(f)
        self.mtx = loadeddict.get('camera_matrix')
        self.dist = loadeddict.get('dist_coeff')
    
    def print_param(self):
        print('Camera Matrix :\n {}'.format(self.mtx))
        print('Distortion Coefficients  :\n {}'.format(self.dist))
        
if __name__ == '__main__':
    intr_calibr = IntrCalibration()
    intr_calibr.get_calibration()
    intr_calibr.print_param()