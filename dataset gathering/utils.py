import cv2
import numpy as np
from IPython.display import display, clear_output 
import tkinter as tk
import dlib
import os
from imutils import face_utils
from scipy.io import loadmat, savemat

def get_eye_landmarks(rects, gray, frame, predictor, draw = False):
    landmarks = []
    for (k, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        if draw:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(frame, "Face #{}".format(k + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw eyes on the image
        for (j, (x, y)) in enumerate(shape):
            if (j+1) in np.arange(37, 49):
                if draw:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                landmarks.append((x, y))
    return landmarks
            
def current_face_model(rects, gray, frame, predictor, model):
    image_points = np.empty((len(rects), 6)).tolist()
    for (k, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        j = 1
        for (x, y) in shape:
            if model == 'dataset':
                if j == 37: # Left eye left corner
                    image_points[k][0] = (x, y)
                elif j == 40: # Left eye right corner
                    image_points[k][1] = (x, y)
                elif j == 43: # Right eye left corner
                    image_points[k][2] = (x, y)
                elif j == 46: # Right eye right corner
                    image_points[k][3] = (x, y)
                elif j == 49: # Left Mouth corner
                    image_points[k][4] = (x, y)
                elif j == 55: # Right mouth corner
                    image_points[k][5] = (x, y)
            else:
                if j == 31: # Nose tip
                    image_points[k][0] = (x, y)
                elif j == 9: # Chin
                    image_points[k][1] = (x, y)
                elif j == 37: # Left eye left corner 
                    image_points[k][2] = (x, y)
                elif j == 46: # Right eye right corner
                    image_points[k][3] = (x, y)
                elif j == 49: # Left Mouth corner
                    image_points[k][4] = (x, y)
                elif j == 55: # Right mouth corner
                    image_points[k][5] = (x, y)
            j += 1
    return image_points
def normalized_eye_frames(frame, rotation_vector, translation_vector, 
                       camera_matrix, dist_coeffs, four_points_plane, center, translation_to_eyes):
    
    #Drawing plane in front of face
    four_points_plane_proj,_ = cv2.projectPoints(four_points_plane, rotation_vector, 
                                                 translation_vector, camera_matrix, dist_coeffs)
    
    # Calculate Homography
    h, status = cv2.findHomography(four_points_plane_proj[:,0,0:2], four_points_plane[:,0:2] + translation_to_eyes)
    
    eyes = cv2.warpPerspective(frame, h, (225 * 2, 70 * 2))
    return eyes[25:-25,:150], eyes[25:-25,-150:]
def get_face_pose(rects, gray, frame, predictor, model):

    #face models
    image_points = current_face_model(rects, gray, frame, predictor, model)
    
    if model == 'dataset':
        matfile = loadmat('../../MPIIGaze/Data/6 points-based face model.mat')
        model_points = matfile['model'].T
        model_points = model_points * np.array([1, 1, -1])
    else:
        matfile = loadmat('6 points-based face model tutorial.mat')
        model_points = matfile['model']
        model_points = model_points * np.array([1, -1, 1])
    
    # Camera internals
    size = frame.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double")
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    
    # labels on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for person_face in image_points:

        person_points = np.array(person_face, dtype="double")
        
        #drawing eyes centers
        eyes = np.array(get_eye_landmarks(rects, gray, frame, predictor))
        l_eye, r_eye = eyes[:6], eyes[6:] 
        l_eye_c,r_eye_c = (l_eye.sum(axis = 0)/l_eye.shape[0]), (r_eye.sum(axis = 0)/r_eye.shape[0])
        
#         cv2.circle(frame, tuple(l_eye_c.astype(int)), 1, (0, 0, 255), -1)
#         cv2.circle(frame, tuple(r_eye_c.astype(int)), 1, (0, 0, 255), -1)
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, 
                                                                      person_points, 
                                                                      camera_matrix, 
                                                                      dist_coeffs, 
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)
        cv2.putText(frame, f'Translation vector=[{translation_vector[0][0]:.2f},{translation_vector[1][0]:.2f}, {(translation_vector[2][0]):.2f}]', 
                   (0,50), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
        
        #extacting normalized eye images
        l_eye, r_eye = model_points[2], model_points[3] 
        eye_height = 70
        four_points_plane = np.array([(l_eye[0], l_eye[1] - eye_height, l_eye[2]), 
                                      (r_eye[0], l_eye[1] - eye_height, l_eye[2]), 
                                      (r_eye[0], l_eye[1] + eye_height, l_eye[2]), 
                                      (l_eye[0], l_eye[1] + eye_height, l_eye[2])], dtype = "double")

        
        #drawing plane of face
        #four_points_plane = np.array([(-225, -240, -135), (225, -240, -135), (225, -100, -135), (-225, -100, -135)], dtype = "double")
        (end_points2D, jacobian) = cv2.projectPoints(four_points_plane, 
                                                         rotation_vector, translation_vector, camera_matrix, dist_coeffs)

#         for p in person_points:
#             cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

#         for i, p in enumerate(end_points2D):
#             cv2.circle(frame, (int(p[0][0]), int(p[0][1])), 3, (255,0,0), -1)
        
#         cv2.line(frame, (int(end_points2D[0][0][0]), int(end_points2D[0][0][1])),
#                  (int(end_points2D[1][0][0]), int(end_points2D[1][0][1])), (255,0,0), 2)
#         cv2.line(frame, (int(end_points2D[1][0][0]), int(end_points2D[1][0][1])),
#                  (int(end_points2D[2][0][0]), int(end_points2D[2][0][1])), (255,0,0), 2)
#         cv2.line(frame, (int(end_points2D[2][0][0]), int(end_points2D[2][0][1])),
#                  (int(end_points2D[3][0][0]), int(end_points2D[3][0][1])), (255,0,0), 2)
#         cv2.line(frame, (int(end_points2D[3][0][0]), int(end_points2D[3][0][1])),
#                  (int(end_points2D[0][0][0]), int(end_points2D[0][0][1])), (255,0,0), 2)
         
    translation_to_eyes = - np.array([l_eye[0], l_eye[1] - eye_height])
    left_eye_frame, right_eye_frame = normalized_eye_frames(frame, rotation_vector, translation_vector, camera_matrix, 
                                  dist_coeffs, four_points_plane, center, translation_to_eyes)
    #left_eye_frame, right_eye_frame = cv2.resize(left_eye_frame, (60, 36)), cv2.resize(right_eye_frame, (60, 36))     
    left_eye_frame, right_eye_frame = cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_eye_frame, cv2.COLOR_BGR2GRAY)
    left_eye_frame, right_eye_frame = cv2.equalizeHist(left_eye_frame), cv2.equalizeHist(right_eye_frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame[:right_eye_frame.shape[0],-right_eye_frame.shape[1]:] = right_eye_frame
    frame[:left_eye_frame.shape[0],-right_eye_frame.shape[1]-left_eye_frame.shape[1]:-right_eye_frame.shape[1]] = left_eye_frame
    return rotation_vector, frame

def draw_random_point(coord, screen):
    white = np.zeros(screen, dtype=np.uint8) 
    cv2.circle(white, coord, 3, (200, 200), 5)
    return white

