from ipywidgets import Button, HBox, Text, IntSlider, interact
from numpy import arange, empty, array, zeros, uint8, random, arange
from IPython.display import display, clear_output
from skimage.draw import polygon_perimeter
from os import path, makedirs
from time import sleep, time
from imutils import face_utils
from scipy.io import loadmat
from skimage import io
from PIL import Image
from glob import glob
import tkinter as tk
import pandas as pd
from cv2 import *
import dlib

# ## Calibration
# ## Dataset creation

class Recognition(object):
    """docstring for Face Recognition."""
    def __init__(self, arg):
        super(Recognition, self).__init__()
        self.arg = arg


def get_eye_landmarks(rects, gray, frame, predictor):
    landmarks = []
    for (k, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        putText(frame, "Face #{}".format(k + 1), (x - 10, y - 10), FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        j = 1
        for (x, y) in shape:
            if j in arange(37, 49):
                circle(frame, (x, y), 1, (0, 0, 255), -1)
                landmarks.append(x)
                landmarks.append(y)
            j += 1
    return tuple(landmarks)

def get_face_pose(rects, gray, frame, predictor):
    image_points = empty((len(rects), 6)).tolist()
    k = -1
    for rect in rects:
        k += 1
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        j = 1
        for (x, y) in shape:
            if j == 31: # Left eye left corner
                image_points[k][0] = (x, y)
            elif j == 9: # Left eye right corner
                image_points[k][1] = (x, y)
            elif j == 37: # Right eye left corner
                image_points[k][2] = (x, y)
            elif j == 46: # Right eye right corner
                image_points[k][3] = (x, y)
            elif j == 49: # Left Mouth corner
                image_points[k][4] = (x, y)
            elif j == 55: # Right mouth corner
                image_points[k][5] = (x, y)

            j += 1

    size = frame.shape
    font = FONT_HERSHEY_SIMPLEX
    file = '../../MPIIGaze/Data/6 points-based face model.mat'
    matfile = loadmat(file)

    for person in image_points:
        person_points = array(person, dtype="double")
        # 3D model points
        #model_points = matfile['model'].T
        model_points =  array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                              ])

        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = array([[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double")
        #print ("Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = solvePnP(model_points, person_points, camera_matrix, dist_coeffs, flags=SOLVEPNP_ITERATIVE)

        putText(frame, f'Rotation vector=[{rotation_vector[0][0]:.1f}, {rotation_vector[1][0]:.1f}, {rotation_vector[2][0]:.1f}, {success}]',
                   (0,50), font, 0.7, (255,255,255), 2, LINE_AA)
        #print ("Rotation Vector:\n {0}".format(rotation_vector))
        #print ("Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose

        (nose_end_point2D, jacobian) = projectPoints(array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in person_points:
            circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        p1 = ( int(person_points[0][0]), int(person_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # delta_x = - int(person_points[0][0]) + int(person_points[3][0])
        # delta_y = - int(person_points[0][1]) + int(person_points[3][1])
        # p13 = ( int(person_points[3][0]), int(person_points[3][1]))
        # p23 = ( int(nose_end_point2D[0][0][0]) + delta_x, int(nose_end_point2D[0][0][1]) + delta_y)

        # delta_x = - int(person_points[0][0]) + int(person_points[2][0])
        # delta_y = - int(person_points[0][1]) + int(person_points[2][1])
        # p14 = ( int(person_points[2][0]), int(person_points[2][1]))
        # p24 = ( int(nose_end_point2D[0][0][0]) + delta_x, int(nose_end_point2D[0][0][1]) + delta_y)
        # print(p1,p2,p13,p23,p14,p24)
          line(frame, p1, p2, (255,0,0), 2)
        # line(frame, p13, p23, (255,0,0), 2)
        # line(frame, p14, p24, (255,0,0), 2)
    return rotation_vector

def draw_random_point(coord, screen):
    white = zeros(screen, dtype=uint8)
    circle(white, coord, 3, (200, 200), 5)
    return white

def create_learning_dataframe(event):
    #try:
    continious_display = True
    cap = VideoCapture(0)
    root = tk.Tk()
    screen = root.winfo_screenheight(), root.winfo_screenwidth()
    coord = random.choice(screen[1]), random.choice(screen[0])
    white = draw_random_point(coord, screen)

    # Display the resulting frame
    namedWindow("test", WND_PROP_FULLSCREEN)
    setWindowProperty("test", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN)

    #init face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    participant = pt_name.value
    path = 'Data/' + participant
    df = pd.DataFrame(columns=['Time', 'X', 'Y', 'Eye landmarks', 'Rvect', 'Image', 'Screen size'])

    if not os.path.exists(path):
        os.makedirs(path)

    i = 0
    while(True):
        if continious_display:
            ret, frame = cap.read()
        key = waitKey(1)

        if key == 27:
            break

        elif key == ord(' ') or continious_display:
            if i%2 == 0 or continious_display:

                #capturing frame
                ret, frame = cap.read()
                ret, frame = cap.read()
                gray = cvtColor(frame, COLOR_BGR2GRAY)

                #faces detection
                rects = detector(gray, 1)

                #sample detection
                sample = {}
                img_name = str(int(i/2)) + '.png'
                sample['Y'], sample['X'] = coord[1]/screen[0], coord[0]/screen[1]
                sample['Time'] = time()
                sample['Screen size'] = screen
                sample['Image'] = img_name
                if len(rects) > 0:
                    sample['Eye landmarks'] = get_eye_landmarks(rects, gray, frame, predictor)
                    sample['Rvect'] = tuple(get_face_pose(rects, gray, frame, predictor).ravel())

                    if len(sample['Eye landmarks']) > 0:
                        imwrite(path + '/' + img_name, frame)
                        sample_df = pd.DataFrame.from_dict(sample, orient='index')
                    if not continious_display:
                        df = df.append(sample, ignore_index=True)

                display(df)
                clear_output(wait=True)

                #drawing random circle
                coord = random.choice(screen[1]), random.choice(screen[0])
                white = draw_random_point(coord, screen)

            i += 1

        if i%2 == 0 and not continious_display:
            imshow('test', white)
        else:
            imshow('test', frame)

    # When everything done, release the capture

    cap.release()
    df.to_csv(path + '/annotations.txt')
    destroyAllWindows()

pt_name = Text(description='Input name:', value = 'valik')
start = Button(description='Start', button_style='success')
start.on_click(create_learning_dataframe)
box = HBox([pt_name, start])

box

df = pd.read_csv('Data/valik/annotations.txt')

destroyAllWindows()

# ## Anotation creation

from scipy.io import loadmat
import numpy as np
from PIL import Image
import glob

files = 'MPIIGaze/Data/Original/p01/day01'
images = glob(files + '/*.jpg', recursive=True)
slider = IntSlider(min = 0, max = len(images), layout=Layout(width='700px'))

def draw_sample(x):
    sample_n = x
    image = imread(images[sample_n])
    annot = pd.read_csv(files + '/annotation.txt', delimiter=' ', names= arange(41))
    im_landmarks = annot.loc[:,:23].iloc[sample_n].values
    im_landmarks = im_landmarks.reshape((12, 2))
    for mark in im_landmarks:
        circle(image, tuple(mark), 1, (200, 0, 0), 2)
    display(Image.fromarray(image))
    clear_output(wait=True)

interact(draw_sample, x=slider)

image = imread(images[sample_n])
Image.fromarray(image)

cap = VideoCapture(0)
ret, frame = cap.read()
cap.release()
imwrite('example.png', frame)
Image.fromarray(frame)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

gray = cvtColor(frame, COLOR_BGR2GRAY)
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
        FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    i = 1
    for (x, y) in shape:
        if i in arange(37, 49):
            circle(frame, (x, y), 1, (0, 0, 255), -1)
        i += 1

Image.fromarray(frame)
destroyAllWindows()

def landmarks_for_pose_detection(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Read Image
im = imread("headPose.jpg");
size = im.shape

#2D image points. If you change the image, you need to change vector
image_points = array([
                       (359, 391),     # Nose tip
                       (399, 561),     # Chin
                       (337, 297),     # Left eye left corner
                       (513, 301),     # Right eye right corne
                       (345, 465),     # Left Mouth corner
                       (453, 469)      # Right mouth corner
                     ], dtype="double")

# 3D model points.
model_points = array([
                       (0.0, 0.0, 0.0),             # Nose tip
                       (0.0, -330.0, -65.0),        # Chin
                       (-225.0, 170.0, -135.0),     # Left eye left corner
                       (225.0, 170.0, -135.0),      # Right eye right corne
                       (-150.0, -150.0, -125.0),    # Left Mouth corner
                       (150.0, -150.0, -125.0)      # Right mouth corner
                     ])

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = array([[focal_length, 0, center[0]],
                       [0, focal_length, center[1]],
                       [0, 0, 1]],
                      dtype = "double")

print ("Camera Matrix :\n {0}".format(camera_matrix))

dist_coeffs = zeros((4,1)) # Assuming no lens distortion
(success, rotation_vector, translation_vector) = solvePnP(model_points,
                                                          image_points,
                                                          camera_matrix,
                                                          dist_coeffs,
                                                          flags=SOLVEPNP_ITERATIVE)

print ("Rotation Vector:\n {0}".format(rotation_vector))
print ("Translation Vector:\n {0}".format(translation_vector))

# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose

(nose_end_point2D, jacobian) = projectPoints(array([(0.0, 0.0, 1000.0)]),
                                             rotation_vector,
                                             translation_vector,
                                             camera_matrix,
                                             dist_coeffs)

for p in image_points:
   circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

p1 = ( int(image_points[0][0]), int(image_points[0][1]))
p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

line(im, p1, p2, (255,0,0), 2)

# Display image
imshow("Output", im)
waitKey(0)
