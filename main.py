from calibration import Calibration
from os import path, listdir
import tkinter as tk
from normalisation import FacesRecognition, extract_normalized_eye_pictures
from estimator import *
import cv2
import numpy as np

def valik():
    # TODO what is this????
    camera = Calibration(board_shape=(6, 4), path_to_dataset=r'calibration\stand_dataset\kinect')
    #TODO calibration don't work properly
    camera.calibrate()

    #camera.load_metadata()

    #cap = cv2.VideoCapture(0)

    path_to_frames = r'..\11_04_18\1523433382\DataSource\cam_0\ColorFrame'

    # for filename in :
    frame = path.join(path_to_frames, listdir(path_to_frames)[0])
    image = cv2.imread(frame)

    # recognitor = FacesRecognition(image.shape, camera_matrix=camera.matrix, dist_coeffs=camera.distortion)
    recognitor = FacesRecognition(image.shape)
    recognitor.set_image(image)
    recognitor.decect_faces()
    if len(recognitor.rects) > 0:
        recognitor.detect_landmarks()
        recognitor.detect_faces_poses()
        recognitor.produce_normalized_eye_frames()
        recognitor.draw_eye_borders()
        recognitor.draw_eye_centeres()
        recognitor.draw_faces_rectangles()


    # while cv2.waitKey(33) % 256 != 27:
    cv2.imshow(__name__ + '1', recognitor.norm_eye_frames[0][0])
    cv2.imshow(__name__ + '2', recognitor.norm_eye_frames[0][1])

    cv2.waitKey(0)

def validator():

    # TODO class for cv2 window??
    # prepare window
    capture = cv2.VideoCapture(0)
    # TODO maybe without tkinter somehow?
    root = tk.Tk()
    screen_resolution = root.winfo_screenheight(), root.winfo_screenwidth()
    root.destroy()
    distance = 20
    screen_diagonal = 13.3
    ratio = screen_resolution[1] / screen_resolution[0]
    angle = np.arctan(ratio)
    inch_width = screen_diagonal * np.sin(angle)
    inch_height = screen_diagonal * np.cos(angle)

    def create_background():
        return np.zeros(screen_resolution, dtype=np.uint8)

    def draw_pog(img, x, y):
        cv2.circle(background, (x, y), 30, (255, 0, 0), -1)
        cv2.circle(background, (x, y), 10, (0, 0, 0), -1)

    def draw_test_circle(img, x, y):
        cv2.circle(background, (x, y), 50, (255, 0, 0), -1)
        cv2.circle(background, (x, y), 40, (0, 0, 0), -1)
        cv2.circle(background, (x, y), 30, (255, 0, 0), -1)

    # Display the resulting frame
    cv2.namedWindow(__name__, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(__name__, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    _, frame = capture.read()
    face_recognitor = FacesRecognition(frame.shape)

    test_tick_counter = 0
    test_tick_threshold = 10
    test_x, test_y = np.random.randint(0, screen_resolution[1], size=1), np.random.randint(0, screen_resolution[0], size=1)
    while not cv2.waitKey(1) == 27:

        _, frame = capture.read()
        try:
            right_eye_img, left_eye_img = extract_normalized_eye_pictures(face_recognitor, frame)[0]
        except TypeError:
            continue
        left_gaze_vector = estimate_gaze(left_eye_img, np.array([0, 0, 0])).reshape((3,))

        font = cv2.FONT_HERSHEY_SIMPLEX
        background = create_background()
        cv2.putText(
            background,
            f'left gaze: {left_gaze_vector}',
            (0, 40),
            font, 1,
            (255,255,255), 2,
            cv2.LINE_AA
            )
        # TODO write function for geometry
        # TODO enhance geometry (calibration, head rotation, both eyes, display and camera calibration)
        x = int((distance * left_gaze_vector[0] / (left_gaze_vector[2])) / (inch_width*0.5) * screen_resolution[1]*0.5+screen_resolution[1]*0.5)
        y = int((distance * left_gaze_vector[1] / (-left_gaze_vector[2])) / inch_height * screen_resolution[0])

        cv2.putText(
            background,
            f'coordinates: {x}, {y}',
            (0, 80),
            font, 1,
            (255,255,255), 2,
            cv2.LINE_AA
            )

        background[-36:, -60:] = left_eye_img.reshape((36, 60))
        if test_tick_counter >= 10:
            test_x, test_y = np.random.randint(0, screen_resolution[1], size=1), np.random.randint(0, screen_resolution[0], size=1)
            test_tick_counter = 0
        test_tick_counter += 1
        draw_test_circle(background, test_x, test_y)
        draw_pog(background, x, y)
        cv2.imshow(__name__, background)

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    validator()
