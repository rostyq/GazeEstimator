import tkinter as tk

from app.normalisation import DlibImageNormalizer
from app.normalisation.utils import *


# from calibration import Calibration


def run_experiment(model = 'tutorial'):
    #camera = Calibration(board_shape=(6, 4), path_to_dataset=r'calibration\stand_dataset\kinect')
    #camera.calibrate()

    cap = cv2.VideoCapture(0)
    root = tk.Tk()
    screen = root.winfo_screenheight(), root.winfo_screenwidth()

    # Display the resulting frame
    cv2.namedWindow(__name__, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(__name__, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ret, frame = cap.read()
    normalizer = DlibImageNormalizer(frame.shape)

    i = 0
    while cv2.waitKey(1) != 27:
        ret, frame = cap.read()

        eyes = normalizer.get_normalized_eye_frames(frame)
        draw_eye_centeres(normalizer)
        draw_faces_rectangles(normalizer)
        draw_eye_borders(normalizer)

        cv2.imshow(__name__, normalizer.frame)
        if eyes:
            cv2.imshow(__name__ + '1', eyes[0][1])


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    run_experiment()