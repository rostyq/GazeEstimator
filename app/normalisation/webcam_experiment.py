from normalisation import FacesRecognition
from normalisation.utils import *
import cv2
import tkinter as tk

def run_experiment(model = 'tutorial'):
    cap = cv2.VideoCapture(0)
    root = tk.Tk()
    screen = root.winfo_screenheight(), root.winfo_screenwidth()

    # Display the resulting frame
    cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fece_recognitor = FacesRecognition(screen)


    i = 0
    while True:
        ret, frame = cap.read()
        key = cv2.waitKey(1)

        if key == 27:
            break

        extract_normalized_eye_pictures(fece_recognitor, frame)




        cv2.imshow('test', fr.frame)

    # When everything done, release the capture    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    run_experiment()