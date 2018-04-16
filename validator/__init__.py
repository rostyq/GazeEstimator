import cv2
import tkinter as tk
import numpy as np

root = tk.Tk()
screen_resolution = root.winfo_screenheight(), \
                    root.winfo_screenwidth()

# Display the resulting frame
cv2.namedWindow(
    __name__,
    cv2.WND_PROP_FULLSCREEN
    )
cv2.setWindowProperty(
    __name__,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
    )

while not cv2.waitKey(1) == 27:
    # eye_detector = Kek()
    # left, right = eye_detector.get_eyes()
    background = np.zeros(screen_resolution)
    cv2.imshow(__name__, background)
