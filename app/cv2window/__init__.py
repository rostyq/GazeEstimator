import cv2
import numpy as np


def ispressed(button):
    return True if cv2.waitKey(33) == button else False

def create_random_coordinates(screen_resolution):
    return np.random.randint(0, screen_resolution[0], size=1), np.random.randint(0, screen_resolution[1], size=1)

def calc_pog_coordinates(distance, gaze_vector, screen_resolution, screen_inches):
    # x = int((average_distance * left_gaze_vector[0] / (left_gaze_vector[2])) / (inch_width*0.5) * screen_resolution[1]*0.5+screen_resolution[1]*0.5)
    # y = int((average_distance * left_gaze_vector[1] / (-left_gaze_vector[2])) / inch_height * screen_resolution[0])
    return tuple(map(
        int,
        [
            ((distance * gaze_vector[0] / gaze_vector[2]) / (0.5 * screen_inches[0]) + 1) * (0.5*screen_resolution[0]),
            (distance * gaze_vector[1] / (-gaze_vector[2])) / screen_inches[1] * screen_resolution[1]
        ]
                ))

def get_screen_resolution():
    from tkinter import Tk
    root = Tk()
    screen_resolution = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return screen_resolution

def get_screen_inches(screen_resolution, screen_diagonal):
    ratio = np.arctan(screen_resolution[0] / screen_resolution[1])
    return screen_diagonal * np.sin(ratio), screen_diagonal * np.cos(ratio)


class ExperimentWindow:

    font = cv2.FONT_HERSHEY_SIMPLEX
    line = cv2.LINE_AA
    @staticmethod
    def close_all():
        cv2.destroyAllWindows()

    def __init__(self, name, screen_diagonal=None):
        self.name = name
        self.screen_resolution = get_screen_resolution()
        self.screen_inches = get_screen_inches(
            self.screen_resolution,
            screen_diagonal
            ) if (screen_diagonal is not None) else (None, None)
        self._background = np.zeros(self.screen_resolution[::-1], dtype=np.uint8)
        self.frame = self.background

    @property
    def background(self):
        return np.copy(self._background)

    @background.setter
    def set_background(self, image):
        self._background = image

    def open(self):
        cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def close(self):
        cv2.destroyWindow(self.name)

    def show(self):
        cv2.imshow(self.name, self.frame)
        self.frame = self.background

    def draw_pog(self, coordinates, radius=30, color=(255, 0, 0)):
        """Draw Point of Gaze"""
        cv2.circle(self.frame, coordinates, radius, color, -1)
        cv2.circle(self.frame, coordinates, int(radius*0.33), (0, 0, 0), -1)

    def draw_test_circle(self, coordinates, radius=50, color=(255, 0, 0)):
        cv2.circle(self.frame, coordinates, radius, color, -1)
        cv2.circle(self.frame, coordinates, int(radius*0.8), (0, 0, 0), -1)
        cv2.circle(self.frame, coordinates, int(radius*0.6), color, -1)

    def put_text(self, text, coordinates, color=(255, 255, 255)):
        cv2.putText(
            self.frame,
            text,
            coordinates,
            ExperimentWindow.font, 1,
            color, 2,
            ExperimentWindow.line
            )

    def put_image(self, image, coordinates):
        self.frame[
            coordinates[0]:coordinates[0]+image.shape[0],
            coordinates[1]:coordinates[1]+image.shape[1]
            ] = image
