from .normalisation import DlibImageNormalizer

from .estimator import estimate_gaze
from .estimator import init_model

from .cv2window import ExperimentWindow
from .cv2window import ispressed

from cv2 import VideoCapture
from numpy import array
from numpy.random import randint


def run_coarse_experiment(average_distance, screen_diagonal, path_to_estimator,
                          capture_target=0, test_ticks_treshold=10, button_code_to_stop=27):
    """
    Run coarse experiment for checking predictions of gaze estimator model.

    Parameters:
    -----------
    average_distance: Average distance from display to user's left eye.
    screen_diagonal: Screen diagonal in inches.
    path_to_estimator: path to gaze estimator model.
    capture_target: Parameters passed to opencv capture function. Default: 0 -- usually webcam.
    test_ticks_treshold: Every n-th tick change test circle. Default 10.
    button_code_to_stop: Code of button on keyboard which will stop the experiment. Default 27 -- ESCAPE.

    Returns:
    --------
    None

    """

    # helping functions
    def create_random_coordinates(screen_resolution):
        return randint(0, screen_resolution[0], size=1), randint(0, screen_resolution[1], size=1)

    def calc_pog_coordinates(distance, gaze_vector, screen_resolution, screen_inches):
        return tuple(map(
            int, [
                ((distance * gaze_vector[0] / gaze_vector[2]) / (0.5 * screen_inches[0]) + 1) * (0.5*screen_resolution[0]),
                (distance * gaze_vector[1] / (-gaze_vector[2])) / screen_inches[1] * screen_resolution[1]
                ]))

    # prepare working objects
    capture = VideoCapture(capture_target)
    window = ExperimentWindow(__name__, screen_diagonal)
    gaze_estimator = init_model(path_to_estimator)
    face_recognitor = DlibImageNormalizer(capture.read()[1].shape)

    # prepare experiment window
    window.open()
    window.draw_test_circle(create_random_coordinates(window.screen_resolution))
    window.set_frame_as_background()

    # dummy head pose
    dummy_head_pose = array([0]*3)

    # run cycles of experiment
    ticks = 0
    while not ispressed(button_code_to_stop):

        if not ticks % test_ticks_treshold:
            window.reset_frame()
            window.draw_test_circle(create_random_coordinates(window.screen_resolution))
            window.set_frame_as_background()
        ticks += 1

        try:
            left_eye_img = face_recognitor.get_normalized_eye_frames(capture.read()[1])[0][1]
            left_gaze_vector = estimate_gaze(left_eye_img,
                                             dummy_head_pose,
                                             gaze_estimator)
            pog_coordinates = calc_pog_coordinates(average_distance,
                                                   left_gaze_vector,
                                                   window.screen_resolution,
                                                   window.screen_inches)
            # checking data on the screen
            window.put_text(f'left gaze: {left_gaze_vector}',
                            (0, 40))
            window.put_text(f'coordinates: {pog_coordinates[0]}, {pog_coordinates[1]}',
                            (0, 80))
            window.put_image(left_eye_img.reshape((36, 60)),
                             (-37, -61))
            window.draw_pog(pog_coordinates)
        except TypeError:
            window.put_text('left gaze: No face - no eye :(',
                            (0, 40))
            continue
        finally:
            window.show()

    capture.release()
    window.close()
