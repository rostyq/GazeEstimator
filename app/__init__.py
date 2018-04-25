from .normalisation import DlibImageNormalizer
from .estimator import estimate_gaze
from .cv2window import ExperimentWindow
from .cv2window import Capture
from .cv2window import Ticker
from .cv2window import create_random_coordinates
from .cv2window import calc_pog_coordinates
from .cv2window import ispressed
from .cv2window import dummy_head_pose

def run_experiment(average_distance, screen_diagonal, test_ticks=10):

    capture = Capture(0)# <=                                                    # why should i be here?
    ticker = Ticker(test_ticks)
    window = ExperimentWindow(__name__, screen_diagonal)
    face_recognitor = DlibImageNormalizer(capture.frame_shape)

    window.open()
    window.draw_test_circle(create_random_coordinates(window.screen_resolution))
    window.set_frame_as_background()

    while not ispressed(ESC):# <=                                                # why should i be here?

        if ticker():
            window.reset_frame()
            window.draw_test_circle(create_random_coordinates(window.screen_resolution))
            window.set_frame_as_background()

        try:
            left_eye_img = face_recognitor.get_normalized_eye_frames(capture.get_frame())
            left_gaze_vector = estimate_gaze(left_eye_img[0][1], dummy_head_pose).reshape((3,))
            pog_coordinates = calc_pog_coordinates(average_distance,
                                                   left_gaze_vector,
                                                   window.screen_resolution,
                                                   window.screen_inches)
            window.put_text(f'left gaze: {left_gaze_vector}',
                            (0, 40))# <=                                        # why should i be here?
            window.put_text(f'coordinates: {pog_coordinates[0]}, {pog_coordinates[1]}',
                            (0, 80))# <=                                        # why should i be here?
            window.put_image(left_eye_img.reshape((36, 60)),#                   # why should i be here?
                             (-37, -61))# <=                                    # why should i be here?
            window.draw_pog(pog_coordinates)
        except TypeError:
            window.put_text('left gaze: No face - no eye :(',
                            (0, 40))# <=                                        # why should i be here?
            continue
        finally:
            window.show()

    capture.release()
    window.close()
