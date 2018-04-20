def show_charuco(path_to_image, screen_diagonal, square_length_cm, shift):

    from .cv2window import ExperimentWindow
    from .cv2window import ispressed
    from .cv2window import read_grayscale_image
    from .cv2window import resize_image
    from numpy import full, uint8
    from cv2 import imshow

    charuco_window = ExperimentWindow('CHARUCO', screen_diagonal)
    charuco_image = read_grayscale_image(path_to_image)

    y, x = charuco_image.shape
    square_length = y // 4
    square_length_inch = square_length_cm * 0.3937007874
    dpi = charuco_window.screen_resolution[0] / charuco_window.screen_inches[0]
    ynew = int(y*square_length_inch*dpi/square_length)
    xnew = int(ynew*(x/y))

    charuco_image = resize_image(charuco_image, (xnew, ynew))
    new_charuco = full(charuco_window.screen_resolution[::-1], 255, dtype=uint8)

    new_charuco[shift[0]:charuco_image.shape[0]+shift[0], shift[1]:charuco_image.shape[1]+shift[1]] = charuco_image
    charuco_window.background = new_charuco

    charuco_window.open()
    while not ispressed(27):
        charuco_window.show()
    charuco_window.close()

def run_experiment(average_distance, screen_diagonal, test_ticks=10):

    from .normalisation import FacesRecognition
    from .normalisation import extract_normalized_eye_pictures
    from .estimator import estimate_gaze
    from .cv2window import ExperimentWindow
    from .cv2window import Capture
    from .cv2window import Ticker
    from .cv2window import create_random_coordinates
    from .cv2window import calc_pog_coordinates
    from .cv2window import ispressed
    from .cv2window import dummy_head_pose

    capture = Capture(0)
    ticker = Ticker(test_ticks)
    window = ExperimentWindow(__name__, screen_diagonal)
    face_recognitor = FacesRecognition(capture.frame_shape)

    window.open()
    window.draw_test_circle(create_random_coordinates(window.screen_resolution))
    window.set_frame_as_background()

    while not ispressed(27):

        if ticker():
            window.reset_frame()
            window.draw_test_circle(create_random_coordinates(window.screen_resolution))
            window.set_frame_as_background()

        try:
            left_eye_img = extract_normalized_eye_pictures(
                face_recognitor,
                capture.get_frame()
                )[0][1]
            left_gaze_vector = estimate_gaze(
                left_eye_img,
                dummy_head_pose
                ).reshape((3,))
            pog_coordinates = calc_pog_coordinates(
                average_distance,
                left_gaze_vector,
                window.screen_resolution,
                window.screen_inches
                )
            window.put_text(
                f'left gaze: {left_gaze_vector}',
                (0, 40)
                )
            window.put_text(
                f'coordinates: {pog_coordinates[0]}, {pog_coordinates[1]}',
                (0, 80)
                )
            window.put_image(
                left_eye_img.reshape((36, 60)),
                (-37, -61)
                )
            window.draw_pog(pog_coordinates)
        except TypeError:
            window.put_text('left gaze: No face - no eye :(', (0, 40))
            continue
        finally:
            window.show()

    capture.release()
    window.close()
