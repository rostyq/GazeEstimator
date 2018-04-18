def create_experiment(average_distance, screen_diagonal):

    from .normalisation import FacesRecognition, extract_normalized_eye_pictures
    from .estimator import estimate_gaze
    from .cv2window import ExperimentWindow, create_random_coordinates, calc_pog_coordinates, ispressed
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(0)
    _, frame = capture.read()

    window = ExperimentWindow(__name__, screen_diagonal=screen_diagonal)
    window.open()
    face_recognitor = FacesRecognition(frame.shape)

    test_tick_counter = 0
    test_tick_threshold = 10
    test_coordinates = create_random_coordinates(window.screen_resolution)

    while not ispressed(27):

        _, frame = capture.read()

        try:
            right_eye_img, left_eye_img = extract_normalized_eye_pictures(face_recognitor, frame)[0]
        except TypeError:
            continue

        left_gaze_vector = estimate_gaze(left_eye_img, np.array([0, 0, 0])).reshape((3,))

        window.put_text(f'left gaze: {left_gaze_vector}', (0, 40))

        pog_coordinates = calc_pog_coordinates(
            average_distance,
            left_gaze_vector,
            window.screen_resolution,
            window.screen_inches
            )

        window.put_text(f'coordinates: {pog_coordinates[0]}, {pog_coordinates[1]}', (0, 80))

        window.put_image(
            left_eye_img.reshape((36, 60)),
            (-37, -61)
            )
        if test_tick_counter >= 10:
            test_coordinates = create_random_coordinates(window.screen_resolution)
            test_tick_counter = 0

        test_tick_counter += 1

        window.draw_test_circle(test_coordinates)
        window.draw_pog(pog_coordinates)
        window.show()

    # When everything done, release the capture
    capture.release()
    window.close()
