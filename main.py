from config import *
from app import train_gaze_net
# from app import run_coarse_experiment
# from app import show_charuco

if __name__ == '__main__':
    # TODO what should I do here?
    pass

    # TODO example 1
    # run_coarse_experiment(average_distance=DEFAULT_AVERAGE_DISTANCE,
    #                       screen_diagonal=screen_diagonal,
    #                       path_to_estimator=PATH_TO_ESTIMATOR,
    #                       capture_target=DEFAULT_WEBCAM_CAPTURE_TARGET,
    #                       test_ticks_treshold=TEST_TICKS_TRESHOLD,
    #                       button_code_to_stop=ESCAPE)

    # TODO example 2
    # show_charuco('./charuco_board.png', 13.3, 4.0, (50, 100))
    faces_path = 'dataset/{index}/0/'
    eyes_path = faces_path+'eyes/{eye}/'
    train_gaze_net(path_to_dataset=path_to_dataset,
                   faces_path=faces_path,
                   eyes_path=eyes_path,
                   json_name=json_name,
                   path_to_save=path_to_save,
                   create_new=True,
                   eye='left',
                   epochs=10,
                   batch_size=64)
