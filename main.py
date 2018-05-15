from config import *
from app import train_gaze_net
# from app import run_coarse_experiment
# from app import show_charuco
import numpy as np
from app.estimator import GazeNet
from app.estimator import gaze2Dto3D, gaze3Dto2D
from app.estimator import prepare
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

    # TODO net training example
    # gaze_estimator = GazeNet().init(PATH_TO_ESTIMATOR)
    #
    # images = np.zeros((1000, 36, 60), dtype=np.uint8)
    # poses = np.zeros((1000, 3), dtype=np.float32)
    # gazes = np.zeros((1000, 3), dtype=np.float32)
    #
    # gaze_estimator.train(prepare(images[:750, :], poses[:750, :]),
    #                      gaze3Dto2D(gazes[:750,:]),
    #                      validation_data=(prepare(images[750:, :], poses[750:, :]), gaze3Dto2D(gazes[750:,:])),
    #                      epochs=1,
    #                      batch_size=64,
    #                      create_new=False,
    #                      path_to_save='./checkpoints')
