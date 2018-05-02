from app import run_coarse_experiment
from config import *

if __name__ == '__main__':
    run_coarse_experiment(average_distance=DEFAULT_AVERAGE_DISTANCE,
                          screen_diagonal=screen_diagonal,
                          path_to_estimator=PATH_TO_ESTIMATOR,
                          capture_target=DEFAULT_WEBCAM_CAPTURE_TARGET,
                          test_ticks_treshold=TEST_TICKS_TRESHOLD,
                          button_code_to_stop=ESCAPE)
