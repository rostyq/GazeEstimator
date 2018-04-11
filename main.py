#Hello
from calibration import Calibration

if __name__ == '__main__':
    camera = Calibration(path_to_dataset=r'C:\Users\Beehiveor\Documents\BAS\GazeEstimator\calibration\stand_dataset')
    camera.calibrate(method='dataset')
