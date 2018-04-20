from os import path, listdir

from app.calibration import Calibration
from app.normalisation import DlibImageNormalizer
from app.normalisation.utils import *


def run_experiment():
    camera = Calibration(board_shape=(6, 4), path_to_dataset=path.join(path.dirname(__file__),
                                                                       r'..\calibration\stand_dataset\kinect'))
    # TODO calibration don't work properly
    camera.calibrate(method='dataset')

    # camera.load_metadata()

    path_to_frames = path.join(path.dirname(__file__), r'..\..\..\11_04_18\1523433382\DataSource\cam_0\ColorFrame')

    # for filename in :
    frame = path.join(path_to_frames, listdir(path_to_frames)[0])
    image = cv2.imread(frame)

    # recognitor = FacesRecognition(image.shape, camera_matrix=camera.matrix, dist_coeffs=camera.distortion)
    normalizer = DlibImageNormalizer(image.shape)
    eyes = normalizer.get_normalized_eye_frames(image)

    draw_eye_borders(normalizer)
    draw_eye_centeres(normalizer)
    draw_faces_rectangles(normalizer)

    # while cv2.waitKey(33) % 256 != 27:
    if eyes:
        cv2.imshow(__name__ + '1', eyes[0][0])
        cv2.imshow(__name__ + '2', eyes[0][1])

    cv2.waitKey(0)


if __name__ == '__main__':
    run_experiment()
