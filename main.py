from calibration import Calibration
from os import path, listdir
from normalisation import FacesRecognition
import cv2

if __name__ == '__main__':
    camera = Calibration(board_shape=(6, 4), path_to_dataset=r'calibration\stand_dataset\kinect')
    #TODO calibration don't work properly
    camera.calibrate()

    #camera.load_metadata()

    #cap = cv2.VideoCapture(0)

    path_to_frames = r'..\11_04_18\1523433382\DataSource\cam_0\ColorFrame'

    # for filename in :
    frame = path.join(path_to_frames, listdir(path_to_frames)[0])
    image = cv2.imread(frame)

    # recognitor = FacesRecognition(image.shape, camera_matrix=camera.matrix, dist_coeffs=camera.distortion)
    recognitor = FacesRecognition(image.shape)
    recognitor.set_image(image)
    recognitor.decect_faces()
    if len(recognitor.rects) > 0:
        recognitor.detect_landmarks()
        recognitor.detect_faces_poses()
        recognitor.produce_normalized_eye_frames()
        recognitor.draw_eye_borders()
        recognitor.draw_eye_centeres()
        recognitor.draw_faces_rectangles()


    # while cv2.waitKey(33) % 256 != 27:
    cv2.imshow(__name__ + '1', recognitor.norm_eye_frames[0][0])
    cv2.imshow(__name__ + '2', recognitor.norm_eye_frames[0][1])

    cv2.waitKey(0)




