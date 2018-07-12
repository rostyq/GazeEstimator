from config import *
from app import Scene
from app.utils import visualize_predict
from app.estimation import PersonDetector
import json
from cv2 import imread

if __name__ == "__main__":

    back = imread('../screen1.png')

    face_detector = PersonDetector(path_to_face_model=PATH_TO_FACE_MODEL,
                                   path_to_face_points=PATH_TO_FACE_POINTS,
                                   path_to_hc_model=PATH_TO_HAARCASCADE_MODEL,
                                   factor=1,
                                   chin_nose_distance=0.065)

    with open('./extrinsic_params.json', 'r') as f:
        extrinsic_params = json.load(f)

    scene = Scene(origin_name=ORIGIN_CAM, intrinsic_params=INTRINSIC_PARAMS, extrinsic_params=extrinsic_params)
    visualize_predict(face_detector, scene, 'checkpoints/LRE_ff_gp+brs_batch_norm/model_1300_0.0021.h5', back=back)
