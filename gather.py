from config import *
from app import Scene
from app.utils import experiment_without_BRS
import sys
from app.estimation import ActorDetector
import json

if __name__ == "__main__":

    face_detector = ActorDetector(path_to_face_model=PATH_TO_FACE_MODEL,
                                  path_to_face_points=PATH_TO_FACE_POINTS,
                                  path_to_hc_model=PATH_TO_HAARCASCADE_MODEL,
                                  factor=1,
                                  chin_nose_distance=0.065)
    if len(sys.argv) == 2:
        DATASET_PATH = sys.argv[1]

    with open('./extrinsic_params.json', 'r') as f:
        extrinsic_params = json.load(f)

    scene = Scene(origin_name=ORIGIN_CAM, intrinsic_params=INTRINSIC_PARAMS, extrinsic_params=extrinsic_params)
    experiment_without_BRS('../', face_detector, scene, 'rostyslav_bohomaz_5', size='_72_120', dataset_size=100)
