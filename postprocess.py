from config import *
from os import path
from app import Scene
from app import create_learning_dataset
from app import ExperimentParser
import sys
from app.estimation import ActorDetector
import json

DATASET_PATH = r'D:\BRS-datasets\13_06_18\1528910098'

if __name__ == "__main__":

    face_detector = ActorDetector(path_to_face_model=PATH_TO_FACE_MODEL,
                                  path_to_face_points=PATH_TO_FACE_POINTS,
                                  path_to_hc_model=PATH_TO_HAARCASCADE_MODEL,
                                  factor=1,
                                  chin_nose_distance=0.07)


    with open('./extrinsic_params.json', 'r') as f:
        extrinsic_params = json.load(f)

    scene = Scene(origin_name=ORIGIN_CAM, intrinsic_params=INTRINSIC_PARAMS, extrinsic_params=extrinsic_params)

    parser = ExperimentParser(session_code=path.split(DATASET_PATH)[-1])
    parser.fit(DATASET_PATH, scene)

    create_learning_dataset('../brs/', parser, face_detector, scene, indices=range(len(parser.snapshots)))
