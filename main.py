from config import *
from os import path
import os
from app import Scene
from app import create_learning_dataset
from app.utils import experiment_without_BRS, visualize_predict
from app import ExperimentParser
import sys
from app.estimation import ActorDetector
import json

if __name__ == "__main__":

    face_detector = ActorDetector(**ACTOR_DETECTOR)
    if len(sys.argv) == 2:
        DATASET_PATH = sys.argv[1]

    scene = Scene(origin_name=ORIGIN_CAM, intrinsic_params=INTRINSIC_PARAMS, extrinsic_params=EXTRINSIC_PARAMS)

    gazes = [
        (192/1920, 108/1080),
        (960/1920, 108/1080),
        (1728/1920, 108/1080),
        (1729/1920, 540/1080),
        (960/1920, 540/1080),
        (192/1920, 540/1080),
        (192/1920, 971/1080),
        (960/1920, 971/1080),
        (1729/1920, 971/1080)
        ]

    for session, gaze in zip(sorted(os.listdir(DATASET_PATH)), gazes):
        full_path = path.join(DATASET_PATH, session)
        parser = ExperimentParser(session_code=path.split(full_path)[-1])
        parser.fit(full_path, scene)
        create_learning_dataset('../', parser, face_detector, scene, indices=range(len(parser.snapshots)), gaze=gaze)