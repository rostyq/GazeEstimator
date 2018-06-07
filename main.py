from config import *
from os import path
from app import Scene
from app import create_learning_dataset
from app.utils import experiment_without_BRS
from app import ExperimentParser
import sys
from app import Actor
from app import MyTCPHandler
from app.estimation import ActorDetector
import json
import socketserver

if __name__ == "__main__":

    face_detector = ActorDetector(path_to_face_model=PATH_TO_FACE_MODEL,
                                  path_to_face_points=PATH_TO_FACE_POINTS,
                                  path_to_hc_model=PATH_TO_HAARCASCADE_MODEL,
                                  factor=6)
    if len(sys.argv) == 2:
        DATASET_PATH = sys.argv[1]

    with open('./extrinsic_params.json', 'r') as f:
        extrinsic_params = json.load(f)

    scene = Scene(origin_name=ORIGIN_CAM, intrinsic_params=INTRINSIC_PARAMS, extrinsic_params=extrinsic_params)
    experiment_without_BRS('../', face_detector, scene, 'zhenya', predict=True)

    # parser = ExperimentParser(session_code=path.split(DATASET_PATH)[-1])
    # parser.fit(DATASET_PATH, scene)

    # create_learning_dataset('../', parser, face_detector, scene, indices=range(len(parser.snapshots)))

    # HOST, PORT = '127.0.0.1', 5055
    #
    # # Create the server, binding to localhost on port 9999
    # server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
    # # server.TCPServer.allow_reuse_address = True
    #
    # # Activate the server; this will keep running until you
    # # interrupt the program with Ctrl-C
    # print("Server started")
    # server.serve_forever()