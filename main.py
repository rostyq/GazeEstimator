from config import *
from app import construct_scene_objects
from app import ExperimentParser
from app import Actor
from app import MyTCPHandler
from app.estimation import ActorDetector
import json
import socketserver


if __name__ == "__main__":

    face_detector = ActorDetector(path_to_face_model=PATH_TO_FACE_MODEL,
                                  path_to_face_points=PATH_TO_FACE_POINTS,
                                  factor=6)

    with open('./extrinsic_params.json', 'r') as f:
        extrinsic_params = json.load(f)

    scene = construct_scene_objects(origin_name=ORIGIN_CAM,
                                    intrinsic_params=INTRINSIC_PARAMS,
                                    extrinsic_params=extrinsic_params)

    origin = scene['origin']
    cams = scene['cameras']
    screens = scene['screens']

    cams_dict = {cams[cam_name]: cam_dir for cam_name, cam_dir in CAM_DIRS.items()}
    parser = ExperimentParser(cams_dict=cams_dict,
                              data_dict=DATA_DIRS)
    parser.fit(DATASET_PATH)
    snapshots = [{'frames': frames, 'data': data} for frames, data in parser.snapshots_iterate(range(0, 50))]

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