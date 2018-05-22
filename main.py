from config import *
from app import construct_scene_objects
import json
from app.parser import ExperimentParser
from app.actor import Actor
from app import ispressed
import cv2
import numpy as np
from app.estimation import ActorDetector


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

    # snapshots = [{'frames': frames, 'data': data} for frames, data in parser.snapshots_iterate(range(0, 50))]
    # frame = snapshots[0]['frames']['basler']
    # actors = face_detector.detect_actors(frame, origin)
    # print(actors[0].landmarks3D)
    # cv2.imshow('kek1', cv2.resize(frame.project_vectors(np.array(left)).image, (300, 300)))
    # cv2.imshow('kek2', cv2.resize(frame.project_vectors(np.array(right)).image, (300, 300)))
    # cv2.waitKey()