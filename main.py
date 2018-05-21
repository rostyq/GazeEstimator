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

    face_detector = ActorDetector(path_to_face_model=PATH_TO_FACE_MODEL, factor=6)

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

    snapshots = [{'frames': frames, 'data': data} for frames, data in parser.snapshots_iterate(range(150, 250))]
    actor = Actor('kek', origin=origin)
    actor.set_landmarks3d(snapshots[0]['data']['face_points'])
    frame = snapshots[0]['frames']['color']

    cv2.imshow('kek', frame.extract_eyes_from_actor(actor)[1])
    print(actor.landmarks3D['eyes']['left']['center'])
    cv2.imshow('kek2', frame.project_vectors(actor.landmarks3D['eyes']['left']['center']).image)
    cv2.waitKey()