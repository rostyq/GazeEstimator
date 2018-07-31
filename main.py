import sys

from config import *
from app.estimation import PersonDetector
from app import Scene

from app.meta import meta
from app.gather import gather
from app.traintest import train
from app.traintest import test
from app.postprocess import postprocess
from app.visualize import visualize

face_detector = PersonDetector(**PERSON_DETECTOR)

scene = Scene(origin_name=ORIGIN_CAM, intrinsic_params=INTRINSIC_PARAMS, extrinsic_params=EXTRINSIC_PARAMS)

run_dict = {
    'meta': meta,
    'visualize': visualize,
    'postprocess': postprocess,
    'train': train,
    'test': test,
    'gather': gather
}

params = {
    'meta': {'face_detector': face_detector, 'scene': scene, 'markers_json': MARKERS},
    'visualize': {'face_detector': face_detector, 'scene': scene},
    'postprocess': {},
    'train': {},
    'test': {},
    'gather': {}
}


def main(*args):

    command = args[1]
    print(command)
    print(args)
    kwargs = {arg.split('=')[0]: arg.split('=')[1] for arg in args[2:]}
    if command:
        kwargs.update(params[command])
        run_dict[command](**kwargs)
    else:
        raise Exception


if __name__ == "__main__":
    main(*sys.argv)
