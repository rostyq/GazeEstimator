from app import ExperimentParser
from app import Person
import numpy as np
import os
from tqdm import tqdm


def get_data_coord(data, sep=','):
    result = ''
    for axis, value in data.items():
        value = str(int(value)) if not value % 1 else format(value, '.9f')
        result += value + sep
    return result


def get_line(data, title=False, sep=','):
    result = ''
    for key, value in data.items():
        if isinstance(value, dict):
            if title:
                for axis in value.keys():
                    result += key + axis + sep
            else:
                result += get_data_coord(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if title:
                    for axis in item.keys():
                        result += key + str(i) + axis + sep
                else:
                    result += get_data_coord(item, sep=sep)
        else:
            raise Exception
    return result[:-1]


def to_xyz_dict(array):
    return {axis: array for axis, array in zip(['X', 'Y', 'Z'], array)}


def norm(vector):
    return vector / np.linalg.norm(vector)


def form_data(snapshot, face_detector, scene):

    dlib_indices = {
        'eyeInnerCornerRight2d': 39,
        'eyeOuterCornerRight2d': 36,
        'eyeInnerCornerLeft2d': 42,
        'eyeOuterCornerLeft2d': 45,
        'nose2d': 33
    }

    kinect_indices = {
        'eyeInnerCornerLeft3d': 210,
        'eyeOuterCornerLeft3d': 469,
        'eyeInnerCornerRight3d': 843,
        'eyeOuterCornerRight3d': 1117,
        'nose3d': 18
    }

    frame_basler = snapshot['frames']['basler']

    persons_dlib = face_detector.detect_persons(frame_basler, origin=scene.origin)

    person_kinect = Person('kinect', origin=scene.origin)
    person_kinect.set_kinect_landmarks3d(snapshot['data']['face_points'])

    person_kinect.set_gazes_to_mark(snapshot['gaze'])

    person_kinect.raw_dlib_landmarks = persons_dlib[0].raw_dlib_landmarks

    dlib_landmarks = {key: to_xyz_dict(person_kinect.raw_dlib_landmarks[value]) for key, value in dlib_indices.items()}
    kinect_landmarks = {key: to_xyz_dict(snapshot['data']['face_points'][value]) for key, value in
                        kinect_indices.items()}

    for eye in ['left', 'right']:
        true_gaze = norm(person_kinect.get_eye_gaze(eye))
        data = snapshot['data']['est_gazes']

        data['gazeTrue' + eye.capitalize()] = to_xyz_dict(true_gaze)
        data['eyeSphereCenter' + eye.capitalize()] = to_xyz_dict(person_kinect.get_eye_center(eye))
    data.update(dlib_landmarks)
    data.update(kinect_landmarks)

    return data


def write_meta_data(session_path, output_path, face_detector, scene, markers):

    session_code = os.path.split(session_path)[-1]

    parser = ExperimentParser(session_code=session_code)
    parser.fit(session_path, scene)
    markers = np.array(markers)

    csv_data = ''
    write_title = True

    # iterate on data
    for marker, ((frames, data), i) in zip(markers, parser.snapshots_iterate(progress_bar=True)):
        snapshot = {
            'frames': frames,
            'data': data,
            'gaze': marker
        }

        data = form_data(snapshot, face_detector=face_detector, scene=scene)

        line = get_line(data)
        # write title if first line
        if write_title:
            title = get_line(data, title=True)
            csv_data += title + '\n'
            write_title = False

        csv_data += line + '\n'

    # write data
    with open(os.path.join(output_path, session_code + '.csv'), 'w') as file:
        file.write(csv_data)
