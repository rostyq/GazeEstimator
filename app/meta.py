from app import SessionReader
from app import Person
import numpy as np
import os
from tqdm import tqdm


def get_data_coord(data, sep=','):
    result = ''
    if isinstance(data, dict):
        for axis, value in data.items():
            value = str(int(value)) if not value % 1 else format(value, '.9f')
            result += value + sep
    elif isinstance(data, list):
        for axis, value in zip('XYZ', data):
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
        elif isinstance(value, int):
            if title:
                result += key + sep
            else:
                result += str(value) + sep
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

    if persons_dlib:
        person_kinect.raw_dlib_landmarks = persons_dlib[0].raw_dlib_landmarks
    else:
        return None

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


def write_meta_data(session_path, output_path, face_detector, scene, markers, markers_idx):

    session_code = os.path.split(session_path)[-1]

    parser = SessionReader()
    parser.fit(session_code, session_path, scene.cams)
    markers = np.array(markers)

    csv_data = ''
    write_title = True

    # iterate on data
    for idx, marker, ((frames, data), i) in zip(markers_idx, markers, parser.snapshots_iterate(progress_bar=True)):

        snapshot = {
            'frames': frames,
            'data': data,
            'gaze': marker
        }

        data = form_data(snapshot, face_detector=face_detector, scene=scene)

        if data:
            data['markerId'] = int(idx)
            line = get_line(data)

            # write title if first line
            if write_title:
                title = get_line(data, title=True)
                csv_data += title + '\n'
                write_title = False

            csv_data += line + '\n'

    # write data
    with open(os.path.join(output_path, session_code, 'result.csv'), 'w') as file:
        file.write(csv_data)


def meta(scene, face_detector, dataset_path, markers_json, output_path=None):

    if not output_path:
        output_path = dataset_path

    markers = []
    markers_idx = []
    counter = 0
    for i in range(3):
        for j in range(8):
            marker = markers_json.get(f'wall_{i}_dot_{j+1}')
            if marker:
                markers.extend([marker] * 100)
                markers_idx.extend([counter] * 100)
                counter += 1

    for session in sorted(os.listdir(dataset_path)):
        session_path = os.path.join(dataset_path, session)
        write_meta_data(session_path, output_path, face_detector, scene, markers, markers_idx)
