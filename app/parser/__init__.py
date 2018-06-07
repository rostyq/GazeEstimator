from os import path as Path
from os import listdir
import json
import bson
from cv2 import imread
from cv2 import flip
from app.frame import Frame
from numpy import array
from numpy import sqrt
from numpy import zeros
from collections import OrderedDict
from tqdm import tqdm


def face_point_to_array(dct):
    return array([dct['X'], dct['Y'], dct['Z']]).reshape(3) * array([-1, -1, 1])


def quaternion_to_angle_axis(quaternion):
    """
    Converts angle-axis to quaternion
    :param quaternion: dict {'X': , 'Y': , 'Z': , 'W': }
    :return: angle-axis rotation vector
    """
    t = sqrt(1 - quaternion['W'] * quaternion['W'])
    if t:
        x = quaternion['X'] / t
        y = quaternion['Y'] / t
        z = quaternion['Z'] / t
        return array([[x], [y], [z]])
    else:
        return zeros((3, 1))


class ExperimentParser:

    def __init__(self, session_code):
        self.path_to_dataset = None
        self.cams_dict = None
        self.data_dict = None
        self.snapshots = None
        self.session_code = session_code

    def fit(self, path_to_dataset, scene):
        self.path_to_dataset = Path.join(path_to_dataset, 'DataSource')
        self.cams_dict, self.data_dict = self.read_device_mapping(Path.join(path_to_dataset, 'DeviceMapping.txt'))
        self.cams_dict = {scene.cams[cam_name]: cam_dir for cam_name, cam_dir in self.cams_dict.items()}
        self.snapshots = sorted([
            Path.splitext(frame_index)[0]
            for frame_index in listdir(Path.join(self.path_to_dataset, list(self.cams_dict.values())[0]))
        ])

    @staticmethod
    def read_device_mapping(file):
        with open(file, mode='r') as mapping:
            mapping = mapping.read().split(sep='\n')
            mapping = {item.split(sep=';')[1]: item.split(sep=';')[0] for item in mapping[:-1]}
            cams_dict = {
                'color': mapping[' Kinect.Color'],
                'basler': mapping[' InfraredCamera'],
                'web_cam': mapping[' WebCamera'],
                'ir': mapping[' Kinect.Infrared']
            }
            data_dict = {
                # 'face_poses': mapping[' Kinect.Face'],
                'gazes': mapping[' Gazepoint'],
                'face_points': mapping[' Kinect.FaceVertices']
            }
            return cams_dict, data_dict

    def read_frame(self, cam, snapshot):
        frame_file = Path.join(self.path_to_dataset, self.cams_dict[cam], snapshot + '.png')
        if Path.isfile(frame_file):
            # if cam.name == 'web_cam':
            #     image = flip(imread(frame_file), 0)
            # else:
            image = flip(imread(frame_file), 1)
            return Frame(cam, image)
        else:
            return None

    def read_frames(self, snapshot):
        return {cam.name: self.read_frame(cam, snapshot) for cam, cam_dir in self.cams_dict.items()}

    @staticmethod
    def load_json_data(file, data_key):
        if data_key is 'face_points':
            face_points = OrderedDict(bson.loads(file.read()))
            return [face_point_to_array(face_points[point]) for point in face_points.keys()]
        if data_key is 'face_poses':
            return [quaternion_to_angle_axis(face_pose['FaceRotationQuaternion']) for face_pose in bson.loads(file.read())]
        if data_key is 'gazes':
            gaze = json.load(file)
            if not gaze or not 'REC' in gaze.keys() or int(gaze['REC']['FPOGV']) == 0:
            # if int(gaze['REC']['FPOGV']):
                return None
            return tuple(map(float, (gaze['REC']['FPOGX'], gaze['REC']['FPOGY'])))
        else:
            raise Exception('Wrong data_key.')

    def read_data(self, snapshot, verbose):
        data = {}
        for data_key, data_dir in self.data_dict.items():
            if data_key == 'gazes':
                ext = '.txt'
                mode = 'r'
            else:
                ext = '.dat'
                mode = 'rb'
            try:
                with open(Path.join(self.path_to_dataset, data_dir, snapshot + ext), mode) as file:
                    data[data_key] = self.load_json_data(file, data_key)
            except FileNotFoundError:
                # TODO add logger
                if verbose:
                    print(f'WARNING: {data_key} {snapshot} in {data_dir} not found.')
                data[data_key] = None
            except AssertionError:
                if verbose:
                    print(f'WARNING: {data_key} {snapshot} have non-valid gaze point.')
                data[data_key] = None
        return data

    def read_snapshot(self, snapshot, verbose):
        return self.read_frames(snapshot), self.read_data(snapshot, verbose)

    def snapshots_iterate(self, indices=None, verbose=0, progress_bar=False):
        """
        Dataset generator
        :param indices: indices of snapshots, according to BRS
        :param verbose: 0 - nothing, 1 - warnings
        :return: yield tuple(frame, face_points, faces_rotations)
        """
        if not progress_bar:
            bar = lambda iterable: iterable
        else:
            bar = tqdm
        if indices is None:
            indices = range(len(self.snapshots))
        for i in bar(indices):
            snapshot_index = self.snapshots[i]
            snapshot_data = self.read_snapshot(snapshot_index, verbose)
            yield snapshot_data, snapshot_index
