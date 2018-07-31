from os import path as Path
from os import listdir

import json
import bson

from cv2 import imread
from cv2 import flip
from cv2 import blur

from app.frame import Frame

from numpy import array
from numpy import sqrt
from numpy import zeros

from collections import OrderedDict
from tqdm import tqdm


def dict_point_to_3d_array(dct, flip_vector=array([1, 1, 1])):
    return array([dct['X'], dct['Y'], dct['Z']]).reshape(3) * flip_vector


def dict_point_to_2d_array(dct, flip_vector=array([1, 1]), trans_vector=array([0, 0])):
    return array([dct['X'], dct['Y']]).reshape(2) * flip_vector + trans_vector


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


class SessionReader:

    cams_map = {
        'Kinect.Color': 'color',
        'InfraredCamera': 'basler',
        'WebCamera': 'web_cam',
        'Kinect.Infrared': 'ir'
    }
    data_map = {
        # 'Kinect.Body': 'face_poses',
        'Gazepoint': 'gazes',
        'cam_100': 'est_gazes',
        'Kinect.FaceVertices': 'face_points'
    }

    device_mapping = 'DeviceMapping.txt'

    def __init__(self, cams_map=None, data_map=None, device_mapping=None):

        # path to folder with DataSource
        self.path_to_data = None

        # data mapping, if None take default
        if cams_map:
            self.cams_map = cams_map
        if data_map:
            self.data_map = data_map
        if device_mapping:
            self.device_mapping = device_mapping

        self.cam_dirs = {}
        self.data_dirs = {}

        # snapshots data
        self.snapshots = None

        # session name
        self.session_code = None

        self.cams = None

    def fit(self, session_code, path_to_dataset, cams, by='est_gazes'):

        self.session_code = session_code
        self.path_to_data = Path.join(path_to_dataset, 'DataSource')
        self.read_device_mapping(path_to_dataset=path_to_dataset)

        self.cams = cams

        if by in self.cam_dirs.keys():
            source = self.cam_dirs
        else:
            source = self.data_dirs

        self.snapshots = sorted([
            Path.splitext(frame_index)[0]
            for frame_index in listdir(Path.join(self.path_to_data, source[by]))
        ])

    def read_device_mapping(self, path_to_dataset):

        # read file
        file_path = Path.join(path_to_dataset, self.device_mapping)
        with open(file_path, mode='r') as mapping:
            mapping = mapping.read().strip('\n').split(sep='\n')

        # map camera dirs
        for line in mapping:

            dir_name, key = map(lambda x: x.strip(), line[:-1].split(';'))
            cam_name = self.cams_map.get(key)
            data_name = self.data_map.get(key)

            if cam_name:
                self.cam_dirs[cam_name] = dir_name
            elif data_name:
                self.data_dirs[data_name] = dir_name
            else:
                pass

        # if there is data from BRS.GazeEstimation
        if Path.exists(Path.join(self.path_to_data, 'cam_100')):
            self.data_dirs['est_gazes'] = 'cam_100'

    def get_cams(self):
        return [cam for cam in self.cam_dirs.keys()]

    def get_data_sources(self):
        return [data_source for data_source in self.data_dirs.keys()]

    def read_frame(self, cam, snapshot, ext='png'):
        frame_file = Path.join(self.path_to_data, self.cam_dirs[cam.name], snapshot + '.' + ext)
        if Path.isfile(frame_file):
            if cam.name == 'web_cam':
                image = flip(imread(frame_file), 0)
            elif cam.name == 'basler':
                image = blur(flip(imread(frame_file), 1), (3, 3))
            else:
                image = flip(imread(frame_file), 1)
            return Frame(cam, image)
        else:
            return None

    def read_frames(self, snapshot):
        return {cam_name: self.read_frame(self.cams[cam_name], snapshot) for cam_name, cam_dir in self.cam_dirs.items()}

    @staticmethod
    def load_json_data(file, data_key):

        # load face points
        if data_key is 'face_points':
            face_points = OrderedDict(bson.loads(file.read()))
            return [dict_point_to_3d_array(face_points[point], flip_vector=array([-1, -1, 1])) for point in
                    face_points.keys()]

        # # face poses
        # elif data_key is 'face_poses':
        #     dct = bson.loads(file.read())
        #     print(dct)
        #     return [quaternion_to_angle_axis(face_pose['FaceRotationQuaternion']) for face_pose in
        #             bson.loads(file.read())]

        # gaze point data
        elif data_key is 'gazes':
            gaze = json.load(file)

            # validate data
            if not gaze or not 'REC' in gaze.keys() or not 'FPOGV' in gaze['REC'].keys() or int(
                    gaze['REC']['FPOGV']) == 0:
                # if int(gaze['REC']['FPOGV']):
                return None
            return {'left': tuple(map(float, (gaze['REC']['LPOGX'], gaze['REC']['LPOGY']))),
                    'right': tuple(map(float, (gaze['REC']['RPOGX'], gaze['REC']['RPOGY'])))}

        # load data from BRS.GazeEstimation
        elif data_key is 'est_gazes':
            est_gaze = json.load(file)
            features_3d = {key: dict_point_to_3d_array(value, flip_vector=array([-1, -1, 1])) for key, value in
                               est_gaze.items() if
                               key in ['gazeLeft', 'gazeCommon', 'gazeRight', 'eyeInnerCornerLeft3d',
                                       'eyeOuterCornerLeft3d', 'eyeInnerCornerRight3d', 'eyeOuterCornerRight3d',
                                       'nose3d', 'eyeSphereCenterLeft', 'eyeSphereCenterRight', 'faceGaze',
                                       'nosePoint']}
            features_2d = {key: dict_point_to_2d_array(value, flip_vector=array([-1, 1]), trans_vector=array([1296, 0])) for key, value in
                               est_gaze.items() if
                               key in ['pupilCenterLeft', 'pupilCenterRight']}
            # est_gaze_result['gazeRight'] = dict_point_to_3d_array(est_gaze['gazeRight'], flip_vector=array([-1, -1, -1]))
            features_3d.update(features_2d)
            return est_gaze
        else:
            raise Exception('Wrong data_key.')

    def read_data(self, snapshot, verbose):
        data = {}
        for data_key, data_dir in self.data_dirs.items():
            if data_key == 'gazes' or data_key == 'est_gazes':
                ext = '.txt'
                mode = 'r'
            else:
                ext = '.dat'
                mode = 'rb'
            try:
                with open(Path.join(self.path_to_data, data_dir, snapshot + ext), mode) as file:
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

    def snapshots_iterate(self, indices=None, verbose=0, progress_bar=False, let_none=False):
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


if __name__ == '__main__':
    pass

