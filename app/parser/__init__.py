from os import path as Path
from os import listdir
import json
from cv2 import imread


class ExperimentParser:

    def __init__(self, path_to_frames, path_to_face_points, path_to_face_poses, path_to_gazes):
        self.path_to_frames = path_to_frames
        self.path_to_face_points = path_to_face_points
        self.path_to_face_poses = path_to_face_poses
        self.path_to_gazes = path_to_gazes
        self.indices = [Path.splitext(frame_index)[0] for frame_index in listdir(self.path_to_frames)]

    def read_sample(self, index):
        frame_file = Path.join(self.path_to_frames, self.indices[index] + '.png')
        face_points_file = Path.join(self.path_to_face_points, self.indices[index] + '.txt')
        face_poses_file = Path.join(self.path_to_face_poses, self.indices[index] + '.txt')
        gazes_file = Path.join(self.path_to_gazes, self.indices[index] + '.txt')

        frame = None
        face_points = None
        faces_rotations = None
        gaze_point = None

        if Path.isfile(frame_file):
            frame = imread(frame_file)
        if Path.isfile(face_points_file):
            with open(face_points_file, 'r') as face_points:
                face_points = json.load(face_points)
        if Path.isfile(face_poses_file):
            with open(face_poses_file, 'r') as face_poses:
                face_poses = json.load(face_poses)
            faces_rotations = [face_pose['FaceRotationQuaternion'] for face_pose in face_poses]
        if Path.isfile(gazes_file):
            with open(gazes_file, 'r') as file:
                gaze = json.load(file)
            if int(gaze['REC']['FPOGV']) or True:
                gaze_point = gaze['REC']['FPOGX'], gaze['REC']['FPOGY']

        return frame, face_points, faces_rotations, gaze_point

    def samples_iterate(self, indices=None):
        """
        Dataset generator
        :param indices: indices of samples, according to BRS
        :return: yield tuple(frame, face_points, faces_rotations)
        """
        for sample in indices:
            yield self.read_sample(sample)
