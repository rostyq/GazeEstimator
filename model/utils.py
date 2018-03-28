import numpy as np
from scipy.io import loadmat
import glob
from cv2 import Rodrigues
from tqdm import tqdm
from keras import backend as K
import tensorflow as tf


def gather_eye_data(path, eye='right'):
    
    mat_files = glob.glob(f'{path}/**/*.mat', recursive=True)
    mat_files.sort()
    
    indices = []
    images = []
    poses = []
    gazes = []
    for file in tqdm(mat_files):
        matfile = loadmat(file)
        
        file_idx = file.split('/')[-2], file.split('/')[-1].split('.')[0]
        
        indices.extend([[*file_idx, jpg[0][0], eye] for jpg in matfile['filenames']])
        images.extend(matfile['data'][eye][0, 0]['image'][0, 0])
        poses.extend(matfile['data'][eye][0, 0]['pose'][0, 0])
        gazes.extend(matfile['data'][eye][0, 0]['gaze'][0, 0])
    
    indices = np.array(indices)
    images = np.array(images).reshape((-1, 36, 60, 1)).astype(np.float32) / 255
    poses = np.array(poses).astype(np.float32)
    gazes = np.array(gazes).astype(np.float32)
    
    return indices, images, poses, gazes


def gather_all_data(path):
    
    mat_files = glob.glob(f'{path}/**/*.mat', recursive=True)
    mat_files.sort()
    
    index = dict(left=list(), right=list())
    image = dict(left=list(), right=list())
    pose = dict(left=list(), right=list())
    gaze = dict(left=list(), right=list())
    
    for file in tqdm(mat_files):
        # read file
        matfile = loadmat(file)
        
        # file name
        file_idx = file.split('/')[-2], file.split('/')[-1].split('.')[0]
        for eye in ['left', 'right']:
            index[eye].extend([[*file_idx, jpg[0][0], eye] for jpg in matfile['filenames']])
            image[eye].extend(matfile['data'][eye][0, 0]['image'][0, 0])
            pose[eye].extend(matfile['data'][eye][0, 0]['pose'][0, 0])
            gaze[eye].extend(matfile['data'][eye][0, 0]['gaze'][0, 0])
    
    index = np.stack(tuple(index.values())).reshape((-1, 4))
    image = np.stack(tuple(image.values())).reshape((-1, 36, 60, 1))
    pose = np.stack(tuple(pose.values())).reshape((-1, 3))
    gaze = np.stack(tuple(gaze.values())).reshape((-1, 3))
    return index, image, pose, gaze


def stack_eyes_data(left_array, right_array):
    return np.stack((left_array, right_array)).T.flatten()


def gather_data(path):
    data = (gather_eye_data(path, eye=eye) for eye in ['right', 'left'])
    for left, right in zip(data[0], data[1]):
        stack_eyes_data(left, right)

        
def gaze3Dto2D(array, stack=True):
    """
    theta = asin(-y)
    phi = atan2(-x, -z)
    """
    if array.ndim == 2:
        assert array.shape[1] == 3
        x, y, z = (array[:, i]for i in range(3))
    elif array.ndim == 1:
        assert array.shape[0] == 3
        x, y, z = (array[i] for i in range(3))
        
    theta = np.arcsin(-y)
    phi = np.arctan2(-x, -z)
    
    if not stack:
        return theta, phi
    elif stack:
        return np.column_stack((theta, phi))
    
    
def gaze2Dto3D(array):
    """
    x = (-1) * cos(theta) * sin(phi) 
    y = (-1) * sin(theta)
    z = (-1) * cos(theta) * cos(phi)
    """
    if array.ndim == 2:
        assert array.shape[1] == 2
        theta, phi = (array[:, i] for i in range(2))
    elif array.ndim == 1:
        assert array.shape[0] == 2
        theta, phi = (array[i] for i in range(2))
    
    x = (-1) * np.cos(theta) * np.sin(phi)
    y = (-1) * np.sin(theta)
    z = (-1) * np.cos(theta) * np.cos(phi)

    return np.column_stack((x, y, z))


def pose3Dto2D(array):
    """
    M = Rodrigues((x,y,z))
    Zv = (the third column of M)
    theta = asin(Zv[1])
    phi = atan2(Zv[0], Zv[2])
    """
    def convert_pose(vect):
        M, _ = Rodrigues(np.array(vect).astype(np.float32))
        Zv = M[:, 2]
        theta = np.arctan2(Zv[0], Zv[2])
        phi = np.arcsin(Zv[1])
        return np.array([theta, phi])
    
    return np.apply_along_axis(convert_pose, 1, array)


def angle_accuracy(target, predicted):

    def to_vector(array):
        
        x = (-1)*K.cos(array[:, 0]) * K.sin(array[:, 1])
        y = (-1)*K.sin(array[:, 0])
        z = (-1)*K.cos(array[:, 0]) * K.cos(array[:, 1])
        
        return (x, y, z)
    
    def calc_norm(vector):
        x, y, z = vector
        return K.sqrt(K.square(x) + K.square(y) + K.square(z))
    
    def calc_angle(vector1, vector2):
        
        x1, y1, z1 = vector1
        x2, y2, z2 = vector2
        norm1, norm2 = calc_norm(vector1), calc_norm(vector2)

        return (x1*x2 + y1*y2 + z1*z2) / (norm1*norm2)
    
    v1, v2 = to_vector(target), to_vector(predicted)
    angle_value = calc_angle(v1, v2)
    
    return K.mean(tf.acos(angle_value) * 180 / 3.1415926)
