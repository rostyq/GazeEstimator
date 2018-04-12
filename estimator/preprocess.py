import numpy as np
from scipy.io import loadmat
import glob
from pandas import DataFrame
from transform import *


def create_dataset_indices(matfiles):

    indices = list()
    for file in matfiles:
        p, d = int(file.split('/')[-2][-2:]), int(file.split('/')[-1].split('.')[0][-2:])
        indices.extend([(p, d, i) for i in range(1, len(loadmat(file)['filenames']) + 1)])

    return DataFrame(indices, columns=['p', 'd', 'sample'])


def gather_data(indices=None, path='./MPIIGaze', batch_size=1000, test_ratio=0.2, random_state=None):

    indices = create_dataset_indices(
        glob.glob(
            path=f'{path}/Data/Normalized/**/*.mat',
            recursive=True
            )
        ) if indices is None else indices

    train_images = list()
    train_poses = list()
    train_gazes = list()

    test_images = list()
    test_poses = list()
    test_gazes = list()

    for p_num, p_df in indices.groupby('p'):
        for day_num, day_df in p_df.sample(batch_size, random_state=random_state).sort_values('sample').groupby('d'):

            # get day file
            p = str(p_num).rjust(2, '0')
            day = str(day_num).rjust(2, '0')
            mat = loadmat(f'{path}/Data/Normalized/p{p}/day{day}.mat', struct_as_record=False, squeeze_me=True)

            # get samples id-s
            samples = day_df['sample'].values - 1

            # get data for left eye
            left_images = mat['data'].left.image[samples].reshape((-1, 36, 60, 1))
            left_poses = pose3Dto2D(mat['data'].left.pose[samples]).tolist()
            left_gazes = gaze3Dto2D(mat['data'].left.gaze[samples]).tolist()

            # get data for right eye and mirror it
            right_images = np.flip(mat['data'].right.image[samples], axis=2).reshape((-1, 36, 60, 1))
            right_poses = (pose3Dto2D(mat['data'].right.pose[samples]) * np.array([-1, 1])).tolist()
            right_gazes = (gaze3Dto2D(mat['data'].right.gaze[samples]) * np.array([-1, 1])).tolist()

            # split train test
            train_size = int(len(samples)*(1 - test_ratio))

            train_images.extend(left_images[:train_size])
            train_images.extend(right_images[:train_size])
            train_poses.extend(left_poses[:train_size])
            train_poses.extend(right_poses[:train_size])
            train_gazes.extend(left_gazes[:train_size])
            train_gazes.extend(right_gazes[:train_size])

            test_images.extend(left_images[train_size:])
            test_images.extend(right_images[train_size:])
            test_poses.extend(left_poses[train_size:])
            test_poses.extend(right_poses[train_size:])
            test_gazes.extend(left_gazes[train_size:])
            test_gazes.extend(right_gazes[train_size:])

    train_images = np.array(train_images, subok=True).astype(np.float64) / 255
    train_poses = np.array(train_poses, subok=True).astype(np.float64)
    train_gazes = np.array(train_gazes, subok=True).astype(np.float64)

    test_images = np.array(test_images, subok=True).astype(np.float64) / 255
    test_poses = np.array(test_poses, subok=True).astype(np.float64)
    test_gazes = np.array(test_gazes, subok=True).astype(np.float64)

    return train_images, train_poses, train_gazes, test_images, test_poses, test_gazes
