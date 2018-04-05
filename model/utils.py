import numpy as np
from scipy.io import loadmat
import glob
from cv2 import Rodrigues


def dataset_indices(matfiles):
    
    indices = list()
    for file in matfiles:
        p, d = int(file.split('/')[-2][-2:]), int(file.split('/')[-1].split('.')[0][-2:])
        indices.extend([(p, d, i) for i in range(1, len(loadmat(file)['filenames']) + 1)])
    
    return pd.DataFrame(indices, columns=['p', 'd', 'sample'])


def gather_batches(indices, path='./MPIIGaze', batch_size=1000, test_ratio=0.2, random_state=None):
    
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
            mat = loadmat(f'{path}/Data/Normalized/p{p}/day{day}.mat')

            # get samples id-s
            samples = day_df['sample'].values - 1
            
            # get data for left eye
            left_images = mat['data']['left'][0, 0]['image'][0, 0][samples].reshape((-1, 36, 60, 1))
            left_poses = mat['data']['left'][0, 0]['pose'][0, 0][samples]
            left_gazes = mat['data']['left'][0, 0]['gaze'][0, 0][samples]

            # get data for right eye and mirror it
            right_images = np.flip(mat['data']['right'][0, 0]['image'][0, 0][samples], axis=2).reshape((-1, 36, 60, 1))
            right_poses = mat['data']['right'][0, 0]['pose'][0, 0][samples] @ np.diag([-1, 1, 1])
            right_gazes = mat['data']['right'][0, 0]['gaze'][0, 0][samples] @ np.diag([-1, 1, 1])
            
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
            
    train_images = np.array(train_images, subok=True) / 255
    train_poses = np.array(train_poses, subok=True)
    train_gazes = np.array(train_gazes, subok=True)
    
    test_images = np.array(test_images, subok=True) / 255
    test_poses = np.array(test_poses, subok=True)
    test_gazes = np.array(test_gazes, subok=True)
    
    return train_images, train_poses, train_gazes, test_images, test_poses, test_gazes

        
def gaze3Dto2D(array, stack=True):
    """
    theta = asin(-y)
    phi = atan2(-x, -z)
    """
    if array.ndim == 2:
        assert array.shape[1] == 3
        x, y, z = (array[:, i] for i in range(3))
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
        theta = np.arcsin(Zv[1])
        phi = np.arctan2(Zv[0], Zv[2])
        return np.array([phi, theta])
    
    return np.apply_along_axis(convert_pose, 1, array)
    
    
    




