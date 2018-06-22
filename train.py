from app.estimation import *
from config import *
import numpy as np
import os
from sklearn.cluster import DBSCAN


if __name__ == '__main__':

    DATASET_PATH = '../normalized_data/'
    SESSIONS = sorted(os.listdir(DATASET_PATH))
    print(SESSIONS)

    datasetparser = DatasetParser(images='dataset/{index}/eyes/{eye}/image',
                                  poses='dataset/{index}/rotation_norm',
                                  gazes='dataset/{index}/eyes/{eye}/gaze_norm')

    val_split_ratio = 0.8

    train_arrays = [], [], [], []
    val_arrays = [], [], [], []

    for SESS in SESSIONS:
        IMAGES_PATH = os.path.join(DATASET_PATH, SESS)

        with open(os.path.join(IMAGES_PATH, 'normalized_dataset.json'), 'r') as session_data:
            datasetparser.fit(jsonfile=session_data, path_to_images=IMAGES_PATH)

            val_split = int(val_split_ratio*(datasetparser.shape-1))
            indices = np.random.permutation(datasetparser.shape-1)
            train_indices = indices[:val_split]
            val_indices = indices[val_split:]
            for part, idx in zip([train_arrays, val_arrays], [train_indices, val_indices]):
                for data, new_data in zip(part, datasetparser.get_full_data(idx)):
                    data.extend(new_data)

    train_eyes, train_poses, train_gazes, train_angles = tuple(map(np.array, train_arrays))
    del train_arrays
    val_eyes, val_poses, val_gazes, val_angles = tuple(map(np.array, val_arrays))
    del val_arrays

    gazes = np.append(train_gazes, val_gazes, axis=0)
    # poses = np.append(train_poses, val_poses, axis=0)

    outlier_detector = DBSCAN(eps=0.5, min_samples=2, p=10)
    result = outlier_detector.fit(gazes)

    train_indices = (result.labels_ == 0)[:len(train_gazes)]
    val_indices = (result.labels_ == 0)[len(train_gazes):]


    gaze_estimator = GazeNet()  # .init(PATH_TO_ESTIMATOR)

    gaze_estimator.train(create_new=True,
                         path_to_save='./checkpoints',
                         sess_name='filtering_without_brs_both_different_poses',
                         x=[train_eyes[train_indices], train_poses[train_indices]],
                         y = train_gazes[train_indices],
                         # y=np.append(train_gazes[train_indices], train_angles[train_indices], axis=1),
                         validation_data=([val_eyes[val_indices], val_poses[val_indices]], val_gazes[val_indices]),
                         batch_size=128,
                         epochs=10000)
