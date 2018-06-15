from app.estimation import *
from config import *
import numpy as np
import os

DATASET_PATH = '../normalized_data/'
SESSIONS = sorted(os.listdir(DATASET_PATH))
print(SESSIONS)

datasetparser = DatasetParser(images='dataset/{index}/0/eyes/{eye}/image',
                              poses='dataset/{index}/0/rotation_norm',
                              gazes='dataset/{index}/0/eyes/{eye}/gaze_norm')

val_split_ratio = 0.8

train_arrays = [], [], []
val_arrays = [], [], []

print(f'train: {SESSIONS[:-3]}')
print(f'test: {SESSIONS[-3:]}')

for SESS in SESSIONS[0:1]:
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

train_eyes, train_poses, train_gazes = tuple(map(np.array, train_arrays))
del train_arrays
val_eyes, val_poses, val_gazes = tuple(map(np.array, val_arrays))
del val_arrays

gaze_estimator = GazeNet()  # .init(PATH_TO_ESTIMATOR)

gaze_estimator.train(create_new=True,
                     path_to_save='./checkpoints',
                     x=[train_eyes, train_poses],
                     y=train_gazes,
                     validation_data=([val_eyes, val_poses], val_gazes),
                     batch_size=32,
                     epochs=5000)

IMAGES_PATH = os.path.join(DATASET_PATH, SESSIONS[-1])
with open(os.path.join(os.path.join(IMAGES_PATH), 'normalized_dataset.json'), 'r') as session_data:
    datasetparser.fit(jsonfile=session_data, path_to_images=IMAGES_PATH)
    test_eyes, test_poses, test_gazes = datasetparser.get_full_data()

print('Score: ', gaze_estimator.model.evaluate([test_eyes, test_poses], test_gazes, batch_size=128))