from app.estimation import *
from config import *
import numpy as np
import os

DATASET_PATH = '../normalized_data/'

SESSIONS = os.listdir(DATASET_PATH)
print(SESSIONS)

datasetparser = DatasetParser(images='dataset/{index}/eyes/{eye}/image',
                              poses='dataset/{index}/rotation_norm',
                              gazes='dataset/{index}/eyes/{eye}/gaze_norm')

gaze_estimator = GazeNet().init('./checkpoints/custom_loss_mean_pose/model_3900_0.1439.h5')  # .init(PATH_TO_ESTIMATOR)

for SESSION in SESSIONS:
    IMAGES_PATH = os.path.join(DATASET_PATH, SESSION)
    with open(os.path.join(os.path.join(IMAGES_PATH), 'normalized_dataset.json'), 'r') as session_data:
        datasetparser.fit(jsonfile=session_data, path_to_images=IMAGES_PATH)
        test_eyes, test_poses, test_gazes = datasetparser.get_full_data()
        left_score = gaze_estimator.model.evaluate([test_eyes[:1000], test_poses[:1000]],
                                                   test_gazes[:1000],
                                                   batch_size=32, verbose=0)
        right_score = gaze_estimator.model.evaluate([test_eyes[1000:], test_poses[1000:]],
                                                    test_gazes[1000:],
                                                    batch_size=32, verbose=0)
    print(f'Session: {SESSION}')
    print(f'\tRight Angle Error: {right_score[1]:.2f}')
    print(f'\tLeft Angle Error: {left_score[1]:.2f}\n')