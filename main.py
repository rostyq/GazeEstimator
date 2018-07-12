import os
import sys

from config import *
from app.estimation import PersonDetector
from app import Scene


face_detector = PersonDetector(**ACTOR_DETECTOR)

scene = Scene(origin_name=ORIGIN_CAM, intrinsic_params=INTRINSIC_PARAMS, extrinsic_params=EXTRINSIC_PARAMS)


def gather():

    from app.utils import experiment_without_BRS

    if len(sys.argv) == 2:
        DATASET_PATH = sys.argv[1]

    experiment_without_BRS('../',
                           face_detector,
                           scene,
                           'Rostyslav_Bohomaz_right_screen',
                           size='_72_120',
                           dataset_size=500)


def postprocess():

    from app import ExperimentParser
    from app.utils import create_learning_dataset

    parser = ExperimentParser(session_code=os.path.split(DATASET_PATH)[-1])
    parser.fit(DATASET_PATH, scene)

    create_learning_dataset('../brs/', parser, face_detector, scene, indices=range(len(parser.snapshots)))


def visualize():

    from cv2 import imread
    from app.utils import visualize_predict

    back = imread('../screen1.png')
    visualize_predict(face_detector, scene, 'checkpoints/LRE_ff_gp+brs_batch_norm/model_1300_0.0021.h5', back=back)


def train():

    from app.estimation import DatasetParser
    from app.estimation import GazeNet
    import numpy as np
    from sklearn.cluster import DBSCAN

    DATASET_PATH = '../normalized_data_72_120/'
    SESSIONS = sorted(os.listdir(DATASET_PATH))
    print(SESSIONS)
    datasetparser = DatasetParser(**DATASET_PARSER)

    val_split_ratio = 0.8

    train_arrays = [], [], [], []
    val_arrays = [], [], [], []
    # debug_idx = SESSIONS.index('Rostyslav_Bohomaz_right_screen')
    for SESS in SESSIONS[:1]:
        IMAGES_PATH = os.path.join(DATASET_PATH, SESS)
        print(SESS)
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

    outlier_detector = DBSCAN(eps=0.05, min_samples=2, p=10)
    result = outlier_detector.fit(gazes)

    train_indices = (result.labels_ == 0)[:len(train_gazes)]
    val_indices = (result.labels_ == 0)[len(train_gazes):]

    gaze_estimator = GazeNet()  # .init('checkpoints/LRE_filter_flip_gp+brs_full/model_200_0.0027.h5')

    gaze_estimator.train(create_new=True,
                         path_to_save='./checkpoints',
                         sess_name='LRE_ff_gp+brs_batch_norm',
                         x=[train_eyes[train_indices], train_poses[train_indices]],
                         y=train_gazes[train_indices],
                         # y=np.append(train_gazes[train_indices], train_angles[train_indices], axis=1),
                         validation_data=([val_eyes[val_indices], val_poses[val_indices]], val_gazes[val_indices]),
                         batch_size=64,
                         epochs=10000)


def test():

    from app.estimation import DatasetParser
    from app.estimation import GazeNet

    DATASET_PATH = '../normalized_data/'

    SESSIONS = os.listdir(DATASET_PATH)
    print(SESSIONS)

    datasetparser = DatasetParser(**DATASET_PARSER)

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


def main():

    from app import ExperimentParser
    from app.utils import create_learning_dataset

    if len(sys.argv) == 2:
        DATASET_PATH = sys.argv[1]

    for session, gaze in zip(sorted(os.listdir(DATASET_PATH)), GAZEPOINT_MARKERS):
        full_path = os.path.join(DATASET_PATH, session)
        parser = ExperimentParser(session_code=os.path.split(full_path)[-1])
        parser.fit(full_path, scene)
        create_learning_dataset('../', parser, face_detector, scene, indices=range(len(parser.snapshots)), gaze=gaze)


run_dict = {
    'main': main,
    'visualize': visualize,
    'postprocess': postprocess,
    'train': train,
    'test': test,
    'gather': gather
}

run = sys.argv[1]

if __name__ == "__main__":
    run_dict[run]()
