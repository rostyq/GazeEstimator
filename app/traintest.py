import os


def train(*args, **kwargs):

    from app.estimation import DatasetParser
    from app.estimation import GazeNet
    from config import DATASET_PARSER
    import numpy as np
    from sklearn.cluster import DBSCAN

    dataset_path = [r'D:\C_Documents\BAS\normalized_data_72_120_filtered', r'D:\C_Documents\BAS\\normalized_data_72_120_raw']
    parser_params = DATASET_PARSER

    SESSIONS = []
    for path in dataset_path:
        SESSIONS.extend(list(map(lambda session: os.path.join(path, session), os.listdir(path))))

    print(SESSIONS)
    datasetparser = DatasetParser(**parser_params)

    val_split_ratio = 0.8

    train_arrays = [], [], [], []
    val_arrays = [], [], [], []
    for SESS in SESSIONS:
        IMAGES_PATH = SESS
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

    # gazes = np.append(train_gazes, val_gazes, axis=0)
    # poses = np.append(train_poses, val_poses, axis=0)

    # outlier_detector = DBSCAN(eps=0.05, min_samples=2, p=10)
    # result = outlier_detector.fit(gazes)

    # train_indices = (result.labels_ == 0)[:len(train_gazes)]
    # val_indices = (result.labels_ == 0)[len(train_gazes):]

    gaze_estimator = GazeNet()  # .init('checkpoints/LRE_filter_flip_gp+brs_full/model_200_0.0027.h5')

    gaze_estimator.train(create_new=True,
                         path_to_save='./checkpoints',
                         sess_name='LRE_ff+brs_batch_norm+full',
                         # x=[train_eyes[train_indices], train_poses[train_indices]],
                         # y=train_gazes[train_indices],
                         x=[train_eyes, train_poses],
                         y=train_gazes,
                         # y=np.append(train_gazes[train_indices], train_angles[train_indices], axis=1),
                         # validation_data=([val_eyes[val_indices], val_poses[val_indices]], val_gazes[val_indices]),
                         validation_data=([val_eyes, val_poses], val_gazes),
                         batch_size=512,
                         epochs=10000)


def test(*args, **kwargs):

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

