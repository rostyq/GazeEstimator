# coding: utf-8

import pandas as pd
from preprocess import *
import os
from nn import *

# If there is no dataset and utils:
# if 'utils.py' not in os.listdir():
#     os.system('wget https://raw.githubusercontent.com/rostyslavb/GazeEstimator/master/model/utils.py')
# if 'MPIIGaze' not in os.listdir():
#     if 'MPIIGaze.tar.gz' not in os.listdir():
#         os.system('wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz')
#     os.system('tar -xzfv MPIIGaze.tar.gz')
#     # os.system('rm MPIIGaze.tar.gz')

# Prepare data
indices = pd.read_csv('./indices.csv')
data = gather_data(indices, batch_size=500, random_state=42)
train_images, train_poses, train_gazes, test_images, test_poses, test_gazes = data

# shuffle
train_idx = np.arange(len(train_gazes))
test_idx = np.arange(len(test_gazes))

np.random.shuffle(train_idx)
np.random.shuffle(test_idx)
train_gazes, test_gazes = train_gazes[train_idx], test_gazes[test_idx]
train_poses, test_poses = train_poses[train_idx], test_poses[test_idx]
train_images, test_images = train_images[train_idx], test_images[test_idx]

# Create NN
model = create_model(learning_rate=0.01, seed=42)
# callbacks = create_callbacks()
# callbacks[0].set_model(model)

# Train
model.fit(
    x=[train_images, train_poses], y=train_gazes,
    batch_size=64,
    verbose=1,
    epochs=10,
    validation_data=([test_images, test_poses], test_gazes),
    #callbacks=callbacks
    )

model.save('./checkpoints/model_last.h5')
