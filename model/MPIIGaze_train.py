# coding: utf-8

import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
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

# ## Prepare data

# ### Gather Data from Structure
indices = pd.read_csv('./indices.csv')
train_images, train_poses, train_gazes, test_images, test_poses, test_gazes = gather_batches(indices, random_state=42)

# transform
train_gazes, test_gazes = gaze3Dto2D(train_gazes), gaze3Dto2D(test_gazes)
train_poses, test_poses = pose3Dto2D(train_poses), pose3Dto2D(test_poses)

# ## Create NN
model = create_model()
callbacks = create_callbacks()

# ### Train
model.fit(x=[train_images, train_poses], y=train_gazes,
          batch_size=100,
          verbose=1,
          epochs=1000,
          validation_data=([test_images, test_poses], test_gazes),
          callbacks=callbacks)

model.save('./checkpoints/model_last.h5')
