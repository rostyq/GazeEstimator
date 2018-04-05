from scipy.io import loadmat
import numpy as np
from PIL import Image
import glob
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pylab

# ## Prepare data

# ### Gather Structure

file = '../../MPIIGaze/Data/Normalized/p00/day01.mat'
matfile = loadmat(file)
img = matfile['data']['right'][0, 0]['image'][0, 0][0]
Image.fromarray(img)

img = matfile['data']['left'][0, 0]['image'][0, 0][0]
Image.fromarray(img)

img = matfile['data']['left'][0, 0]['image']
print(matfile['data'].dtype)
print(matfile['data']['left'][0, 0].dtype)
print(matfile['data']['left'][0, 0]['image'][0, 0].dtype)

gaze = pd.DataFrame(matfile['data']['left'][0, 0]['gaze'][0,0])
pose = pd.DataFrame(matfile['data']['left'][0, 0]['pose'][0,0])
pp.ProfileReport(pose * np.pi * 2)

file = '../../MPIIGaze/Data/6 points-based face model.mat'
matfile = loadmat(file)
dataset_model = matfile['model']
tutorial_model =  np.array([(0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                           ]).T/10

dataset_model.shape, tutorial_model.shape

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import numpy as np

trace1 = go.Scatter3d(
    x=dataset_model[0],
    y=dataset_model[1],
    z=dataset_model[2],
    text = np.arange(0,6),
    mode='markers',
    marker=dict(size=12,line=dict(color='rgba(217, 217, 217, 0.14)', width=0.5), opacity=0.8)
)

trace2 = go.Scatter3d(
    x=tutorial_model[0],
    y=tutorial_model[1],
    z=-tutorial_model[2],
    text = np.arange(0,6),
    mode='markers',
    marker=dict(size=12,line=dict(color='rgba(217, 217, 217, 0.14)', width=0.5), opacity=0.8)
)

data = [trace1, trace2]
layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


fig, ax = plt.subplots()
ax.scatter(x = dataset_model[0], y = dataset_model[1])
for i in range(6):
    ax.annotate(str(i), (dataset_model[0][i], dataset_model[1][i]))
pylab.savefig('dataset_model.png')


fig, ax = plt.subplots()
ax.scatter(x = tutorial_model[0], y = tutorial_model[1])
for i in range(6):
    ax.annotate(str(i), (tutorial_model[0][i], tutorial_model[1][i]))
pylab.savefig('tutorial_model.png')

len(mat_files)

image_size = matfile['data']['left'][0, 0]['image'][0, 0][0].shape
image_size

data = [loadmat(file)['data'] for file in mat_files]

def extract_images(data, direction = 'right'):
    result = []
    for sample in data:
        if len(result) == 0:
            result = sample[direction][0,0]['image'][0, 0]
        result = np.append(result, sample[direction][0,0]['image'][0, 0])
    return result

extract_images(data[0:20]).shape

data[]

(0, *image_size)
