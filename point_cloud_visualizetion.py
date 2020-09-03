import numpy as np
import plotly.graph_objects as go
import glob
import natsort
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from time import sleep
import sys
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
expect_r_file = 'D:/data_signal_MTI/project_util/test_data/expect_r_%4.npy'
expect_z_file = 'D:/data_signal_MTI/project_util/test_data/expect_z_%4.npy'
label_r_file = 'D:/data_signal_MTI/project_util/test_data/label_r_%4.npy'
label_z_file = 'D:/data_signal_MTI/project_util/test_data/label_z_%4.npy'

dbf_r_file = 'D:/data_signal_MTI/project_util/test_data/dbf_r%4.npy'
dbf_z_file = 'D:/data_signal_MTI/project_util/test_data/dbf_z%4.npy'

music_r_file = 'D:/data_signal_MTI/project_util/test_data/music_r%4.npy'
music_z_file = 'D:/data_signal_MTI/project_util/test_data/music_z%4.npy'
esprit_z_file = 'D:/data_signal_MTI/project_util/test_data/esprit_z%4.npy'


colors_1 = [1.0,0,0,0.5]
colors_2 = [0,1.0,0,0.5]

## ====== prediction ======


expect_r = np.load(expect_r_file)
expect_z = np.load(expect_z_file)
label_r = np.load(label_r_file)
label_z = np.load(label_z_file)
dbf_r = np.load(dbf_r_file)
music_r = np.load(music_r_file)
dbf_z = np.load(dbf_z_file)
music_z = np.load(music_z_file)
esprit_z = np.load(esprit_z_file)
label_z = label_z[:,1].reshape(-1)
# print(expect_r.shape, label_r[0], label_z[0,0])
print(expect_z.shape)

music_z = music_z.reshape(-1)
mean_music_z = np.sqrt(np.mean((label_z - music_z[::-1])**2))

esprit_z = esprit_z.reshape(-1)
mean_esprit_z = np.sqrt(np.mean((label_z - esprit_z[::-1])**2))
print(mean_music_z, mean_esprit_z)


# x = expect_r * np.cos(label_z[:,2].reshape(-1)) * np.sin(expect_z)
# z = expect_r * np.cos(label_z[:,2].reshape(-1)) * np.cos(expect_z)

# xl = label_r * np.cos(label_z[:,2].reshape(-1)) * np.sin(label_z[:,1].reshape(-1))
# zl = label_r * np.cos(label_z[:,2].reshape(-1)) * np.cos(label_z[:,1].reshape(-1))
expect_r = expect_r[::10]
expect_z = expect_z[::10]
label_r = label_r[::10]
label_z = label_z[::10]
dbf_r = dbf_r[::10]
music_r = music_r[::10]
music_r = music_r.reshape(-1)
dbf_z = dbf_z[::10]
music_z = music_z[::10]
music_z = music_z.reshape(-1)
esprit_z = esprit_z[::10]
esprit_z = esprit_z.reshape(-1)
print(esprit_z.shape)





N = expect_r.shape[0]
xn = np.arange(N)
p = label_r.argsort()
k = label_z.argsort()


fig = make_subplots(rows=2, cols=1)

# fig.add_trace(go.Scatter(x=xn, y=label_r[p],
#                     mode='lines',
#                     name='label'), row=1, col=1)
# fig.add_trace(go.Scatter(x=xn, y=expect_r[p],
#                     mode='markers',
#                     name='our_model'), row=1, col=1)
# fig.add_trace(go.Scatter(x=xn, y=dbf_r[p],
#                     mode='markers',
#                     name='2D-FFT'), row=1, col=1)
# fig.add_trace(go.Scatter(x=xn, y=music_r[p],
#                     mode='markers',
#                     name='MUSIC'), row=1, col=1)

fig.add_trace(go.Scatter(x=xn, y=label_z[k],
                    mode='lines',
                    name='lines'), row=1, col=1)
# fig.add_trace(go.Scatter(x=xn, y=expect_z[k],
#                     mode='markers',
#                     name='our_model'), row=1, col=1)
# fig.add_trace(go.Scatter(x=xn, y=dbf_z[k],
#                     mode='markers',
#                     name='markers'), row=1, col=1)
# fig.add_trace(go.Scatter(x=xn, y=music_z[k][::-1],
#                     mode='markers',
#                     name='music'), row=1, col=1)
fig.add_trace(go.Scatter(x=xn, y=esprit_z[k][::-1],
                    mode='markers',
                    name='ESPRIT'), row=1, col=1)
fig.update_layout(
    title="Range_plot",
    xaxis_title="Number of data point",
    yaxis_title="Azimuth(radian)",
    # legend_title="Legend Title",
    # font=dict(
    #     family="Courier New, monospace",
    #     size=18,
    #     color="RebeccaPurple"
    # )
)


fig.show()