import numpy as np
import plotly.graph_objects as go
import glob
import natsort
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from time import sleep
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

expect_r_2dfft ='D:/data_signal_MTI/project_util_3/result_for_paper/expect_r_2dfft_pad_100%_fold.npy'
expect_z_2dfft ='D:/data_signal_MTI/project_util_3/result_for_paper/expect_z_2dfft_pad_100%_fold.npy'
expect_r_music ='D:/data_signal_MTI/project_util_3/result_for_paper/expect_r_music_pad_100%_fold.npy'
expect_z_music ='D:/data_signal_MTI/project_util_3/result_for_paper/expect_z_music_pad_100%_fold.npy'
expect_r_esprit ='D:/data_signal_MTI/project_util_3/result_for_paper/expect_r_esprit_pad_100%_fold_cut.npy'
expect_z_esprit ='D:/data_signal_MTI/project_util_3/result_for_paper/expect_z_esprit_pad_100%_fold_cut.npy'
expect_r_model = 'D:/data_signal_MTI/project_util_3/result_for_paper/expec_r_fold_'
expect_z_model = 'D:/data_signal_MTI/project_util_3/result_for_paper/expec_z_fold_'
expect_r_remove = 'D:/data_signal_MTI/project_util_3/result_for_paper/expect_r_remove_outlier.npy'
expect_z_remove = 'D:/data_signal_MTI/project_util_3/result_for_paper/expect_z_remove_outlier.npy'


expect_r = []
expect_z = []

for i in range(10):
    name_r = expect_r_model + str(i+1) + '.npy'
    name_z = expect_z_model + str(i+1) + '.npy'
    print(np.load(name_r).shape)
    expect_r.extend(np.load(name_r))
    expect_z.extend(np.load(name_z))
predict = np.array(expect_r)
predict_zeta = np.array(expect_z)

ground_truth = 'D:/data_signal_MTI/project_util_3/result_for_paper/label_z_2dfft_fold.npy'

T_r_2dfft = 'D:/data_signal_MTI/project_util_3/result_for_paper/optimize_r_2dfft.npy'
T_z_2dfft = 'D:/data_signal_MTI/project_util_3/result_for_paper/optimize_z_2dfft.npy'
T_r_music = 'D:/data_signal_MTI/project_util_3/result_for_paper/optimize_r_music.npy'
T_z_music = 'D:/data_signal_MTI/project_util_3/result_for_paper/optimize_z_music.npy'
T_r_esprit = 'D:/data_signal_MTI/project_util_3/result_for_paper/optimize_r_esprit.npy'
T_z_esprit = 'D:/data_signal_MTI/project_util_3/result_for_paper/optimize_z_esprit.npy'
T_r_2dfft = np.load(T_r_2dfft)
T_z_2dfft = np.load(T_z_2dfft)
T_r_music = np.load(T_r_music)
T_z_music = np.load(T_z_music)
T_r_esprit = np.load(T_r_esprit)
T_z_esprit = np.load(T_z_esprit)



# # print(T_r_2dfft, T_z_2dfft)
predict = np.load(expect_r_music)
predict_zeta = np.load(expect_z_music)
print(predict.shape, predict_zeta.shape)
g_t = np.load(ground_truth)
print(max(g_t[:,0]), min(g_t[:,0]))
outlier_ = np.max(g_t[:,0])

# for i in range(predict.shape[0]):
#     actual_zeta = predict_zeta[i]
#     if actual_zeta > 0.40 or actual_zeta < -0.80:
#         predict[i] = np.nan
#         predict_zeta[i] = np.nan
        
# k = ~np.isnan(predict)
# predict = predict[k]
# predict_zeta = predict_zeta[k]


# g_t = g_t[k]

print(predict.shape, predict_zeta.shape, g_t.shape)

predict += T_r_music[0] 
predict_zeta += T_z_music[0]

r_mse = np.sqrt(np.mean((predict - g_t[:,0])**2))
z_mse = np.sqrt(np.mean((predict_zeta - g_t[:,1])**2))
print(r_mse, z_mse)

# xp = predict * np.cos(g_t[:,2]) * np.sin(predict_zeta)
# yp = predict * np.sin(g_t[:,2]) + 100
# zp = predict * np.cos(g_t[:,2]) * np.cos(predict_zeta)

# x = g_t[:,0] * np.cos(g_t[:,2]) * np.sin(g_t[:,1])
# y = g_t[:,0] * np.sin(g_t[:,2]) + 100
# z = g_t[:,0] * np.cos(g_t[:,2]) * np.cos(g_t[:,1])

# fig = go.Figure()

# fig.add_trace(go.Scatter(x=x[37220:37340], y=z[37220:37340], mode= 'markers', marker=dict(size=10), name="Ground Truth"))
# fig.add_trace(go.Scatter(x=xp[37220:37340], y=zp[37220:37340], mode= 'markers', marker=dict(size=10), name="Esprit"))
# fig.update_layout(xaxis_title = "X (mm)", yaxis_title="Y (mm)", xaxis_range =[-150,150], yaxis_range =[0,250]
# )
# if not os.path.exists("images"):
#     os.mkdir("images")
# fig.write_image("images/triangle_Esprit.jpg")
# fig.show()

# print(predict.shape, g_t.shape,np.max(x),np.max(z), np.max(predict), np.max(zp))