import cv2
import numpy as np
import glob
import pyqtgraph as pg
from numpy.linalg import inv, norm
import sys
from scipy import ndimage, misc, stats
import collections
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import re
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

label_z = []
fold_dir = 'D:/data_signal_MTI/project_util_3/10_fold_validation/100%_data/test_data/test_index_fold_'
expect_r_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4_esprit_pad_fold.npy'
expect_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4_esprit_pad_fold.npy'
label_z_file = 'D:/data_signal_MTI/project_util_3/label_all/label_'
trans_file = 'D:/data_signal_MTI/project_util_3/prediction_result/2dfft_op_param_trans.npy'

def cartesian_to_spherical(label):
    
    y_offset = 100
    r = np.sqrt(label[:,0,0]**2 + (label[:,0,1] - y_offset)**2 + label[:,0,2]**2)
    zeta = np.arctan2(label[:,0,0], label[:,0,2])
    phi = np.arctan2(label[:,0,1] - y_offset , np.sqrt(label[:,0,0]**2 + label[:,0,2]**2))
    # print(zeta)
    # print(r)
    return r, zeta, phi


for i in range(10):
    test_dir = fold_dir + str(i+1) +'.npy'
    fold_num = np.load(test_dir)
    for j in fold_num:
        real_name = label_z_file + str(j+1) + '.npy'
        test_data = np.load(real_name)
        r, zeta, phi = cartesian_to_spherical(test_data)
        label = np.array([r, zeta, phi]).T
        label_z.extend(label)

trans = np.load(trans_file)
label_z = np.array(label_z)
print(label_z.shape)
expect_r = np.load(expect_r_file)
expect_z = np.load(expect_z_file)
print(expect_r.shape)
# label_z = np.load(label_z_file)

# xp = expect_r * np.cos(label_z[:,2]) * np.sin(expect_z)
# yp = expect_r * np.sin(label_z[:,2]) + 100
# zp = expect_r * np.cos(label_z[:,2]) * np.cos(expect_z)

# expect_p = np.array([[xp,yp,zp]])
# expect_p = np.swapaxes(expect_p,0,2)
# f_r = np.load('D:/data_signal_MTI/project_util_3/prediction_result/RT_2dfft/rotation.npy')
# f_t = np.load('D:/data_signal_MTI/project_util_3/prediction_result/RT_2dfft/translation.npy')
# expect_p = expect_p.reshape(-1,3)@f_r.T + f_t

# y_offset = 100
# r = np.sqrt(expect_p[:,0]**2 + (expect_p[:,1] - y_offset)**2 + expect_p[:,2]**2)
# zeta = np.arctan2(expect_p[:,0], expect_p[:,2])
# # print(expec.shape)
expect_r = expect_r + trans[0]
expect_z = expect_z + trans[1]
label_r = label_z[:,0]
label_zeta = label_z[:,1]

r_rmse = np.sqrt(np.mean((label_r - expect_r)**2))
z_rmse = np.sqrt(np.mean((label_zeta - expect_z)**2))
r_abs = np.mean(abs(label_r - expect_r))
z_abs = np.mean(abs(label_zeta - expect_z))

print(r_abs, r_rmse, z_abs, z_rmse)