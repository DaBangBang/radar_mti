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


expect_r_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4_music_pad_nocut+3600.npy'
expect_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4_music_pad_nocut+3600.npy'
label_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/label_z_%4.npy'

expect_r = np.load(expect_r_file)
expect_z = np.load(expect_z_file)
label_z = np.load(label_z_file)

xp = expect_r * np.cos(label_z[:,2]) * np.sin(expect_z)
yp = expect_r * np.sin(label_z[:,2]) + 100
zp = expect_r * np.cos(label_z[:,2]) * np.cos(expect_z)

expect_p = np.array([[xp,yp,zp]])
expect_p = np.swapaxes(expect_p,0,2)
f_r = np.load('D:/data_signal_MTI/project_util_3/prediction_result/RT_2dfft/rotation.npy')
f_t = np.load('D:/data_signal_MTI/project_util_3/prediction_result/RT_2dfft/translation.npy')
expect_p = expect_p.reshape(-1,3)@f_r.T + f_t

y_offset = 100
r = np.sqrt(expect_p[:,0]**2 + (expect_p[:,1] - y_offset)**2 + expect_p[:,2]**2)
zeta = np.arctan2(expect_p[:,0], expect_p[:,2])
# print(expec.shape)
label_r = label_z[:,0]
label_z = label_z[:,1]

r_rmse = np.sqrt(np.mean(r - label_r)**2)
z_rmse = np.sqrt(np.mean(zeta - label_z)**2)

print(r_rmse, z_rmse)