import numpy as np
import plotly.graph_objects as go
import glob
import natsort
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from time import sleep
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

sig_dir = 'D:/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
trajectories = 120

def animate(i):
    line_range.set_array(abs(range_[i,:,:range_.shape[2]//4,0]))
    line_velo.set_array(abs(velocity_[i,:,:range_.shape[2]//4,0]))
    return line_range, line_velo

def range_fft(iq):
    range_ = np.fft.fft(iq, axis=2)
    return range_

def doppler_fft(range_):
    velocity_ = np.fft.fftshift(np.fft.fft(range_, axis=1), axes=1)
    return velocity_

for i in range(trajectories):
    global range_, velocity_

    sig_f = sig_dir + 'raw_iq_w_mti_' + str(i+1) + '.npy'
    iq = np.load(sig_f)
    range_ = range_fft(iq)
    velocity_ = doppler_fft(range_)

    fig = plt.figure(1)

    ax_r = fig.add_subplot(121)
    line_range = ax_r.imshow(abs(range_[0,:,:range_.shape[2]//4,0]), aspect= 'auto', cmap='jet', interpolation='catrom')
    ax_v = fig.add_subplot(122)
    line_velo = ax_v.imshow(abs(velocity_[0,:,:range_.shape[2]//4,0]), aspect='auto', cmap='jet' , interpolation='catrom')
    
    ani = FuncAnimation(fig, animate, frames = 340, interval = 100)
    plt.show()