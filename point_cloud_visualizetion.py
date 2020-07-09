import numpy as np
import plotly.graph_objects as go
import glob
import natsort
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from time import sleep

expect_folder = 'D:/data_signal_MTI/data_ball_move_39_graph/'
folder_name = 'D:/data_signal_MTI/data_ball_move_39_real_imag_clean/p*'
folder_name = glob.glob(folder_name)
folder_name = natsort.natsorted(folder_name)
count = 0

test_label_all = []
train_label_all = []
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

colors_1 = [1.0,0,0,0.5]
colors_2 = [0,1.0,0,0.5]

for f_name in folder_name[:]:
    
    count += 1
    label_name = f_name +'/radar_pos_label_*'
    label_name = glob.glob(label_name)


    if count%6 == 0:
        test_label = np.load(label_name[0])
        test_label = test_label[5:]
        test_label = test_label/10
        test_label_all.extend(test_label)
        # sp0 = gl.GLScatterPlotItem(pos=test_label[10:100], color=colors_1)
        # w.addItem(sp0)
    
    else:
        train_label = np.load(label_name[0])
        train_label = train_label[5:]
        train_label = train_label/10
        train_label_all.extend(train_label)
        # sp1 = gl.GLScatterPlotItem(pos=train_label[10:100], color=colors_2)
        # w.addItem(sp1)
test_label_all = np.array(test_label_all)
train_label_all = np.array(train_label_all)
print(test_label_all.shape, train_label_all.shape)

## ====== prediction ======


expect_r = np.load(expect_folder + 'expect_r_2.npy')
expect_z = np.load(expect_folder + 'expect_z_2.npy')
expect_phi = np.load(expect_folder + 'expect_phi_2.npy')

x = expect_r * np.cos(expect_phi) * np.sin(expect_z)
y = expect_r * np.sin(expect_phi) + 110
z = expect_r * np.cos(expect_phi) * np.cos(expect_z)

expect_xyz = np.array([[x, y, z]])
expect_xyz = np.swapaxes(np.swapaxes(expect_xyz,0,2),1,2)
expect_xyz = expect_xyz/10


# sp0 = gl.GLScatterPlotItem(pos=test_label_all, color=colors_1)
# w.addItem(sp0)
# # sp1 = gl.GLScatterPlotItem(pos=train_label_all[5:], color=colors_2)
# # w.addItem(sp1)
# sp2 = gl.GLScatterPlotItem(pos=expect_xyz)
# w.addItem(sp2)

print(x.shape, y.shape, z.shape, expect_xyz.shape)

i = 0
def update():
    global i
    sp0 = gl.GLScatterPlotItem(pos=test_label_all[i], color=colors_1)
    w.addItem(sp0)
    sp2 = gl.GLScatterPlotItem(pos=expect_xyz[i])
    w.addItem(sp2)
    i += 2

time = QtCore.QTimer()
time.timeout.connect(update)
time.start(5)


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()