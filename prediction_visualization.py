import glob
import natsort
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from time import sleep
import numpy as np

expect_r_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4.npy'
expect_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4.npy'
label_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/label_z_%4.npy'
ground_truth = 'D:/data_signal_MTI/project_util_3/label_all/'

colors_1 = [1.0,0,0,0.5]
colors_2 = [0,1.0,0,0.5]
colors_3 = [0,0,1.0,0.5]
colors_4 = [1,1,1,0.5]
label = []
c = 0
for i in range(120):
    c += 1
    f_name = ground_truth + 'label_' + str(c) +'.npy'
    label.append(np.load(f_name))
label = np.array(label)
label = label/10

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

# expect_r = np.load(expect_r_file)
# expect_z = np.load(expect_z_file)
# label_z = np.load(label_z_file)
# print(label_z.shape)

# x = label_z[:,0] * np.cos(label_z[:,2]) * np.sin(label_z[:,1])
# y = label_z[:,0] * np.sin(label_z[:,2]) + 100
# z = label_z[:,0] * np.cos(label_z[:,2]) * np.cos(label_z[:,1])

# expect_xyz = np.array([[x, y, z]])
# expect_xyz = np.swapaxes(np.swapaxes(expect_xyz,0,2),1,2)
# print(expect_xyz.shape)
# expect_xyz = expect_xyz/10

# xp = expect_r * np.cos(label_z[:,2]) * np.sin(expect_z)
# yp = expect_r * np.sin(label_z[:,2]) + 100
# zp = expect_r * np.cos(label_z[:,2]) * np.cos(expect_z)

# expect_p = np.array([[xp,yp,zp]])
# expect_p = np.swapaxes(np.swapaxes(expect_p,0,2),1,2)
# expect_p = expect_p/10
# for i in range(200):
#     print(expect_r[4200+i] - label_z[4200+i,0], 4200+i)

sp0 = gl.GLScatterPlotItem(pos=label[0:10,:], color=colors_2, pxMode=True, size=20)
w.addItem(sp0)
sp1 = gl.GLScatterPlotItem(pos=label[10:20], color = colors_1, pxMode=True, size=20)
w.addItem(sp1)
sp2 = gl.GLScatterPlotItem(pos=label[20:30], color = colors_3, pxMode=True, size=20)
w.addItem(sp2)
sp3 = gl.GLScatterPlotItem(pos=label[30:40], color = colors_4, pxMode=True, size=20)
w.addItem(sp3)



# i = 8000
# def update():
#     global i
#     sp0 = gl.GLScatterPlotItem(pos=expect_xyz[i], color=colors_1, size=20)
#     w.addItem(sp0)
#     sp2 = gl.GLScatterPlotItem(pos=expect_p[i], size=20)
#     w.addItem(sp2)
#     i += 2

# time = QtCore.QTimer()
# time.timeout.connect(update)
# time.start(5)


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()