import numpy as np
import plotly.graph_objects as go
import glob
import natsort
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from time import sleep

expect_r_file = 'D:/data_signal_MTI/project_util_3/test_data/expect_r_%4.npy'
expect_z_file = 'D:/data_signal_MTI/project_util_3/test_data/expect_z_%4.npy'
label_r_file = 'D:/data_signal_MTI/project_util_3/test_data/label_r_%4.npy'
label_z_file = 'D:/data_signal_MTI/project_util_3/test_data/label_z_%4.npy'

dbf_r_file = 'D:/data_signal_MTI/project_util/test_data/dbf_r%4.npy'
dbf_z_file = 'D:/data_signal_MTI/project_util/test_data/dbf_z%4.npy'

music_z_file = 'D:/data_signal_MTI/project_util/test_data/music_z%4.npy'
esprit_z_file = 'D:/data_signal_MTI/project_util/test_data/esprit_z%4.npy'

colors_1 = [1.0,0,0,0.5]
colors_2 = [0,1.0,0,0.5]

# test_label_all = []
# train_label_all = []
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

expect_r = np.load(expect_r_file)
expect_z = np.load(expect_z_file)
label_r = np.load(label_r_file)
label_z = np.load(label_z_file)
dbf_r = np.load(dbf_r_file)
dbf_z = np.load(dbf_z_file)
music_z = np.load(music_z_file)
esprit_z = np.load(esprit_z_file)
# label_z = label_z[:,1].reshape(-1)
# print(expect_r.shape, label_r[0], label_z[0,0])
print(expect_z.shape)
## ====== prediction ======


# expect_r = np.load(expect_folder + 'expect_r_2.npy')
# expect_z = np.load(expect_folder + 'expect_z_2.npy')
# expect_phi = np.load(expect_folder + 'expect_phi_2.npy')

x = label_z[:,0] * np.cos(label_z[:,2]) * np.sin(label_z[:,1])
y = label_z[:,0] * np.sin(label_z[:,2]) + 105
z = label_z[:,0] * np.cos(label_z[:,2]) * np.cos(label_z[:,1])

expect_xyz = np.array([[x, y, z]])
expect_xyz = np.swapaxes(np.swapaxes(expect_xyz,0,2),1,2)
expect_xyz = expect_xyz/10

xl = dbf_r * np.cos(label_z[:,2]) * np.sin(dbf_z)
yl = dbf_r * np.sin(label_z[:,2]) + 105
zl = dbf_r * np.cos(label_z[:,2]) * np.cos(dbf_z)

test_label_all = np.array([[xl,yl,zl]])
test_label_all = np.swapaxes(np.swapaxes(test_label_all,0,2),1,2)
test_label_all = test_label_all/10


xp = expect_r * np.cos(label_z[:,2]) * np.sin(expect_z)
yp = expect_r * np.sin(label_z[:,2]) + 105
zp = expect_r * np.cos(label_z[:,2]) * np.cos(expect_z)

expect_p = np.array([[xp,yp,zp]])
expect_p = np.swapaxes(np.swapaxes(expect_p,0,2),1,2)
expect_p = expect_p/10

sp0 = gl.GLScatterPlotItem(pos=test_label_all[1000:2000], color=colors_2)
w.addItem(sp0)
sp1 = gl.GLScatterPlotItem(pos=expect_p[1000:2000][::5])
w.addItem(sp1)
sp2 = gl.GLLinePlotItem(pos=expect_xyz[1000:2000], color = colors_1)
w.addItem(sp2)

# print(x.shape, y.shape, z.shape, expect_xyz.shape)

# i = 0
# def update():
#     global i
#     sp0 = gl.GLScatterPlotItem(pos=test_label_all[i], color=colors_1)
#     w.addItem(sp0)
#     sp2 = gl.GLScatterPlotItem(pos=expect_xyz[i])
#     w.addItem(sp2)
#     i += 2

# time = QtCore.QTimer()
# time.timeout.connect(update)
# time.start(5)

fig.add_trace(go.Scatter(x=x[15040:15140], y=z[15040:15140], mode= 'markers', marker=dict(size=10), name="Ground Truth"))
fig.add_trace(go.Scatter(x=xp[15040:15140], y=zp[15040:15140], mode= 'markers', marker=dict(size=10), name="Esprit"))
fig.update_layout(xaxis_title = "X (mm)", yaxis_title="Y (mm)", xaxis_range =[-150,150], yaxis_range =[0,250]
)
if not os.path.exists("images"):
    os.mkdir("images")
fig.write_image("images/triangle_Esprit.jpg")
fig.show()

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()