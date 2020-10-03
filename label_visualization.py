import numpy as np
import plotly.graph_objects as go
import glob
import natsort
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from time import sleep


label = 'D:/data_signal_MTI/project_util_3/label_all/'
label_all = []

for i in range(120):
    label_f = label + 'label_' + str(i+1) + '.npy'
    label_pos = np.load(label_f)
    label_all.append(label_pos)
    print(label_pos.shape) 

label_all = np.array(label_all)
label_all = label_all/100

colors_1 = [1.0,0,0,0.5]
colors_2 = [0,1.0,0,0.5]

# test_label_all = []
# train_label_all = []
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

sp0 = gl.GLScatterPlotItem(pos=label_all[85:89], color=colors_2)
w.addItem(sp0)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()