import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import itertools
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
# should be odd number
padding = 100
n_pad = ((0,0),(0,0),(0,0),(0,padding))

signal_dir = 'D:/data_signal_MTI/signal_test_6'
save_dir = 'D:/data_signal_MTI/signal_test_6/' 

training_cell_along_range = 15
training_cell_along_velocity = 15
training_cell_along_range = int(training_cell_along_range/2)
training_cell_along_velocity = int(training_cell_along_velocity/2)
print(training_cell_along_range, training_cell_along_velocity)

guarding_cell_along_range = 9
guarding_cell_along_velocity = 9
guarding_cell_along_range = int(guarding_cell_along_range/2)
guarding_cell_along_velocity = int(guarding_cell_along_velocity/2)
print(guarding_cell_along_range, guarding_cell_along_velocity )

def plotFuction(velocity_fft, tresh_plane, doppler_tresh):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(221, projection='3d')
    X, Y = np.mgrid[:velocity_fft.shape[1], :velocity_fft.shape[2]]
    ax1.plot_surface(X, Y, abs(velocity_fft[350,:,:,0]), cmap='jet', alpha=0.7)
    ax1.plot_surface(X, Y, tresh_plane[350,:,:,0], alpha=1)

    ax2 = fig.add_subplot(222)
    ax2 = plt.imshow(doppler_tresh[202,:,:,0], aspect='auto')
    

    ax3 = fig.add_subplot(223)
    ax3 = plt.imshow(abs(velocity_fft[202,:,:,0]), aspect= 'auto', origin='lower' , cmap='jet', 
         extent=[0, velocity_fft.shape[2]-1, velocity_fft.shape[1]/2., -velocity_fft.shape[1]/2.], interpolation = 'catrom')
    
    plt.show()

    

def kernel(velocity_fft):
    # select weight
    weight = 7
    
    peak_value = abs(velocity_fft)
    tresh_plane = np.zeros((velocity_fft.shape[0], velocity_fft.shape[1], velocity_fft.shape[2], velocity_fft.shape[3]))
            
    for i, j in itertools.product(range(velocity_fft.shape[1]), range(velocity_fft.shape[2])):
            
            kernel_ = peak_value[:, max(0,i-training_cell_along_velocity):i+training_cell_along_velocity+1, 
                                    max(0,j-training_cell_along_range):j+training_cell_along_range+1, :].copy()
            kernel_[:, max(0,i-guarding_cell_along_velocity):i+guarding_cell_along_velocity+1, 
                                    max(0,j-guarding_cell_along_range):j+guarding_cell_along_range+1, :] = np.nan
            
           
            # =================== chose one algorithm to detect peak ====================
            # CA-CFAR 
            kernel_ = weight*np.nanmean(kernel_, axis=(1,2))
            tresh_plane[:,i,j,:] = kernel_

            # OS-CFAR
            # kernel_ = np.reshape(kernel_, (kernel_.shape[0], -1, kernel_.shape[3]))
            # kernel_ = np.sort(kernel_, axis=1)
            # kernel_nan = np.count_nonzero(np.isnan(kernel_[0,:,0]))
            # N = kernel_.shape[1] - kernel_nan 
            # k_order = round((3/4)*N)     
            # tresh_plane[:,i,j,:] = weight*kernel_[:,k_order,:]
            # ============================================================================
    
    doppler_tresh = (tresh_plane < peak_value)

    np.save(save_dir + '/doppler_tresh_mov', doppler_tresh)
    np.save(save_dir + '/tresh_plane_mov', tresh_plane)
    

    return tresh_plane, doppler_tresh                          

def angleFFT(AoA):
    weight = 1.8
    training_cell_along_aoa = 10
    guarding_cell_along_aoa = 5
    tresh_line = np.zeros((AoA.shape[0], AoA.shape[1], AoA.shape[2], AoA.shape[3]))
    n = AoA.shape[3]
    fft_aoa = np.fft.fftshift(np.fft.fft(AoA, axis=3) / n, axes=3)
    peak_value = abs(fft_aoa)
    # print(np.where(AoA[:,0,0,:] > 0))
    # plt.figure(1)
    # for i in range(fft_aoa.shape[0]):
    #     plt.imshow(abs(fft_aoa[i,32,:,:]), aspect='auto')
    # # plt.imshow(abs(fft_aoa[201,32,:,:]), aspect='auto')
    # # plt.imshow(abs(fft_aoa[195,32,:,:]), aspect='auto')
    # plt.figure(2)
    # plt.plot(abs(fft_aoa[45,32,40,:]))
    # plt.show()

    for m in range(fft_aoa.shape[3]):
        kernel_ = peak_value[:, :, :, max(0,m-training_cell_along_aoa):m+training_cell_along_aoa+1].copy()
        kernel_[:, :, :,  max(0,m-guarding_cell_along_aoa):m+guarding_cell_along_aoa+1] = np.nan
        kernel_ = weight*np.nanmean(kernel_, axis= 3)            
        tresh_line[:,:,:,m] = kernel_

    
    aoa_line = (tresh_line < peak_value)
    fft_aoa_tresh = np.multiply(aoa_line, abs(fft_aoa))

    plt.plot(abs(fft_aoa[45,32,40,:]))
    plt.plot(tresh_line[45,32,40,:])
    plt.plot(fft_aoa_tresh[45,32,40,:])
    plt.show()

    np.save(save_dir + '/fft_aoa_tresh', fft_aoa_tresh)

def main():
    velocity_fft = np.load(signal_dir + '/doppler_fft.npy')
    velocity_fft = velocity_fft[:,:,:,:]
    print(velocity_fft.shape)

    #========== solve doppler tresh and tresh plane ================
    # tresh_plane, doppler_tresh = kernel(velocity_fft)
    # tresh_plane = np.load(save_dir + 'tresh_plane_mov.npy')
    # doppler_tresh = np.load(save_dir + '/doppler_tresh_mov.npy')
    #===============================================================
    # plotFuction(velocity_fft, tresh_plane, doppler_tresh)
    # doppler_tresh = doppler_tresh.astype(int)
    # AoA = np.multiply(velocity_fft, doppler_tresh)
    # AoA = np.pad(AoA, pad_width=n_pad, mode='constant', constant_values=0)
    # AoA = np.complex64(AoA)
    # print(AoA.shape)
    # angleFFT(AoA)


    #===============================================================
    fft_aoa_tresh = np.load(save_dir + '/fft_aoa_tresh.npy')
    
    app = QtGui.QApplication(sys.argv)
    mw = QtGui.QMainWindow()
    mw.resize(800, 800)
    view = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(view)
    mw.setWindowTitle('pyqtgraph example: ScatterPlot')
    w1 = view.addPlot()
    time = QtCore.QTime.currentTime()
    QtCore.qsrand(time.msec())
    for i in range(100,400,4):
        for j in range(fft_aoa_tresh.shape[2])[30:60]:
            for k in range(fft_aoa_tresh.shape[3])[40:80]:
                if fft_aoa_tresh[i,32,j,k] > 0:
                    sp2 = pg.ScatterPlotItem(x=str(i),y=str(k) ,size=10, pen=pg.mkPen(None), brush='r')
                    w1.addItem(sp2)
    mw.show()
    sys.exit(QtGui.QApplication.exec_())

if __name__ == "__main__":

    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()
    
    main()