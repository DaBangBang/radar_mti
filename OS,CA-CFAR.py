import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import itertools

# should be odd number

training_cell_along_range = 35
training_cell_along_velocity = 11
training_cell_along_range = int(training_cell_along_range/2)
training_cell_along_velocity = int(training_cell_along_velocity/2)
print(training_cell_along_range, training_cell_along_velocity)

guarding_cell_along_range = 21
guarding_cell_along_velocity = 5
guarding_cell_along_range = int(guarding_cell_along_range/2)
guarding_cell_along_velocity = int(guarding_cell_along_velocity/2)
print(guarding_cell_along_range, guarding_cell_along_velocity )

def plotFuction(velocity_fft, tresh_plane, doppler_tresh):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(221, projection='3d')
    X, Y = np.mgrid[:velocity_fft.shape[1], :velocity_fft.shape[2]]
    ax1.plot_surface(X, Y, abs(velocity_fft[0,:,:,0]), cmap='jet', alpha=0.7)
    ax1.plot_surface(X, Y, tresh_plane[0,:,:,0], alpha=1)

    ax2 = fig.add_subplot(222)
    ax2 = plt.imshow(doppler_tresh[0,:,:,0], aspect='auto')
    

    ax3 = fig.add_subplot(223)
    ax3 = plt.imshow(abs(velocity_fft[0,:,:,0]), aspect= 'auto', origin='lower' , cmap='jet', 
         extent=[0, velocity_fft.shape[2]-1, velocity_fft.shape[1]/2., -velocity_fft.shape[1]/2.], interpolation = 'catrom')
    
    plt.show()

    

def kernel(velocity_fft):
    # select weight
    weight = 5
    
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

    doppler_tresh = np.zeros((tresh_plane.shape[0], tresh_plane.shape[1],tresh_plane.shape[2],tresh_plane.shape[3]))
    
    for m, n, o, p in itertools.product(range(doppler_tresh.shape[0]), range(doppler_tresh.shape[1]), 
                                        range(doppler_tresh.shape[2]), range(doppler_tresh.shape[3])):        

        if tresh_plane[m,n,o,p] < peak_value[m,n,o,p]:
            doppler_tresh[m,n,o,p] += 1
    
    return tresh_plane, doppler_tresh                          

def main():
    velocity_fft = np.load('D:/data_signal_MTI/data_ball_circle/dop_66.npy')
    tresh_plane, doppler_tresh = kernel(velocity_fft)
    plotFuction(velocity_fft, tresh_plane, doppler_tresh)

  
if __name__ == "__main__":
    main()