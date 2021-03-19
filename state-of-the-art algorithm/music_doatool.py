import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import doatools.plotting as doaplot
import doatools

MHz = 1e6
us = 1e-6
m = 1
k = 1000
c = 299792458 * m
fs = 13*MHz

pad_multi_range = 0
pad_multi_angle = 0

adc_range = 512
Rx = 8
antenna_array = np.linspace(0,((Rx+(pad_multi_angle*Rx))-1)/2, Rx+(pad_multi_angle*Rx))
aoa_search = np.linspace(-np.pi/2,np.pi/2,3600)
# antenna_array = np.reshape(antenna_array,(-1,1))
print("antenna", antenna_array.shape)

r_pad = (0, pad_multi_range*adc_range)
a_pad = ((0,pad_multi_range*adc_range),(0, pad_multi_angle*Rx))

all_trajectory = 120
range_res = 47.6 / (1+pad_multi_range)
angle_res = 3.1415 / (Rx*(pad_multi_angle+1)-1)

signal_dir = 'D:/data_signal_MTI/project_util_3/signal_all_w_mti_cutoff_12/'
fold_dir = 'D:/data_signal_MTI/project_util_3/10_fold_validation/100%_data/test_data/test_index_fold_'

all_trajectory = 120
M = 100
p = 2

def range_estimation(data_frame):
    data_frame = np.pad(data_frame, pad_width=r_pad, mode='constant', constant_values=0)
    data_frame = np.reshape(data_frame, (-1,1))
    f = np.arange(0, fs, (fs/(data_frame.shape[0])))
    se = np.arange(0,M)
    se = np.reshape(se, (-1,1))
    omega = 2*np.pi*(f/(fs))

    # omega = np.reshape(omega, (-1,1))
    print(omega[-1])
    cov_matrix = data_frame@data_frame.conj().transpose()
    
    # cov_matrix = np.zeros((M,M))
    # for i in range((data_frame.shape[0])-M):
    #     cov_matrix = cov_matrix + data_frame[i:i+M]@data_frame[i:i+M].conj().transpose()
    # print("conv", cov_matrix.shape)
    
    grid = estimation.FarField1DSearchGrid(size=512, unit='deg')
    # print(grid.axes)
    estimator = estimation.MUSIC(rula, wavelength, grid)
    resolved, estimates, sp = estimator.estimate(cov_matrix, 1, 
    return_spectrum=True)
    print('Estimates: {0}'.format(estimates.locations))
    # print('Ground truth: {0}'.format(sources.locations))
    # print(sp)
    doaplot.plot_spectrum({'MUSIC': sp}, grid, use_log_scale=True)


wavelength = 1.0 # Normalized wavelength. Recall that d0 = wavelength / 2.
d0 = wavelength / 2
power_source = 1.0
power_noise = 1.0 # SNR = 0 dB
n_snapshots = 100
ula = model.UniformLinearArray(8, d0)
rula = model.UniformLinearArray(512, 6.27)
sources_1d = model.FarField1DSourcePlacement(
    np.linspace(-np.pi/3, np.pi/3, 5)
)
print("aaa", sources_1d.valid_ranges)
# print(rula.element_locations)
# print("ula", ula.element_locations, ula.size)
# print("range_ula", rula.element_locations, rula.size)


def angle_estimation(data_frame):
    data_frame = np.pad(data_frame, pad_width=a_pad, mode='constant', constant_values=0)
    data_frame = np.swapaxes(data_frame,0,1)
    cov_matrix = data_frame@data_frame.conj().transpose()
    grid = estimation.FarField1DSearchGrid()
    estimator = estimation.MUSIC(ula, wavelength, grid)
    resolved, estimates, sp = estimator.estimate(cov_matrix, 1, 
    return_spectrum=True)
    print('Estimates: {0}'.format(estimates.locations))
    # print('Ground truth: {0}'.format(sources.locations))
    doaplot.plot_spectrum({'MUSIC': sp}, grid, use_log_scale=True)
    


if __name__ == '__main__':
    count = 0
    expect_r = []
    expect_z = []
    for i in range(10):
        test_dir = fold_dir + str(i+1) +'.npy'
        print(test_dir)
        fold_num = np.load(test_dir)
        print(fold_num)
        for j in fold_num[:1]:
            print(j+1)  
            real_name = signal_dir + 'raw_iq_w_mti_' + str(j+1) + '.npy'
            test_data = np.load(real_name)
            for k in range(test_data.shape[0])[:1]:
                psd_range = range_estimation(test_data[k,0,:,0])
                psd_aoa = angle_estimation(test_data[k,0,:,:])
                # range_max = np.argmax(psd_range[:,0])
                
                # if range_max > (50+(pad_multi_range*50)):
                #     expect_r.append(np.nan)
                #     expect_z.append(np.nan)
                #     print("outlier")
                # else:
                #     angle_max = np.argmax(psd_aoa)
                #     actual_range = range_max*range_res
                #     actual_doa =  -1*aoa_search[angle_max]
                #     expect_r.append(actual_range)
                #     expect_z.append(actual_doa)
            print("finish")
    expect_r = np.array(expect_r)
    expect_z = np.array(expect_z)