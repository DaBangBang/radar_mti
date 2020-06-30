import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import sys
import itertools
from scipy import stats
import glob
import natsort
import plotly.graph_objects as go
from scipy import signal
from plotly.subplots import make_subplots

signal_dir = 'D:/data_signal_MTI/data_ball_move_39_real_imag_clean/pos*'
label_name = 'D:/data_signal_MTI/data_ball_move_39_label/radar_pos*'
label_name = glob.glob(label_name)
label_name = natsort.natsorted(label_name)

def cal_snr(range_fft_move_filter_all, peak_position_all, pos_all):

    # ---
    model_result = np.load('D:/data_signal_MTI/data_ball_move_39_graph/result_pad_0.npy')
    complex_result = model_result[:,0,:] + 1j*model_result[:,1,:]
    complex_fft = np.fft.fft(complex_result, axis=1) / complex_result.shape[1]
    complex_modulus = abs(complex_fft)
    plt.figure(1)
    plt.plot(complex_modulus[500,:])
    plt.plot(complex_modulus[200,:])
    plt.show()


    s_t = pos_all.argsort()
    range_fft_move_filter_all = range_fft_move_filter_all[s_t]
    complex_modulus = complex_modulus[s_t]
    peak_position_all =  peak_position_all[s_t]

    snr_all = []
    snr_result = []
    for i in range(0,range_fft_move_filter_all.shape[0]):
        a = range_fft_move_filter_all[i,0, peak_position_all[i]+1, 0]
        mean_clutter_F = np.mean(range_fft_move_filter_all[i, 0, :peak_position_all[i]-2, 0])
        mean_clutter_R = np.mean(range_fft_move_filter_all[i, 0, peak_position_all[i]+2:, 0])
        mean_clutter = (mean_clutter_F+mean_clutter_R)/2
        
        b = complex_modulus[i, 0]
        # mean_clutter_F_r = np.mean(complex_modulus[i, :peak_position_all[i]-2])
        mean_clutter_R_r = np.mean(complex_modulus[i, peak_position_all[i]+2:])
        # mean_clutter_r = (mean_clutter_F_r+mean_clutter_R_r)/2


        snr = a / mean_clutter
        snr_r = b / mean_clutter_R_r
        snr_all.append(snr)
        snr_result.append(snr_r)
        # y.append(scr)
    snr_all = np.array(snr_all)
    snr_result = np.array(snr_result)
    # print(snr_all.shape)
    # print(peak_position_all[6000])
    # print(peak_position_all[425])
    plt.figure(1)
    plt.plot(snr_all)
    plt.plot(snr_result)
    plt.figure(2)
    # plt.plot(range_fft_move_filter_all[6000,0,:,0])
    plt.plot(range_fft_move_filter_all[425,0,:,0])
    plt.plot(range_fft_move_filter_all[30,0,:,0])
    plt.show()

    return snr_all, snr_result

def plot_snr(snr_all, pos_all):
    s_t = pos_all.argsort()
    pos_all = pos_all[s_t]
    # pos_all = pos_all[5:pos_all.shape[0]+3: 20]
    print("pos_all_shape", pos_all.shape)
    snr_stove_1000 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_1000pad_stove.npy')
    snr_moving_1000 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_1000pad_moving_filter.npy')
    snr_fir_1000 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_1000pad_fir.npy')
    snr_iir_1000 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_1000pad_iir.npy')
    # print(snr_fir.shape)
    snr_stove_500 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_500pad_stove.npy')
    snr_moving_500 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_500pad_moving_filter.npy')
    snr_fir_500 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_500pad_fir.npy')
    snr_iir_500 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_500pad_iir.npy')

    snr_stove_100 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_100pad_stove.npy')
    snr_moving_100 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_100pad_moving_filter.npy')
    snr_fir_100 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_100pad_fir.npy')
    snr_iir_100 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_100pad_iir.npy')

    snr_result_0 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_0pad_result.npy')
    snr_stove_0 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_0pad_stove.npy')
    snr_moving_0 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_0pad_moving_filter.npy')
    snr_fir_0 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_0pad_fir.npy')
    snr_iir_0 = np.load('D:/data_signal_MTI/data_ball_move_39_graph/snr_0pad_iir.npy')


    snr_stove_1000 = snr_stove_1000[5:snr_stove_1000.shape[0]-1: 20]
    snr_moving_1000 = snr_moving_1000[5:snr_moving_1000.shape[0]-1: 20]
    snr_fir_1000 = snr_fir_1000[5:snr_fir_1000.shape[0]-1: 20]
    snr_iir_1000 = snr_iir_1000[5:snr_iir_1000.shape[0]-1: 20]

    snr_stove_500 = snr_stove_500[5:snr_stove_500.shape[0]-1: 20]
    snr_moving_500 = snr_moving_500[5:snr_moving_500.shape[0]-1: 20]
    snr_fir_500 = snr_fir_500[5:snr_fir_500.shape[0]-1: 20]
    snr_iir_500 = snr_iir_500[5:snr_iir_500.shape[0]-1: 20] 

    
    snr_stove_100 = snr_stove_100[5:snr_stove_100.shape[0]-1: 20]
    snr_moving_100 = snr_moving_100[5:snr_moving_100.shape[0]-1: 20]
    snr_fir_100 = snr_fir_100[5:snr_fir_100.shape[0]-1: 20]
    snr_iir_100 = snr_iir_100[5:snr_iir_100.shape[0]-1: 20] 

        
    # snr_stove_0 = snr_stove_0[5:snr_stove_0.shape[0]-1: 20]
    # snr_moving_0 = snr_moving_0[5:snr_moving_0.shape[0]-1: 20]
    # snr_fir_0 = snr_fir_0[5:snr_fir_0.shape[0]-1: 20]
    # snr_iir_0 = snr_iir_0[5:snr_iir_0.shape[0]-1: 20] 

    # x = np.linspace(0,1,snr_moving_1000.shape[0])
    fig = make_subplots(rows=4, cols=1)


    # fig.add_trace(go.Scatter(x=pos_all, y=snr_moving,
    #                 mode='lines+markers',
    #                 name='moving_r'),  row = 1, col = 1)
    # fig.add_trace(go.Scatter(x=pos_all, y=snr_stove,
    #             mode='lines+markers',
    #             name='stove_r'),  row = 1, col = 1)
    # fig.add_trace(go.Scatter(x=pos_all, y=snr_fir_1000,
    #                 mode='lines+markers',
    #                 name='fir_r'),  row = 1, col = 1)
    # fig.add_trace(go.Scatter(x=pos_all, y=snr_iir,
    #             mode='lines+markers',
    #             name='iir_r'),  row = 1, col = 1)
   

    # fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_moving_1000,59,3),
    #                 mode='lines+markers',
    #                 name='moving'), row = 1, col = 1)
    # fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_stove_1000,59,3),
    #             mode='lines+markers',
    #             name='stove'),  row = 1, col = 1)
    # fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_fir_1000,59,3),
    #                 mode='lines+markers',
    #                 name='fir'),  row = 1, col = 1)
    # fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_iir_1000,59,3),
    #             mode='lines+markers',
    #             name='iir'),  row = 1, col = 1)
   
    
    fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_moving_500,59,3),
                    mode='lines+markers',
                    name='moving'), row = 2, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_stove_500,59,3),
                mode='lines+markers',
                name='stove'),  row = 2, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_fir_500,59,3),
                    mode='lines+markers',
                    name='fir'),  row = 2, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_iir_500,59,3),
                mode='lines+markers',
                name='iir'),  row = 2, col = 1)
    
    
    fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_moving_100,59,3),
                    mode='lines+markers',
                    name='moving'), row = 3, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_stove_100,59,3),
                mode='lines+markers',
                name='stove'),  row = 3, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_fir_100,59,3),
                    mode='lines+markers',
                    name='fir'),  row = 3, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=signal.savgol_filter(snr_iir_100,59,3),
                mode='lines+markers',
                name='iir'),  row = 3, col = 1)

    
    fig.add_trace(go.Scatter(x=pos_all, y=snr_moving_0,
                    mode='lines+markers',
                    name='moving'), row = 4, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=snr_stove_0,
                mode='lines+markers',
                name='stove'),  row = 4, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=snr_fir_0,
                    mode='lines+markers',
                    name='fir'),  row = 4, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=snr_iir_0,
                mode='lines+markers',
                name='iir'),  row = 4, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=snr_result_0,
                mode='lines+markers',
                name='iir'),  row = 4, col = 1)

    
    fig.add_trace(go.Scatter(x=pos_all, y=snr_moving_0,
                    mode='lines+markers',
                    name='moving'), row = 4, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=snr_stove_0,
                mode='lines+markers',
                name='stove'),  row = 4, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=snr_fir_0,
                    mode='lines+markers',
                    name='fir'),  row = 4, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=snr_iir_0,
                mode='lines+markers',
                name='iir'),  row = 4, col = 1)
    fig.add_trace(go.Scatter(x=pos_all, y=snr_result_0,
                mode='lines+markers',
                name='iir'),  row = 4, col = 1)



   
    fig['layout'].update(height=1920, width=1920)
    # fig.update_yaxes(range=[-0.5, 140])
    fig.update_layout()    
    fig.show()

    # print(x.shape)
    # print(x)
    # print(pos_all)

def main():

    mm = 1e-3
    pos_all = []
    count = 0
    for label in label_name:
        count += 1
        f_label = np.load(label)
        
        if count%4 == 0:
            for dis in f_label:
                y_offset = 110
                r = np.sqrt(dis[0,0]**2 + (dis[0,1] - y_offset)**2 + dis[0,2]**2)
                # r = dis[0,2]
                r = r*mm
                pos_all.append(r)
        else:
            print("training_set : pass")

    pos_all = np.array(pos_all)
    print(np.array(pos_all).shape)

    folder_name = glob.glob(signal_dir)
    folder_name = natsort.natsorted(folder_name)
    range_fft_move_filter_all = []
    peak_position_all = []
    count = 0

    for f_name in folder_name:
        count += 1
        if count%4 == 0:
            move_filter = f_name + '/range_fft_zero_pad_0_iir.npy'
            peak_pos = f_name + '/peak_position_1chirp_0pad.npy'
            range_fft_move_filter = np.load(move_filter)
            peak_position = np.load(peak_pos)
            range_fft_move_filter_all.extend(abs(range_fft_move_filter))
            peak_position_all.extend(peak_position)
        else:
            print("training_data : pass")

    # print(np.array(range_fft_move_filter_all).shape, np.array(peak_position_all).shape)
    # snr_all, snr_result = cal_snr(np.array(range_fft_move_filter_all), np.array(peak_position_all), pos_all)
    # np.save('D:/data_signal_MTI/data_ball_move_39_graph/snr_0pad_iir', snr_all)
    # np.save('D:/data_signal_MTI/data_ball_move_39_graph/snr_0pad_result', snr_result)
    snr_all = 0 
    plot_snr(snr_all, pos_all)       

if __name__ == '__main__':
    main()

    # snr_all = 0
    # plot_snr(snr_all)