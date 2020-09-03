#%%
import numpy as np
import matplotlib.pyplot as plt
import glob
import natsort
import re

us = 1e-6
GHz = 1e9
MHz = 1e6
m = 1
c = 299792458 * m
k = 1000
mm = 1e-3
cm = 1e-2
RampEndTime = 20 * us
IdleTime = 9 * us
Bandwidth = 3.9995 * GHz
Frequency = 77 * GHz
FrequencySlope = 199.975 * MHz / us
ADCSamples = 256
SamplingRate = 18750 * k
ADCStartTime = 3 * us

ADCSamplingTime = ADCSamples / SamplingRate
ChirpCycleTime = RampEndTime + IdleTime
RealBandwidth = ADCSamplingTime * FrequencySlope
WaveLength = c / Frequency
RangeResolution = c / (2 * RealBandwidth)
CompletePhaseDistance = WaveLength / 2
MaxRange = SamplingRate * c / (2 * FrequencySlope)
Vmax = WaveLength / (4 * ChirpCycleTime)

def fbeat(obj_distance):
  # print(RealBandwidth / ADCSamplingTime * 2 * obj_distance / c)
  return RealBandwidth / ADCSamplingTime * 2 * obj_distance / c

def read_label():
  folder_name = 'D:/data_signal_MTI/data_ball_move_39_label/radar_pos*'
  folder_name = glob.glob(folder_name)
  folder_name = natsort.natsorted(folder_name)
  save_folder = 'D:/data_signal_MTI/data_ball_move_39_real_imag_clean/pos_'

  for f_name in folder_name:
    n_label = np.array(list(map(int, re.findall("\d+", f_name))))
    f_label = np.load(f_name)
    sim_chirp = []
    peak_position = []
    for dis in f_label:
      y_offset = 110
      r = np.sqrt(dis[0,0]**2 + (dis[0,1] - y_offset)**2 + dis[0,2]**2)
      # r = dis[0,2]
      r = r*mm
      r = 2000
      print("range r", r)
      simulate(r, sim_chirp, peak_position)
    sim_chirp = np.array(sim_chirp)
    peak_position = np.array(peak_position)
    # print(sim_chirp.shape, sim_chirp[0])
    print(peak_position.shape, peak_position[0])
    # np.save(save_folder + str(n_label[1]) +'/simulate_radar_1chirp_0pad', sim_chirp)
    # np.save(save_folder + str(n_label[1]) +'/peak_position_1chirp_0pad', peak_position)

def simulate(dis, sim_chirp, peak_position):

  
  print("Real Bandwidth\n", RealBandwidth / GHz, "GHz")
  print("Wavelength\n", WaveLength / mm, "mm")
  print("Range Resolution\n", RangeResolution / cm, "cm")
  print("Complete Phase Distance\n", CompletePhaseDistance / mm, "mm")
  print("Complete Cycles per Range Resolution\n", RangeResolution / CompletePhaseDistance)
  print("ADC Sampling Time\n", ADCSamplingTime / us, "us")
  print("Max Range\n", MaxRange, "m")
  print("Number of cycles at 10cm\n", fbeat(0.1) * ADCSamplingTime)
  print("Max Measurable Velocity (phase diff < pi) \n", Vmax, "mm/s")

  ls = []
  fft_all = []
  print("distance ", dis)
  obj_distance = dis
  # print("Plot of IF if object is at", obj_distance, "m")
  for i in range(ADCSamples):
    t = i / ADCSamples * ADCSamplingTime
    ls.append(np.exp(2 * np.pi * t * 1j * fbeat(obj_distance)))

  n_pad = ((0,0))
  ls = np.pad(ls, pad_width=n_pad, mode='constant', constant_values=0)
  
  random_multiply_peak = np.random.randint(low=1, high=20, size=10)
  # print(random_multiply_peak)
  for j in random_multiply_peak:
    # print(j)
    fft = np.fft.fft(ls)
    fft_modulus = j*abs(fft / ADCSamples)
    fft_all.append(fft_modulus)
  
  fft_all = np.array(fft_all)
  # print(fft_all.shape)
  pos_peak = np.where(fft_modulus == np.max(fft_modulus))
  pos_peak = int(np.array(pos_peak[0]))
  # print(pos_peak)
  # print(pos_peak-2)
  mean_clutter_F = np.mean(fft_modulus[:pos_peak-2])
  mean_clutter_R = np.mean(fft_modulus[pos_peak+2:])
  mean_clutter = (mean_clutter_F+mean_clutter_R)/2
  mean_clutter = np.full((100), mean_clutter)
  # print(mean_clutter.shape)
  sim_chirp.append(fft_all)
  peak_position.append(pos_peak)
  
  print('pos =', pos_peak)
  plt.rcParams['figure.figsize'] = [20, 8]
  plt.subplot(1, 2, 1)
  plt.plot(np.real(ls))
  plt.subplot(1, 2, 2)
  plt.plot(fft_all[0,:])
  # plt.plot(fft_all[1,:])
  # plt.plot(fft_all[2,:])
  # plt.plot(fft_all[3,:])
  # plt.plot(fft_all[4,:])
  # plt.plot(mean_clutter)
  plt.show()
  # np.save(save_folder + str(n_label))

read_label()
