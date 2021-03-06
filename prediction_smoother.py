from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import simdkalman

expect_r_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4.npy'
expect_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4.npy'
label_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/label_z_%4.npy'
robot_r_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4_robot.npy'
robot_z_file = 'D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4_robot.npy'

expect_r = np.load(expect_r_file)
expect_z = np.load(expect_z_file)
expect_r_robot = np.load(robot_r_file)
expect_z_robot = np.load(robot_z_file)
label_z = np.load(label_z_file)

expect_r = expect_r.reshape((30, -1))
expect_z = expect_z.reshape((30, -1))
expect_r_robot = expect_r_robot.reshape((3,-1))
expect_z_robot = expect_z_robot.reshape((3,-1))
label_z = label_z.reshape((30, -1, 3))

expect_r_smooth = savgol_filter(expect_r, window_length = 11, polyorder = 2, axis=1)
expect_z_smooth = savgol_filter(expect_z, window_length = 11, polyorder = 2, axis=1)
expect_r_robot_smooth = savgol_filter(expect_r_robot, window_length = 5, polyorder = 1, axis=1)
expect_z_robot_smooth = savgol_filter(expect_z_robot, window_length = 5, polyorder = 1, axis=1)

expect_r_smooth = expect_r_smooth.reshape(-1)
expect_z_smooth = expect_z_smooth.reshape(-1)
expect_r_robot_smooth = expect_r_robot_smooth.reshape(-1)
expect_z_robot_smooth = expect_z_robot_smooth.reshape(-1)

print(expect_r_smooth.shape, expect_z_smooth.shape)
np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4_smooth', expect_r_smooth)
np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4_smooth', expect_z_smooth)
np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_r_%4_robot_smooth', expect_r_robot_smooth)
np.save('D:/data_signal_MTI/project_util_3/prediction_result/expect_z_%4_robot_smooth', expect_z_robot_smooth)

plt.figure(1)
plt.plot(expect_z_robot_smooth[680:,])
# plt.figure(2)
plt.plot(expect_z_robot[2,:])
plt.show()
